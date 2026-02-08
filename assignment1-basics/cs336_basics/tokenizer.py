import re
import ast
import json
import time
import regex
import multiprocessing as mp
from typing import Iterator
from typing import Iterable
from typing import TypeAlias
from collections import defaultdict
from utils.logger import Logger
from utils.data import WordData
from utils.data import StreamingNpyWriter
from utils.data import MultiNpyReader
from utils.data import ByteTuple
from utils.config import conf
from utils.utils import *

class BPETokenizer:
    def __init__(
            self, 
            vocab:dict[int,bytes],
            merges:list[ByteTuple],
            special_tokens=list[str]|None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = \
                special_tokens
        self.r_vocab:dict[bytes,int] = \
                    defaultdict(dict)
        for _id, word in self.vocab.items():
            self.r_vocab[word] = _id
    
    @classmethod
    def from_files(
            cls,
            vocab_filepath:str,
            merges_filepath:str,
            special_tokens:list[str]):
        if special_tokens is None:
            special_tokens = []
        vocab = cls._read_vocab_file(\
                vocab_filepath)
        merges = cls._read_merges_file(\
                merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _read_vocab_file(
            filepath:str)->dict[int,bytes]:
        vocab:dict[int,bytes] = {}
        with open(filepath,'r',\
                encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ',1)
                if len(parts) != 2:
                    continue
                _id = int(parts[0].strip())
                token_str = parts[1].strip()
                token = ast.literal_eval(token_str)
                vocab[_id] = token
        return vocab
    
    @staticmethod
    def _read_merges_file(
            filepath:str)->list[ByteTuple]:
        merges:list[ByteTuple] = []
        with open(filepath,'r',\
                encoding='utf-8') as f:
            for line in f:
                pair = parse_merge_pair(
                        line.strip())
                if pair is not None:
                    merges += [pair]
        return merges
    
    @staticmethod
    def _init_word_data(
            words:list[ByteTuple],
            word_data:WordData
            ):
        words_idx = word_data.words_idx
        word_pair_d = word_data.word_pair_d
        pair_word_d = word_data.pair_word_d
        for x , word in enumerate(words):
            words_idx[word].add(x)
            if word in word_pair_d:
                continue
            pairs = \
                calc_word_pair(word)
            word_pair_d[word] = pairs
            for pair in pairs:
                pair_word_d[pair].add(word)
   
    @staticmethod
    def _update_word_data(
            word:ByteTuple,
            new_word:ByteTuple,
            word_data:WordData
            ):
        words_idx = word_data.words_idx
        word_pair_d = word_data.word_pair_d
        pair_word_d = word_data.pair_word_d
        
        words_idx[new_word] = words_idx[word]
        words_idx.pop(word)
        for pair in word_pair_d[word]:
            pair_word_d[pair].remove(word)
        word_pair_d.pop(word)
        new_pairs = \
            calc_word_pair(new_word)
        word_pair_d[new_word] = new_pairs
        for pair in new_pairs:
            pair_word_d[pair].add(new_word)

    def merge(
            self, 
            merge:ByteTuple,
            word_data:WordData):
        _pair_word = \
            word_data.pair_word_d[merge].copy()
        for word in _pair_word:
            new_word = calc_new_word(
                    word, 
                    merge)
            if word == new_word:
                continue
            BPETokenizer._update_word_data(
                    word, new_word, word_data) 

    def apply_merges(
            self,
            words:list[ByteTuple]
            )->list[ByteTuple]:
        
        word_data = WordData()
        new_words = list(words)
        BPETokenizer._init_word_data(
                words,
                word_data
                )
        for merge in self.merges: 
            self.merge(merge, word_data)
        for word, ids in \
                word_data.words_idx.items():
            for _id in ids:
                new_words[_id] = word
        return new_words

    def vocab_lookup(
            self, 
            word:bytes)->int:
        if word in self.r_vocab:
            return self.r_vocab[word]
        raise ValueError(
            f"token {word} not in vocabulary.")
        return -1

    def encode(
            self, 
            text:str)->list[int]:
        words = calc_gpt2_words_v2(
                text, self.special_tokens)
        new_words = self.apply_merges(words)
        ids = []
        for word in new_words:
            for m_byte in word:
                _id = self.vocab_lookup(m_byte)
                ids += [_id]
        return ids
    
    def encode_iterable(
            self, 
            iterable:Iterable[str]
            )->Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for _id in ids:
                yield _id

    def decode(
            self, 
            ids:list[int])->str:
        tokens = [self.vocab[_id] \
                for _id in ids]
        text = b"".join(tokens).decode(
                "utf-8", errors="replace")
        return text
    
    def decode_from_file(
            self,
            encode_dir:str,
            chunk_size:int
            )->Iterator[str]:
        reader = MultiNpyReader(encode_dir)
        for data in reader.read(
                chunk_size=chunk_size):
            yield tokenizer.decode(data)

    def encode_mp(
            self,
            input_file:str,
            output_dir:str,
            batch_num:int,
            num_workers:int,
            chunk_size:int,
            split_token:bytes
        ):
        logger = Logger(
                'BPETokenizerProcess',
                './log/tokenizer.log')
        logger = logger.logger.info
        writer = StreamingNpyWriter(
                    output_dir=output_dir, 
                    chunk_size=chunk_size)
        with open(input_file, "rb") as infile:
            v_proc_num = 4 * num_workers
            chunk_num = batch_num * v_proc_num
            chunk_boundaries = \
            find_chunk_boundaries(
                        chunk_num,
                        split_token,
                        infile
                    )
            logger("mp_process: "
                    f"batch_vproc_chunk_bd_num:"
                    f"[{batch_num}|"
                    f"{v_proc_num}|"
                    f"{chunk_num}|"
                    f"{len(chunk_boundaries)}]")
            logger("mp_process: "
                "read data and process begin")
            start_time = time.time()
            token_cunt = 0
            for x in range(0, chunk_num, v_proc_num):
                _start_time = time.time()
                batch_boundaries = \
                    chunk_boundaries[x:x+v_proc_num]
                logger("mp_process: "
                f"process batch-{x//v_proc_num} begin")
                def chunk_generator():
                    for idx, (start, end) in \
                        enumerate(batch_boundaries):
                        infile.seek(start)
                        chunk_text = \
                            infile.read(end-start)\
                            .decode('utf-8')
                        yield chunk_text
                with mp.Pool(processes = num_workers) \
                        as pool:
                    for token_ids in pool.imap(
                            self.encode, 
                            chunk_generator()):
                        writer.add(token_ids)
                        token_cunt+=len(token_ids)
                        logger("mp_process: "
                            f"encoded tokens "
                            f"[size:{token_cunt}]")
                _end_time = time.time()
                logger("mp_process: "
                f"process batch-{x//v_proc_num} complete "
                f"[ct:{(_end_time-_start_time):.3f}s]")
            writer.finish()
            end_time = time.time()
            logger("mp_process: "
                f"mp_process all complete "
                f"[ct:{(end_time-start_time):.3f}s]")

if __name__ == '__main__':
    sample_text = "Hello, how are you?"
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)
    print(sample_text == decoded_text)
    print(f"Encoded Ids: {encoded_ids}")
    print(f"Decoded Text: {decoded_text}")
