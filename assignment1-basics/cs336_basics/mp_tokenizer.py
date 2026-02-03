import os
import re
import ast
import time
import regex
import cProfile
from typing import Iterator
from typing import Iterable
from typing import TypeAlias

from memory_profiler import profile

from collections import defaultdict

ByteTuple:TypeAlias = tuple[bytes, ...]

class WordData:
    def __init__(self):
        self.word_pair_d:\
            dict[ByteTuple:set[ByteTuple]]\
            = defaultdict(set)
        self.pair_word_d:\
            dict[ByteTuple:set[ByteTuple]]\
            = defaultdict(set)
        self.words_idx:\
            dict[ByteTuple:set[int]] \
            = defaultdict(set)
class BPETokenizer:
    def __init__(
            self, 
            vocab:dict[int,bytes],
            merges:list[ByteTuple],
            special_tokens=list[str]|None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.r_vocab:dict[bytes,int] = \
                    defaultdict(dict)
        for _id, word in self.vocab.items():
            self.r_vocab[word] = _id

    @classmethod
    def from_files(
            cls,
            vocab_filepath:str,
            merges_filepath:str,
            special_tokens:list[str]|None=None):
        if special_tokens is None:
            special_tokens = []
        vocab = cls._read_vocab_file(vocab_filepath)
        merges = cls._read_merges_file(merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _read_vocab_file(
            filepath:str)->dict[int,bytes]:
        vocab:dict[int,bytes] = {}
        with open(filepath,'r',encoding='utf-8') as f:
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
    def _parse_merge_pair(
            line:str)->ByteTuple|None:
        # format (b'left', b'right')
        if not (line.startswith('(') and \
                line.endswith(')')):
            return None
        pair = ast.literal_eval(line)
        if not (isinstance(pair, tuple) and \
                len(pair) == 2):
            return None
        left, right = pair
        if isinstance(left, bytes):
            left_bytes = left
            right_bytes = right
        else:
            left_bytes = str(left).encode()
            right_byges = str(right).encode()
        return (left_bytes, right_bytes)
    
    @staticmethod
    def _read_merges_file(
            filepath:str)->list[ByteTuple]:
        merges:list[ByteTuple] = []
        with open(filepath,'r',encoding='utf-8') as f:
            for line in f:
                pair = BPETokenizer._parse_merge_pair(
                        line.strip())
                if pair is not None:
                    merges += [pair]
        return merges
    
    @staticmethod
    def _calc_gpt2_words(
            text:str,
            special_tokens:list[str]
            )->list[ByteTuple]:

        pattern = (
                r"'(?:[sdmt]|ll|ve|re)" +
                r"| ?\p{L}+" +
                r"| ?\p{N}+" +
                r"| ?[^\s\p{L}\p{N}]+" +
                r"|\s+(?!\S)" +
                r"|\s+"
        )
        words:ByteTuple = []
        _words:dict = {}
        splits = BPETokenizer._split_by_tokens(
                text,
                special_tokens)
        for split in splits:
            if special_tokens is not None and\
                    split in special_tokens:
                words += [(split.encode('utf-8'),)]
                continue
            #print(split)
            #print("---------")
            gpt2_words = regex.findall(
                    pattern,
                    split,
                    regex.UNICODE)
            #print(gpt2_words)
            for word in gpt2_words:
                word_byte = tuple(bytes([x])\
                    for x in word.encode('utf-8'))
                _words[word_byte]=0
                #words += [word_byte]
        return words
    
    @staticmethod
    def _split_by_tokens(
            text:str,
            tokens:list[str])->list[str]:
        if not tokens:
            return [text]
        tokens = sorted(
                tokens, 
                key=len, 
                reverse=True)
        escaped_tokens = [re.escape(token)
                    for token in tokens
        ]
        pattern = f'({"|".join(escaped_tokens)})'
        splits = re.split(pattern, text)
        return splits

    @staticmethod
    def _calc_new_word(
            word:ByteTuple,
            merge:ByteTuple)->ByteTuple:
        x = 0
        new_word:ByteTuple = ()
        while x < len(word):
            if x == len(word) - 1:
                new_word += (word[x],)
                x += 1
                break
            pair_bytes = word[x] + word[x+1]
            merge_bytes = merge[0] + merge[1]
            if pair_bytes == merge_bytes:
                new_word += (merge_bytes,)
                x += 2
            else:
                new_word += (word[x],)
                x += 1
        return new_word
    @staticmethod
    def _calc_word_pair(
            word:ByteTuple)->set[ByteTuple]:
        pairs:set[ByteTuple] = set()
        for x in range(len(word)-1):
            pair = (word[x], word[x+1])
            pairs.add(pair)
        return pairs
    
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
            #pair_word_d[pair].add(new_word)
        word_pair_d.pop(word)
        new_pairs = \
            BPETokenizer._calc_word_pair(new_word)
        word_pair_d[new_word] = new_pairs
        for pair in new_pairs:
            pair_word_d[pair].add(new_word)

    def merge(
            self, 
            merge:ByteTuple,
            word_data:WordData):
        _pair_word = word_data.pair_word_d[merge].copy()
        for word in _pair_word:
            new_word = BPETokenizer._calc_new_word(
                    word, 
                    merge)
            if word == new_word:
                continue
            BPETokenizer._update_word_data(
                    word, new_word, word_data) 
    
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
                BPETokenizer._calc_word_pair(word)
            word_pair_d[word] = pairs
            for pair in pairs:
                pair_word_d[pair].add(word)

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
        cunt = 1
        for merge in self.merges: 
            self.merge(merge, word_data)
            cunt+=1
            print("merge round compliete",cunt)
        for word, ids in \
                word_data.words_idx.items():
            for _id in ids:
                new_words[_id] = word
        return new_words

    def vocab_lookup(self, word:bytes)->int:
        if word in self.r_vocab:
            return self.r_vocab[word]
        raise ValueError(
            f"token {word} not in vocabulary.")
        return -1

    def encode(self, text:str)->list[int]:
        profiler = cProfile.Profile()
        profiler.enable()
        
        words = BPETokenizer._calc_gpt2_words(
                text, self.special_tokens)
        new_words = self.apply_merges(words)
        ids = []
        for word in new_words:
            for m_byte in word:
                _id = self.vocab_lookup(m_byte)
                ids += [_id]
        
        profiler.disable()
        profiler.print_stats(sort='cumtime')

        return ids
    
    @profile
    def encode_iterable(
            self, 
            iterable:Iterable[str]
            )->Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for _id in ids:
                yield _id

    def decode(self, ids:list[int])->str:
        tokens = [self.vocab[_id] for _id in ids]
        text = b"".join(tokens).decode(
                    "utf-8",errors="replace")
        return text

    def mp_encode(self):
        with open(self.input_file, "rb") as file:
            batch_num = self.batch_num
            v_proc_num = 2 * self.num_processes
            chunk_num = batch_num * v_proc_num
            chunk_boundaries = \
                    self.find_chunk_boundaries(
                        chunk_num,
                        self.split_token,
                        file
                    )
            for idx in range(0, chunk_num, v_proc_num):
                batch_boundaries = \
                        chunk_boundaries[idx:idx+v_proc_num]
                def read_chunks_generator():
                    for start , end in batch_boundaries:
                        file.seek(start)
                        chunk_size = end - start
                        chunk_text = \
                                file.read(chunk_size\
                                ).decode('utf-8')
                        yield chunk_text
                with mp.Pool(processes = \
                        self.num_processes) as pool:
                    for result in pool.imap_unordered(
                            self.calc_gpt2_words,
                            read_chunks_generator()):

def output(
        encode_file:str,
        decode_file:str,
        encode_ids:list[int],
        decode_text:str
        ):
    with open(encode_file, 'w') as f:
        print(encode_ids, file = f)
    with open(decode_file, 'w') as f:
        print(decode_text, file = f)


if __name__ == '__main__':
    vocab_file = './output/ts.vocab'
    merges_file = './output/ts.merges'
    special_tokens =["<|endoftext|>"]
    #special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    tokenizer = BPETokenizer.from_files(
            vocab_file,
            merges_file,
            special_tokens)
    input_file = "./data/TinyStoriesV2-GPT4-train.txt"
    #input_file ="./tests/fixtures/tinystories_sample.txt"
    #input_file ="./tests/fixtures/tinystories_sample_5M.txt"
    encode_file = './output/ts.e'
    decode_file = './output/ts.d'
    t1=time.time()
    with open(input_file,'r',encoding='utf-8') as f:
        sample_text = f.read()
    print("read_data complet",time.time()-t1)
    #with open("./tests/fixtures/tinystories_sample_5M.txt") as f:
    #    ids = []
    #    print(f)
   #     for _id in tokenizer.encode_iterable(f):
            #print(_id,tokenizer.decode([_id]))
    #        ids.append(_id)
    #print(tokenizer.decode(ids) == sample_text)
    
    #sample_text = "Hello, how are you?"
    #sample_text = ""
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)
    output(encode_file,decode_file,encoded_ids,decoded_text)
    #tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    #print(tokenized_string)
    #print(sample_text == decoded_text)
    #output(encode_file, decode_file, encoded_ids, decoded_text)
    #print(decoded_text)
    #print(sample_text)
    #print(f"Encoded Ids: {encoded_ids}")
    #print(f"Decoded Text: {decoded_text}")
    t2=time.time()
    #print(encoded_ids)
    print(t2-t1)
