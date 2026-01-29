import os
import re
import gc
import time
import regex
import pstats
import logging
import cProfile
import datetime
import multiprocessing as mp
from functools import partial
from collections import defaultdict

from typing import TypeAlias
from typing import BinaryIO


ByteTuple:TypeAlias = tuple[bytes, ...]

class BPETokenizer:
    def __init__(
            self, 
            train_data_path:str='',
            special_tokens:list[str]=[],
            vocab_size:int=256,
            num_processes:int=8,
            batch_num:int=4):
        self.train_data_path = train_data_path
        self.train_samples:list[str] = []
        self.train_samples_new:list[str] = []
        
        self.vocab_size = vocab_size
        self.vocab:dict[bytes,int] = {}
        self.r_vocab:dict[bytes,int] = {}
        self.special_tokens = special_tokens
        self.split_token = b"<|endoftext|>"
        self.num_processes = num_processes
        self.batch_num = batch_num

        self.new_words_freq:dict[
                bytes, 
                int] = defaultdict(int)
        self.words_freq:dict[
                ByteTuple, 
                int] = defaultdict(int)
        self.pair_freq:dict[
                ByteTuple,
                int] = defaultdict(int)
        self.pair_word_d:dict[
                ByteTuple,
                set(ByteTuple)] = defaultdict(set)
        self.word_pair_d:dict[
                ByteTuple,
                set(ByteTuple)] = defaultdict(set)
        self.merges:list[ByteTuple] = []
        self.setup_logging()
    
    def setup_logging(self):
        self.logger = \
                logging.getLogger('BPEProcess')
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return
        formatter = logging.Formatter(
                '%(asctime)s '
                '%(levelname)s:'
                '%(message)s',
                datefmt='%H:%M:%S'
        )
        file_handler = logging.FileHandler(\
                './log/bpe_train.log', 
                encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("setup_logging: log init sucess")

    def add_special_tokens(self):
        for token in self.special_tokens:
            e_token = token.encode('utf-8')
            if e_token not in self.vocab:
                self.vocab[e_token] = len(self.vocab)
    
    def init_vocab(self):
        self.vocab= {bytes([x]):x \
                        for x in range(256)}
        self.add_special_tokens()

    def split_by_special_tokens(
            self,
            text:str,
            special_tokens:list[str]
        ) -> list[str]:
        if not special_tokens:
            return [text]
        escaped_tokens = [ re.escape(token) 
              for token in special_tokens
            ]
        pattern = '|'.join(escaped_tokens)
        splits = re.split(pattern, text)
        result = [x for x in splits if x.strip()]
        return result
    
    def find_chunk_boundaries(
            self,
            chunk_num:int,
            split_token:bytes,
            file:BinaryIO)->list[int]:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // chunk_num
        boundaries = [\
                x * chunk_size \
                for x in range(chunk_num+1)
            ]
        boundaries[-1] = file_size
        
        min_chunk_size = 4096
        for x in range(1, len(boundaries)):
            cur_pos = boundaries[x]
            file.seek(cur_pos)
            while True:
                min_chunk = file.read(min_chunk_size)
                if min_chunk == b"":
                    boundaries[x] = file_size
                    break
                sptoken_pos = min_chunk.find(split_token)
                if sptoken_pos != -1:
                    boundaries[x] = cur_pos + sptoken_pos
                    break
                cur_pos += min_chunk_size
        
        boundaries=sorted(set(boundaries))
        return list(zip(
            boundaries[:-1], boundaries[1:]))
    
    def calc_gpt2_words(
            self, 
            train_text:str
            )->dict[ByteTuple,int]:
        pattern = (
            r"'(?:[sdmt]|ll|ve|re)" +
            r"| ?\p{L}+" +
            r"| ?\p{N}+" +
            r"| ?[^\s\p{L}\p{N}]+" +
            r"|\s+(?!\S)" +
            r"|\s+"
        )

        train_samples = []
        splits = self.split_by_special_tokens(
                        train_text,
                        self.special_tokens
                    )
        train_samples.extend(splits)
        
        words_freq:dict[ByteTuple:int] \
                = defaultdict(int)

        for sample in train_samples:
            gpt2_words = regex.findall(
                    pattern, 
                    sample, 
                    regex.UNICODE)
            for word in gpt2_words:
                word = tuple(bytes([x])\
                    for x in word.encode('utf-8'))
                words_freq[word] += 1
        
        return words_freq

    def mp_pre_tokenizer(self): 
        self.logger.info("mp_pre_tokenizer: "
                    "mp_pre_tokenizer process start")
        with open(self.train_data_path, "rb") as file:
            batch_num = self.batch_num
            v_proc_num = 2 * self.num_processes
            chunk_num = batch_num * v_proc_num
            chunk_boundaries = \
                    self.find_chunk_boundaries(
                    chunk_num,
                    self.split_token,
                    file
                    )
            self.logger.info("mp_pre_tokenizer: "
                    f"batch_vproc_chunk_bd_num:"
                    f"[{batch_num}|"
                    f"{v_proc_num}|"
                    f"{chunk_num}|"
                    f"{len(chunk_boundaries)}]")

            self.logger.info("mp_pre_tokenizer: "
                    "read data and process begin")
            start_time=time.time()
            for idx in range(0, chunk_num, v_proc_num):
                batch_boundaries = \
                    chunk_boundaries[idx:idx+v_proc_num] 
                self.logger.info("mp_pre_tokenizer: "
                    f"read data batch-{idx//v_proc_num}")
                def read_chunks_generator():
                    for start , end in batch_boundaries:
                        file.seek(start)
                        chunk_size = end - start
                        chunk_text = \
                            file.read(chunk_size\
                            ).decode('utf-8')
                        yield chunk_text
                self.logger.info("mp_pre_tokenizer: "
                    f"process batch-{idx//v_proc_num} begin")
                with mp.Pool(processes = \
                    self.num_processes) as pool:
                    for result in pool.imap_unordered(
                            self.calc_gpt2_words, 
                            read_chunks_generator()):
                        for word,freq in result.items():
                            self.words_freq[word] += freq
                        del result
                self.logger.info("mp_pre_tokenizer: "
                    f"process batch-{idx//v_proc_num} complete")
            end_time=time.time()
            self.logger.info("mp_pre_tokenizer: "
                f"pre-tokenizer complete ["
                f"ct:{(end_time-start_time):.3f}s]")
    
    def find_max(
                self,
                target_words:set[ByteTuple],
                pre_m_bytes:bytes
        )->ByteTuple:
        for word in target_words:
            for x in range(len(word) - 1):
                if pre_m_bytes !=b'' \
                    and word[x] != pre_m_bytes \
                    and word[x+1] != pre_m_bytes:
                    continue
                byte_pair = (word[x], word[x+1])
                self.pair_freq[byte_pair] += \
                        self.words_freq[word]
                self.pair_word_d[byte_pair].add(word)
                self.word_pair_d[word].add(byte_pair)
        
        max_freq = -1
        max_freq_pair = None
        for pair, freq in self.pair_freq.items():
            if freq > max_freq or \
                (freq == max_freq and \
                    pair > max_freq_pair):
                max_freq, max_freq_pair = freq, pair
        
        return max_freq_pair
    
    def calc_new_word(
            self,
            freq:int,
            word:ByteTuple,
            max_freq_pair:ByteTuple)->tuple:
        x = 0
        new_word = ()
        while x < len(word):
            if x == len(word)-1:
                new_word += (word[x],)
                x += 1
                continue
            byte_pair = (word[x], word[x+1])
            if byte_pair == max_freq_pair:
                new_word += (word[x] + word[x+1],)
                if x - 1 >= 0:
                    l_pair = (word[x-1], word[x])
                    self.pair_freq[l_pair] -= freq
                if x + 2 < len(word):
                    r_pair = (word[x+1], word[x+2])
                    self.pair_freq[r_pair] -= freq
                x += 2
            else:
                new_word += (word[x],)
                x += 1
        return new_word
    
    def update_words_dict(
            self,
            word:ByteTuple,
            new_word:ByteTuple,
            max_freq_pair:ByteTuple):
        freq = self.words_freq[word]
        self.words_freq[new_word] = freq
        self.words_freq.pop(word)
        for pair in self.word_pair_d[word]:
            self.pair_word_d[pair].remove(word)
            self.pair_word_d[pair].add(new_word)
        self.word_pair_d.pop(word)
        
        for x in range(len(new_word)-1):
            pair = (new_word[x], new_word[x+1])
            self.word_pair_d[new_word].add(pair)
        
        for x in range(len(word) - 1):
            pair = (word[x], word[x+1])
            found = False
            for y in range(len(new_word) - 1):
                new_pair = (new_word[y], 
                            new_word[y+1])
                if pair == new_pair:
                    found = True
                    break
            if found == True:
                continue
            if new_word in self.pair_word_d[pair]:
                self.pair_word_d[pair].remove(new_word)

    def update(
            self, 
            max_freq_pair:ByteTuple
            )->set[ByteTuple]:
        target_words = \
            self.pair_word_d[max_freq_pair].copy()
        new_t_words = target_words.copy()
        for word in target_words:
            freq = self.words_freq[word]
            new_word = self.calc_new_word(
                    freq, 
                    word, 
                    max_freq_pair)
            if new_word == word:
                continue
            new_t_words.remove(word)
            new_t_words.add(new_word)
            self.update_words_dict(
                    word, 
                    new_word, 
                    max_freq_pair)
        self.pair_freq.pop(max_freq_pair)

        return new_t_words

    def merge(self):
        merge_bytes = b''
        target_words = self.words_freq.keys()
        self.logger.info('merge: bpe merge process start')
        start_time = time.time()
        while len(self.vocab) < self.vocab_size:
            max_freq_pair = \
                self.find_max(
                        target_words,
                        merge_bytes)
            if len(max_freq_pair) < 2:
                break
            self.merges.append(max_freq_pair)
            merge_bytes = \
                    max_freq_pair[0] +\
                    max_freq_pair[1]
            if merge_bytes not in self.vocab:
                self.vocab[merge_bytes] = len(self.vocab)
                target_words = self.update(max_freq_pair)
            if len(self.vocab) % 1000 == 0:
                self.logger.info("merge: "
                    f"merge_step:[{len(self.vocab)}:"
                    f"{merge_bytes}]")
        self.r_vocab = \
                {y:x for (x,y) in self.vocab.items()}
        end_time = time.time()
        self.logger.info("merge: "
                f"merge process complete ["
                f"ct:{(end_time-start_time):.3f}s]")
        self.logger.info("merge: "
                f"final_vocab_size:{len(self.vocab)}")

    def run(self):
        self.init_vocab()
        self.mp_pre_tokenizer()
        self.merge()

    def validate(self):
        pass

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        **kwargs
        )->tuple[dict[int, bytes],
                 list[tuple[bytes, bytes]]]:
    
    bpe_tokenizer = BPETokenizer(
        input_path,
        special_tokens,
        vocab_size)
    start_time = time.time()
    bpe_tokenizer.run()
    end_time = time.time()
    bpe_tokenizer.logger.info(
        f"train bpe complete ["
        f"ct:{(end_time-start_time):.3f}s]")

    return (bpe_tokenizer.r_vocab,
                bpe_tokenizer.merges)


def output(
        output_file:str,
        vocab:dict[int, bytes],
        vocab_size:int,
        merges:list[tuple[bytes,bytes]]
        ):
    with open(output_file, 'w') as f:
        for idx,token in vocab.items():
            print(idx, token, file = f)

if __name__ == "__main__":  

    input_path = './data/owt_train.txt'
    output_file = './output/owt_vocab.out'
    #input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    #input_path = "./tests/fixtures/corpus.en"
    #vocab_size = 10000
    #vocab_size=512
    vocab_size = 32000
    special_tokens =["<|endoftext|>"]
    vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
    output(
        output_file = output_file,
        vocab = vocab,
        vocab_size=vocab_size,
        merges=merges)
