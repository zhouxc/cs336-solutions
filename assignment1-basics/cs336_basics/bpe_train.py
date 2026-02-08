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
from utils.data import WordData
from utils.data import ByteTuple
from utils.config import conf
from utils.utils import *
from utils.logger import Logger

class BPETrainer:
    def __init__(
            self, 
            train_data_path:str='',
            special_tokens:list[str]=[],
            vocab_size:int=256,
            num_workers:int=8,
            batch_num:int=4):
        self.train_data_path = train_data_path
        
        self.vocab_size = vocab_size
        self.vocab:dict[bytes,int] = {}
        self.r_vocab:dict[bytes,int] = {}
        self.merges:list[ByteTuple] = []
        self.special_tokens = special_tokens
        self.split_token = b"<|endoftext|>"
        self.num_workers = num_workers
        self.batch_num = batch_num
        word_data = WordData()
        self.words_freq = word_data.words_freq
        self.pair_freq = word_data.pair_freq
        self.pair_word_d = word_data.pair_word_d
        self.word_pair_d = word_data.word_pair_d
        self.logger = Logger(
            'BPETrainProcess',
            './log/bpe_train.log').logger.info

    def add_special_tokens(self):
        for token in self.special_tokens:
            e_token = token.encode('utf-8')
            if e_token not in self.vocab:
                self.vocab[e_token] = \
                    len(self.vocab)
    
    def init_vocab(self):
        self.vocab= {bytes([x]):x \
                        for x in range(256)}
        self.add_special_tokens()

    def mp_pre_tokenizer(self): 
        self.logger("mp_pre_tokenizer: "
            "mp_pre_tokenizer process start")
        with open(self.train_data_path, \
                "rb") as file:
            batch_num = self.batch_num
            v_proc_num = 2 * self.num_workers
            chunk_num = batch_num * v_proc_num
            chunk_boundaries = \
                    find_chunk_boundaries(
                    chunk_num,
                    self.split_token,
                    file
                    )
            self.logger("mp_pre_tokenizer: "
                    f"batch_vproc_chunk_bd_num:"
                    f"[{batch_num}|"
                    f"{v_proc_num}|"
                    f"{chunk_num}|"
                    f"{len(chunk_boundaries)}]")

            self.logger("mp_pre_tokenizer: "
                    "read data and process begin")
            start_time=time.time()
            for idx in range(0, chunk_num, v_proc_num):
                batch_boundaries = \
                    chunk_boundaries[idx:idx+v_proc_num] 
                self.logger("mp_pre_tokenizer: "
                    f"read data batch-{idx//v_proc_num}")
                def read_chunks_generator():
                    for start , end in batch_boundaries:
                        file.seek(start)
                        chunk_size = end - start
                        chunk_text = \
                            file.read(chunk_size\
                            ).decode('utf-8')
                        yield chunk_text
                self.logger("mp_pre_tokenizer: "
                f"process batch-{idx//v_proc_num} begin")
                calc_words_func = partial(
                        calc_gpt2_words_v1,
                        special_tokens = \
                        self.special_tokens)
                with mp.Pool(processes = \
                    self.num_workers) as pool:
                    for result in pool.imap_unordered(
                            calc_words_func, 
                            read_chunks_generator()):
                        for word,freq in result.items():
                            self.words_freq[word] += freq
                        del result
                self.logger("mp_pre_tokenizer: "
                f"process batch-{idx//v_proc_num} complete")
            end_time=time.time()
            self.logger("mp_pre_tokenizer: "
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
        
        x = 0
        while x < len(word) - 1:
            pair = (word[x], word[x+1])
            if pair != max_freq_pair:
                x += 1
                continue
            if x - 1 >= 0:
                l_pair = (word[x-1], word[x])
                self.pair_freq[l_pair] -= freq
            if x + 2 < len(word):
                r_pair = (word[x+1], word[x+2])
                self.pair_freq[r_pair] -= freq
            x += 2

        for x in range(len(word) - 1):
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
            new_word = calc_new_word(
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
        self.logger('merge: bpe merge process start')
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
                self.logger("merge: "
                    f"merge_step:[{len(self.vocab)}:"
                    f"{merge_bytes}]")
        self.r_vocab = \
                {y:x for (x,y) in self.vocab.items()}
        end_time = time.time()
        self.logger("merge: "
                f"merge process complete ["
                f"ct:{(end_time-start_time):.3f}s]")
        self.logger("merge: "
                f"final_vocab_size:{len(self.vocab)}")

    def run(self):
        self.init_vocab()
        self.mp_pre_tokenizer()
        self.merge()


def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        vocab_file:str,
        merges_file:str):
    bpe_trainer = BPETrainer(
        input_path,
        special_tokens,
        vocab_size)
    bpe_trainer.logger(f"train bpe begin "
        f"[input:{input_path}]")
    start_time = time.time()
    bpe_trainer.run()
    end_time = time.time()
    bpe_trainer.logger(f"train bpe complete ["
        f"ct:{(end_time-start_time):.3f}s]")
    
    vocab = bpe_trainer.vocab
    merges = bpe_trainer.merges
    with open(vocab_file, \
            'w', encoding='utf-8') as f:
        for token, idx in vocab.items():
            f.write(f"{idx} {token}\n")
    with open(merges_file, \
            'w', encoding='utf-8') as f:
        for (byte1, byte2) in merges:
            f.write(f"({byte1}, {byte2})\n")
    bpe_trainer.logger(f"train output file:"
        f"[{vocab_file},{merges_file}]")

    print(vocab_file, merges_file)

if __name__ == "__main__":  

    train_bpe(
        input_path=conf['owt']['data_file'],
        vocab_size=conf['owt']['vocab_size'],
        special_tokens=conf['special_tokens'],
        vocab_file=conf['owt']['vocab'],
        merges_file=conf['owt']['merges']
    )
