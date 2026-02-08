import json
import shutil
import numpy as np
from pathlib import Path
from typing import TypeAlias
from collections import defaultdict
from typing import Iterator
from typing import Iterable

ByteTuple:TypeAlias = tuple[bytes, ...]

class WordData:
    def __init__(self):
        self.words_idx:\
            dict[ByteTuple:set[int]] \
            = defaultdict(set)
        self.words_freq:\
            dict[ByteTuple,int] \
            = defaultdict(int)
        self.pair_freq:\
            dict[ByteTuple,int] \
            = defaultdict(int)
        self.word_pair_d:\
            dict[ByteTuple:set[ByteTuple]]\
            = defaultdict(set)
        self.pair_word_d:\
            dict[ByteTuple:set[ByteTuple]]\
            = defaultdict(set)

class StreamingNpyWriter:
    def __init__(
            self,
            output_dir:str,
            chunk_size:int
            ):
        self.chunk_size = chunk_size
        self.cur_tokens = []
        self.chunks = []

        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def add(self, tokens:list):
        self.cur_tokens += tokens
        if len(self.cur_tokens) \
                >= self.chunk_size:
            self.save_chunk()
    
    def save_chunk(self):
        chunk_id = len(self.chunks)
        filename = \
            self.output_dir / \
            f'chunk_{chunk_id:04d}.npy'
        arr = np.array(
                self.cur_tokens, 
                dtype=np.uint16)
        np.save(filename, arr)
        
        cur_total = \
            sum(chunk['tokens'] \
            for chunk in self.chunks)

        self.chunks.append({
            'file': filename.name,
            'tokens': len(arr),
            'start': cur_total
        })
        
        self.cur_tokens.clear()
    
    def finish(self):
        self.save_chunk()
        with open(self.output_dir \
            / 'index.json', 'w') as f:
            json.dump({
                'chunks':self.chunks,
                'total_tokens':sum(c['tokens']\
                    for c in self.chunks)
            },f, indent=2)

class MultiNpyReader:
    def __init__(
            self, 
            data_dir, 
            file_pattern="*.npy"):
        self.data_dir = Path(data_dir)
        self.files = sorted(\
            self.data_dir.glob(
                file_pattern))
        if not self.files:
            raise FileNotFoundError(\
            f"no {file_pattern} files")
    
    def read(self, chunk_size=10000):
        for file_path in self.files:
            data = np.load(
                    file_path, 
                    mmap_mode='r')
            for start_idx in \
                range(0,len(data),chunk_size):
                end_idx = min(\
                    start_idx + chunk_size, 
                    len(data))
                yield data[start_idx:end_idx]

