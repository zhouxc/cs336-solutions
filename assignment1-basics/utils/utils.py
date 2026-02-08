import os
import re
import regex
import ast
from typing import BinaryIO
from utils.data import ByteTuple
from collections import defaultdict

def get_compression_ratio(
        string: str,
        indices: list[int])->float:
    num_bytes = \
        len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def split_by_special_tokens(
        text:str,
        tokens:list[str]
        )->list[str]:
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

def calc_gpt2_words_v1(
        text:str,
        special_tokens:list[str]
        )->dict[ByteTuple,int]:
    pattern = (
            r"'(?:[sdmt]|ll|ve|re)" +
            r"| ?\p{L}+" +
            r"| ?\p{N}+" +
            r"| ?[^\s\p{L}\p{N}]+" +
            r"|\s+(?!\S)" +
            r"|\s+"
    )
    words_freq:dict[ByteTuple:int] \
            = defaultdict(int)
    splits = split_by_special_tokens(
            text,
            special_tokens)
    for split in splits:
        if special_tokens is not None and\
                split in special_tokens:
            continue
        gpt2_words = regex.findall(
                pattern,
                split,
                regex.UNICODE)
        for word in gpt2_words:
            word_byte = tuple(bytes([x])\
                for x in word.encode('utf-8'))
            words_freq[word_byte] += 1
    return words_freq

def calc_gpt2_words_v2(
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
    splits = split_by_special_tokens(
            text,
            special_tokens)
    for split in splits:
        if special_tokens is not None and\
                split in special_tokens:
            words += [(split.encode('utf-8'),)]
            continue
        gpt2_words = regex.findall(
                pattern,
                split,
                regex.UNICODE)
        for word in gpt2_words:
            word_byte = tuple(bytes([x])\
                for x in word.encode('utf-8'))
            words += [word_byte]
    return words

def calc_word_pair(
        word:ByteTuple
        )->set[ByteTuple]:
    pairs:set[ByteTuple] = set()
    for x in range(len(word)-1):
        pair = (word[x], word[x+1])
        pairs.add(pair)
    return pairs

def calc_new_word(
        word:ByteTuple,
        merge:ByteTuple
        )->ByteTuple:
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

def find_chunk_boundaries(
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
    
def chunk_generator(
        batch_boundaries,
        f):
    def generator():
        for idx, (start, end) in \
            enumerate(batch_boundaries):
            f.seek(start)
            _size = end - start
            chunk_text = \
                f.read(_size).decode('utf-8')
            yield chunk_text
    return generator


def parse_merge_pair(
        line:str
        )->ByteTuple|None:
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
