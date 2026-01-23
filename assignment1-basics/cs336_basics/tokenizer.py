import re
import time
import regex
from typing import TypeAlias

ByteTuple:TypeAlias = tuple[bytes, ...]

class BPETokenizer:
    def __init__(
            self, 
            train_data_path:str='',
            special_tokens:list[str]=[],
            vocab_size:int=256
            ):
        self.train_data_path = train_data_path
        self.train_samples:list[str] = []
        
        self.vocab_size = vocab_size
        self.vocab:dict[bytes,int] = {}
        self.r_vocab:dict[bytes,int] = {}
        self.special_tokens = special_tokens
        
        self.words_freq:dict[
                ByteTuple, 
                int] = {}
        self.pair_freq:dict[ByteTuple,int] = {}
        self.merges:list[ByteTuple] = []
        
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
    
    def read_data(self): 
        with open(self.train_data_path, "rb") as f:
            train_sample = f.read().decode('utf-8')

        self.train_samples = \
                self.split_by_special_tokens(
                    train_sample,
                    self.special_tokens
                )

    def pre_tokenizer(self):
        # gpt2 pre-tokenizer regex
        pattern = (
            r"'(?:[sdmt]|ll|ve|re)" +
            r"| ?\p{L}+" +
            r"| ?\p{N}+" +
            r"| ?[^\s\p{L}\p{N}]+" +
            r"|\s+(?!\S)" +
            r"|\s+"
        )
        gpt2_words = []
        for sample in self.train_samples:
            gpt2_words += regex.findall(
                    pattern, 
                    sample, 
                    regex.UNICODE)
        for word in gpt2_words:
            word = tuple(bytes([x])\
                    for x in word.encode('utf-8'))
            self.words_freq[word] = \
                self.words_freq.get(word, 0) + 1
    
    def find_max_freq_pair(
                self,
                pre_merge_bytes:bytes
        )->ByteTuple:
        for word, freq in \
                self.words_freq.items():
            if pre_merge_bytes !=b'' and \
                  pre_merge_bytes not in word:
                continue
            for x in range(len(word)-1):
                if pre_merge_bytes !=b'' \
                    and word[x] != pre_merge_bytes\
                    and word[x+1] != pre_merge_bytes:
                    continue
                byte_pair= (word[x], word[x+1])
                self.pair_freq[byte_pair] = \
                    self.pair_freq.get(byte_pair, 0)\
                    + freq
        
        max_freq = max(self.pair_freq.values())
        max_freq_pairs = [pair for pair, freq in 
                self.pair_freq.items() 
                if freq == max_freq
            ]
        max_freq_pairs.sort(reverse = True)
        max_freq_pair= max_freq_pairs[0]
        #print("----max freq pair----")
        #print(max_freq_pair, max_freq)
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

    def update_word_dict(
            self, 
            max_freq_pair:ByteTuple):
        new_words_freq:dict[ByteTuple] = {}
        for word, freq in self.words_freq.items():
            if (max_freq_pair[0] not in word) \
                 or (max_freq_pair[1] not in word):
                new_words_freq[word] = freq
                continue
            new_word = self.calc_new_word(
                    freq, word, max_freq_pair)
            new_words_freq[new_word] = freq
        self.words_freq = new_words_freq
        self.pair_freq.pop(max_freq_pair)
    
    def train(self):
        merge_bytes=b''
        while len(self.vocab) < self.vocab_size:
            max_freq_pair = \
                self.find_max_freq_pair(merge_bytes)
            if len(max_freq_pair) < 2:
                break
            self.merges.append(max_freq_pair)
            merge_bytes = \
                    max_freq_pair[0] +\
                    max_freq_pair[1]
            if merge_bytes not in self.vocab:
                self.vocab[merge_bytes] = len(self.vocab)
                self.update_word_dict(max_freq_pair)
        #        print("add new-token:", merge_bytes, \
        #            " vocab_size:",len(self.vocab))
        self.r_vocab = \
                {y:x for (x,y) in self.vocab.items()}

    def run(self):
        self.read_data()
        self.init_vocab()
        self.pre_tokenizer()
        self.train()

    def validate(self):
        """Test the BPE tokenizer"""
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
    bpe_tokenizer.run()
    return (bpe_tokenizer.r_vocab,
                bpe_tokenizer.merges)


def validate(
        vocab:dict[int, bytes],
        vocab_size:int,
        merges:list[tuple[bytes,bytes]]
        ):
    
    #print("--------merges--------------")
    #print(len(merges))
    #for merge_pair in merges:
    #    print(merge_pair)
    
    print("---------vocab--------------")
    print(vocab_size)
    print(len(vocab))
    for idx,token in vocab.items():
          print(idx,token)


if __name__ == "__main__":  

    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    #input_path = "./tests/fixtures/corpus.en"
    vocab_size = 10000
    special_tokens=["<|endoftext|>"]
    t1 = time.time()
    vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
    t2=time.time()
    print("-----time------")
    print(t2-t1)
    #validate(
    #        vocab=vocab,
    #        vocab_size=vocab_size,
    #        merges=merges
    #    )
