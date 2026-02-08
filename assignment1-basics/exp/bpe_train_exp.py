import os
import time
from utils.config import conf
from cs336_basics.bpe_train import BPETrainer

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
        input_path=conf['ts']['data_file'],
        vocab_size=conf['ts']['vocab_size'],
        special_tokens=conf['special_tokens'],
        vocab_file=conf['ts']['vocab'],
        merges_file=conf['ts']['merges']
    )
    
    train_bpe(
        input_path=conf['owt']['data_file'],
        vocab_size=conf['owt']['vocab_size'],
        special_tokens=conf['special_tokens'],
        vocab_file=conf['owt']['vocab'],
        merges_file=conf['owt']['merges']
    )
