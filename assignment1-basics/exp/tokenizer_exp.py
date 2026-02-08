import time
from utils.config import conf
from utils.utils import *
from cs336_basics.tokenizer import BPETokenizer

def exp1():
    with open(conf['ts']['sample_file'],\
            "r",encoding='utf-8') as f:
        text = f.read()
        tokenizer = BPETokenizer.from_files(
            conf['ts']['vocab'], 
            conf['ts']['merges'],
            special_tokens = \
                    conf['special_tokens']
        )
        encoded_ids = tokenizer.encode(text)
        ts_compress_ratio = \
                get_compression_ratio(
                text, encoded_ids)
    
    with open(conf['owt']['sample_file'],\
            "r",encoding='utf-8') as f:
        text = f.read()
        tokenizer = BPETokenizer.from_files(
            conf['owt']['vocab'],
            conf['owt']['merges'],
            special_tokens = \
                    conf['special_tokens']
        )
        encoded_ids = tokenizer.encode(text)
        owt_compress_ratio = \
                get_compression_ratio(
                text, encoded_ids)
    return (ts_compress_ratio, \
                owt_compress_ratio)

def exp2():
    with open(conf['owt']['sample_file'],\
            "r",encoding='utf-8') as f:
        text = f.read()
        tokenizer = BPETokenizer.from_files(
            conf['ts']['vocab'],
            conf['ts']['merges'],
            special_tokens = \
                    conf['special_tokens']
        )
        encoded_ids = tokenizer.encode(text)
        compress_ratio =\
                get_compression_ratio(
                text, encoded_ids)
    return compress_ratio

def exp3():
    with open(conf['owt']['sample_file'],\
            "r",encoding='utf-8') as f:
        text = f.read()
        tokenizer = BPETokenizer.from_files(
            conf['owt']['vocab'],
            conf['owt']['merges'],
            special_tokens = \
                    conf['special_tokens']
        )
        start_time = time.time()
        encoded_ids = tokenizer.encode(text)
        end_time = time.time()
    text_bytes = len(bytes(text,\
            encoding="utf-8"))
    second = (end_time - start_time)
    throughput = text_bytes/second
    pile_cost_h = 825*1024**3/throughput/3600
    
    return (throughput,pile_cost_h)

def exp4():
    t1 = time.time()
    tokenizer = BPETokenizer.from_files(
            conf['ts']['vocab'],
            conf['ts']['merges'],
            special_tokens = \
                conf['special_tokens']
    )
    tokenizer.encode_mp(
        input_file=conf['ts']['data_file'], 
        output_dir=conf['ts']['encode_dir'],
        batch_num=conf['batch_num'],
        num_workers=conf['num_workers'],
        chunk_size=conf['chunk_size'],
        split_token=conf['split_token']
    )
    t2 = time.time()
    tokenizer = BPETokenizer.from_files(
            conf['owt']['vocab'],
            conf['owt']['merges'],
            special_tokens = \
                conf['special_tokens']
    )
    tokenizer.encode_mp(
        input_file=conf['owt']['data_file'], 
        output_dir=conf['owt']['encode_dir'],
        batch_num=conf['batch_num'],
        num_workers=conf['num_workers'],
        chunk_size=conf['chunk_size'],
        split_token=conf['split_token']
    )
    t3 = time.time()
    return ((t2-t1)/3600,(t3-t2)/3600, 
            conf['ts']['encode_dir'], 
            conf['owt']['encode_dir'])

if __name__ == '__main__':
    
    exp1_result = exp1()
    print("tokenizer-exp1-result:",exp1_result)
    exp2_result = exp2()
    print("tokenizer-exp2-result:",exp2_result)
    exp3_result = exp3()
    print("tokenizer-exp3-result:",exp3_result)
    exp4_result = exp4()
    print("tokenizer-exp4-result:",exp4_result)
