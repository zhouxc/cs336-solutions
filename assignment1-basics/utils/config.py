from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR/"output"
DATA_DIR = BASE_DIR/"data"

TS_CONFIG = {
    "name": "TinyStories",
    "vocab": OUTPUT_DIR/"ts.vocab",
    "merges": OUTPUT_DIR/"ts.merges",
    "data_file": DATA_DIR/"TinyStoriesV2-GPT4-train.txt",
    "sample_file": DATA_DIR/"ts-sample-100.txt",
    "encode_dir": OUTPUT_DIR/"ts-idx",
    "vocab_size": 10000,
}

OWT_CONFIG = {
    "name": "OpenWebText",
    "vocab": OUTPUT_DIR/"owt.vocab",
    "merges": OUTPUT_DIR/"owt.merges",
    "data_file": DATA_DIR/"owt_train.txt",
    "sample_file": DATA_DIR/"owt-sample-100.txt",
    "encode_dir": OUTPUT_DIR/"owt-idx",
    "vocab_size": 32000,
}

conf = {
    "batch_num":12,
    "chunk_size":100000000,
    "num_workers":16,
    'special_tokens':["<|endoftext|>"],
    'split_token':b"<|endoftext|>",
    "ts": TS_CONFIG,
    "owt": OWT_CONFIG,
}
