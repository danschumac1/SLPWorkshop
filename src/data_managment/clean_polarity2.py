import sys
sys.path.append("./src")
from utils.file_io import collect_corpus, write_jsonl, read_txt_file_as_list

import json
from pathlib import Path
from typing import List, Optional
import random

def main():
    random.seed(42)
    pos_lex = read_txt_file_as_list("./data/lexicon/pos.txt")
    neg_lex = read_txt_file_as_list("./data/lexicon/neg.txt")

    # load data
    neg = collect_corpus("./data/raw/txt_sentoken/neg", pos_lex, neg_lex)
    data = collect_corpus("./data/raw/txt_sentoken/pos", pos_lex, neg_lex, neg)

    # shuffle and split
    random.shuffle(data)
    n = len(data)
    train_size = int(n * 0.7)
    dev_size = int(n * 0.15)

    train = data[:train_size]
    dev = data[train_size:train_size + dev_size]
    test = data[train_size + dev_size:]

    # save
    splits = {"train": train, "dev": dev, "test": test}
    for split, dat in splits.items():
        print(len(dat))
        write_jsonl(dat, f"./data/polarity2/{split}.jsonl")

if __name__ == "__main__":
    main()
