'''
python ./src/data_managment/clean_polarity2.py
'''
from collections import Counter
import sys
sys.path.append("./src")
from utils.file_io import collect_corpus, write_jsonl, read_txt_file_as_list

import json
from pathlib import Path
from typing import List, Optional
import random
import numpy as np

def count_vectorizer(corpus:List[dict], lam:Optional[str]=0) -> List[dict]:
    vocab_set = set()
    docs_tokens = []
    for doc in corpus:
        tokens  = [term.strip().lower() for term in doc['txt'].split()]
        docs_tokens.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)
    idx = {t: i for i, t in enumerate(vocab)}

    # make a matrix the size of vocab, n-rows
    X = np.ones((len(corpus), len(vocab)), dtype=int) * lam
    # for each index, for each list of tokens
    for i, tokens in enumerate(docs_tokens):
        c = Counter(tokens)
        for term, count in c.items():
            X[i, idx[term]] = count

    for i, doc in enumerate(corpus):
        doc["counts"] = X[i].tolist()

    return corpus

def main():
    random.seed(42)

    print("Loading lexicons...")
    pos_lex = read_txt_file_as_list("./data/lexicons/pos.txt")
    neg_lex = read_txt_file_as_list("./data/lexicons/neg.txt")
    print(f"   ➡ Loaded {len(pos_lex)} positive words and {len(neg_lex)} negative words.")

    print("Collecting negative corpus...")
    neg = collect_corpus("./data/raw/txt_sentoken/neg", pos_lex, neg_lex)
    print(f"   ➡ Collected {len(neg)} negative examples.")

    print("Collecting positive corpus...")
    data = collect_corpus("./data/raw/txt_sentoken/pos", pos_lex, neg_lex, neg)
    print(f"   ➡ Total collected dataset size: {len(data)} examples.")


    data = count_vectorizer(data)

    print("Shuffling data...")
    random.shuffle(data)

    print("Splitting into train/dev/test...")
    n = len(data)
    train_size = int(n * 0.7)
    dev_size = int(n * 0.15)

    train = data[:train_size]
    dev = data[train_size:train_size + dev_size]
    test = data[train_size + dev_size:]
    print(f"   ➡ Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    print("Saving splits to ./data/polarity2/ ...")
    splits = {"train": train, "dev": dev, "test": test}
    for split, dat in splits.items():
        out_path = f"./data/polarity2/{split}.jsonl"
        write_jsonl(dat, out_path)
        print(f"   ✅ Wrote {len(dat)} {split} examples to {out_path}")

    print("🎉 All done!")

if __name__ == "__main__":
    main()
