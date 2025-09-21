
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import json
import re
from collections import Counter

def load_jsonl(path) -> List[dict]:
    with open(path, "r") as fi:
        data = [json.loads(line) for line in fi]
    return data

def write_jsonl(data: List[dict], out_path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for line in data:
            fo.write(json.dumps(line) + "\n")

def read_txt_file_as_list(path:str)-> list:
    data = []
    with open(path,"r") as fi:
        for line in fi:
            data.append(line.strip())
    return data

def count_capital_letters_sum(input_string):
    return sum(1 for char in input_string if char.isupper())

def collect_corpus(dir: str, pos_lex: List[str], neg_lex: List[str], 
                   data: Optional[List[dict]] = None) -> List[dict]:
    """Collect all .txt files from a directory and label with sentiment + features."""
    if data is None:
        data = []

    idx = len(data)
    sentiment = "pos" if Path(dir).name == "pos" else "neg"
    p = Path(dir)


    for i, entry in tqdm(enumerate(p.iterdir()), total=sum(1 for _ in p.iterdir())):
        if entry.is_file() and entry.suffix == ".txt":
            raw_txt = entry.read_text(encoding="utf-8", errors="ignore")
            txt = raw_txt.lower()

            # features
            n_capitals = sum(1 for char in raw_txt if char.isupper())
            n_exclamations = raw_txt.count("!")

            # tokenize on words (basic split, could improve with regex)
            words = re.findall(r"\b\w+\b", txt)
            word_counts = Counter(words)

            n_pos_words = sum(word_counts[w] for w in pos_lex if w in word_counts)
            n_neg_words = sum(word_counts[w] for w in neg_lex if w in word_counts)

            data.append({
                "idx": idx + i,
                "sentiment": sentiment,
                "n_capitals": n_capitals,
                "n_exclamations": n_exclamations,
                "n_pos_words": n_pos_words,
                "n_neg_words": n_neg_words,
                "txt": txt,
            })
    return data


