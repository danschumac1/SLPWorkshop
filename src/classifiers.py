'''
python ./src/classifiers.py
'''

import numpy as np
from utils.file_io import load_jsonl

class LogisticClassifier:
    def __init__(self, feature_array:np.array, ground_truths:np.array, threshold=.5, z=8):
        self.feature_array = self._normalize_features(feature_array)
        self.ground_truths = ground_truths
        self.threshold = threshold
        self.z = z
        
    def _normalize_features(self,feature_array):
        mus = np.mean(feature_array, axis=0)
        stds = np.std(feature_array, axis=0)
        normalized_feature_array = (feature_array - mus) / stds
        return normalized_feature_array

    def sigmoid(self, z:float) -> float:
        sigz = 1/(1 + np.exp(-z))
        return sigz
    
def main():
    train = load_jsonl("./data/polarity2/train.jsonl")
    # dev = load_jsonl("./data/polarity2/dev.jsonl")
    # test = load_jsonl("./data/polarity2/test.jsonl")

    ground_truths = np.array([row["sentiment"] == "pos" for row in train])
    feature_array = np.array([
        np.array([row["n_capitals"] for row in train]),
        np.array([row["n_exclamations"] for row in train]),
        np.array([row["n_pos_words"] for row in train]),
        np.array([row["n_neg_words"] for row in train]),
        ])

    lc = LogisticClassifier(feature_array, ground_truths)
    print(lc.feature_array)
    # sigz = lc.sigmoid(lc.z)
    # print(sigz)
    

if __name__ == '__main__':
    main()