'''
python ./src/classifiers.py
'''

import numpy as np
from utils.file_io import load_jsonl

class LogisticClassifier:
    def __init__(
            self, 
            feature_array:np.array, 
            ground_truths:np.array, 
            threshold:float=.5,
            lr:float= 0.01
            ):
        self.feature_array = self._normalize_features(feature_array)
        self.wandb = np.random.rand(self.feature_array.shape[1]) 
        self.ground_truths = ground_truths
        self.threshold = threshold
        self.lr = lr

    def _normalize_features(self, feature_array:np.array) -> np.array:
        # add bias vector
        mus = np.mean(feature_array, axis=0)
        stds = np.std(feature_array, axis=0)
        normalized_feature_array = (feature_array - mus) / stds
        biases = np.ones(feature_array.shape[0], 1)
        normalized_feature_array = np.column_stack(normalized_feature_array, biases)
        return normalized_feature_array

    def sigmoid(self, z:float) -> float:
        sigz = 1/(1 + np.exp(-z))
        return sigz
    
    def calc_avg_ce_loss(self, gt:int, features_vec:np.array) -> float:
        """
        gts: shape (n_samples,)
        features_vec: shape (n_samples, n_features)
        self.wandb: shape (n_features,)
         """
        z = features_vec @ self.wandb
        yhat = self.sigmoid(z)
        cel = -1 * (gt*np.log(yhat) + (1 - gt) * np.log(1 - yhat))
        return cel
    
    def update(self, features, ground_truth):
        z = features @ self.wandb
        yhat = self.sigmoid(z)
        gradients = (yhat - ground_truth)  * features
        self.wandb = self.wandb - self.lr * gradients


    def train(self):
        for x, y in zip(self.feature_array, self.ground_truths):
            self.update(x, y)
                
                
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