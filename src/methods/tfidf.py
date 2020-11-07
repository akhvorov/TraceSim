import pickle
from abc import ABC
from functools import lru_cache
from typing import List, Dict, Tuple, Union, Iterable

import numpy as np
from collections import Counter

from data.readers import sim_data_stack_ids
from methods.base import SimStackModel
from preprocess.seq_coder import SeqCoder


class TfIdfComputer:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.word2idx = {}
        self.doc_freq = []
        self.N = None

    def fit(self, stack_ids: List[int]) -> 'TfIdfComputer':
        if self.N is not None:
            print("TfIdf model already fitted (skipped)")
            return self
        self.coder.fit(stack_ids)
        texts = [" ".join(self.coder.to_seq(id)) for id in stack_ids]
        self.N = len(texts)
        for text in texts:
            for word in set(text.split(' ')):
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.doc_freq.append(0)
                self.doc_freq[self.word2idx[word]] += 1
        for i, v in enumerate(self.doc_freq):
            self.doc_freq[i] = 1 + np.log(self.N / v)
        return self

    def idf(self, frame: str, default: float = 0.):
        if frame not in self.word2idx:
            return default
        return self.doc_freq[self.word2idx[frame]]

    @lru_cache(maxsize=20_000)
    def transform(self, stack_id: int) -> Dict[str, Tuple[float, float]]:
        vec = {}
        words = self.coder.to_seq(stack_id)
        words_freqs = Counter(words)
        for word, freq in words_freqs.items():
            if word not in self.word2idx:
                idf = np.log(self.N)
            else:
                idf = self.doc_freq[self.word2idx[word]]
            tf = np.sqrt(words_freqs[word])
            vec[word] = tf, idf
        return vec


class TfIdfModule:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.tfidf_vectorizer = TfIdfComputer(self.coder)

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: List[int] = None):
        train_stacks = sim_data_stack_ids(sim_train_data)
        if unsup_data is not None:
            train_stacks = unsup_data

        # self.coder.fit(train_stacks)
        self.tfidf_vectorizer.fit(train_stacks)
        return self

    def name(self) -> str:
        return self.coder.name() + "_tfidf"

    def save(self, name: str = ""):
        with open("models/" + self.name() + "_" + name + ".model", 'wb') as f:
            pickle.dump(self, f)

    def load(self, name: str = "") -> Union[None, 'TfIdfModule']:
        path = "models/" + self.name() + "_" + name + ".model"
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(e)
            return None


class TfIdfBaseModel(SimStackModel, ABC):
    def __init__(self, coder: SeqCoder):
        self.tfidf_module = TfIdfModule(coder)

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'TfIdfBaseModel':
        self.tfidf_module.fit(sim_train_data, unsup_data)
        return self

    def save(self, name: str = ""):
        self.tfidf_module.save(name)

    def load(self, name: str = "") -> Union[None, 'TfIdfBaseModel']:
        tfidf_module = self.tfidf_module.load(name)
        if tfidf_module is not None:
            print("Load model")
            self.tfidf_module = tfidf_module
        return self
