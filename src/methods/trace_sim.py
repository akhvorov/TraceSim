import math
from typing import List, Tuple, Iterable

from methods.hyperopt import HyperoptModel
from methods.levenshtein import levenshtein_dist
from methods.tfidf import TfIdfBaseModel
from preprocess.seq_coder import SeqCoder


class TraceSimModel(TfIdfBaseModel, HyperoptModel):
    def __init__(self, coder: SeqCoder = None, alpha: float = None, beta: float = None, gamma: float = None):
        super().__init__(coder)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'TraceSimModel':
        super().fit(sim_train_data, unsup_data)

        params_edges = {
            "alpha": (0, 0.5),
            "beta": (0, 7),
            "gamma": (0, 15)
        }
        self.find_params(sim_train_data, params_edges)

        return self

    def weights(self, stack_id: int, coded_seq: List[str], alpha: float, beta: float, gamma: float) -> List[float]:
        local_weight = [1 / (1 + i) ** alpha for i, _ in enumerate(coded_seq)]

        idfs = self.tfidf_module.tfidf_vectorizer.transform(stack_id)
        global_weight = []
        for word in coded_seq:
            tf, idf = idfs.get(word, (0, 0))
            score = idf
            global_weight.append(1 / (1 + math.exp(-beta * (score - gamma))))

        return [lw * gw for lw, gw in zip(local_weight, global_weight)]

    def predict(self, anchor_id: int, stack_ids: List[int],
                alpha: float = None, beta: float = None, gamma: float = None) -> List[float]:
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma

        scores = []
        anchor_seq = self.tfidf_module.coder.to_seq(anchor_id)
        anchor_weights = self.weights(anchor_id, anchor_seq, alpha, beta, gamma)
        for stack_id in stack_ids:
            stack_seq = self.tfidf_module.coder.to_seq(stack_id)
            stack_weights = self.weights(stack_id, stack_seq, alpha, beta, gamma)
            max_dist = sum(anchor_weights) + sum(stack_weights)
            dist = levenshtein_dist(anchor_seq, anchor_weights, stack_seq, stack_weights)
            score = 0 if max_dist == 0 else 1 - dist / max_dist
            scores.append(score)
        return scores

    def name(self) -> str:
        return self.tfidf_module.name() + f"_tracesim_{self.alpha}_{self.beta}_{self.gamma}"
