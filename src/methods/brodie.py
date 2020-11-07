import math
import numpy as np
from typing import List, Tuple, Dict, Iterable

from methods.hyperopt import HyperoptModel
from methods.tfidf import TfIdfBaseModel
from preprocess.seq_coder import SeqCoder


def needleman_wunsch_dist(frames1: List[str], frames2: List[str], tfidf1: Dict[str, float], d: float = -1) -> float:
    matrix = [[0.0 for _ in range(len(frames1) + 1)] for _ in range(len(frames2) + 1)]

    for i in range(len(frames2) + 1):
        matrix[i][0] = d * i
    for i in range(len(frames1) + 1):
        matrix[0][i] = d * i

    for i in range(1, len(frames2) + 1):
        for j in range(1, len(frames1) + 1):
            if frames2[i - 1] == frames1[j - 1]:
                p1 = tfidf1.get(frames1[j - 1], 1)
                p2 = 1 - j / len(frames1)
                p3 = math.exp(-abs(i - j) / 2)
                sim = p1 * p2 * p3
            else:
                sim = 0
            match = matrix[i - 1][j - 1] + sim
            delete = matrix[i - 1][j] + d
            insert = matrix[i][j - 1] + d
            matrix[i][j] = max(match, insert, delete)

    return matrix[-1][-1]


def needleman_wunsch(first, second, tfidf1: Dict[str, float], d: float = -1):
    tab = np.full((len(second) + 2, len(first) + 2), ' ', dtype=object)
    tab[0, 2:] = first
    tab[1, 1:] = list(range(0, -len(first) - 1, -1))
    tab[2:, 0] = second
    tab[1:, 1] = list(range(0, -len(second) - 1, -1))
    for f in range(2, len(first) + 2):
        for s in range(2, len(second) + 2):

            p1 = tfidf1.get(first[f - 2], 1)
            p2 = 1 - f / len(first)
            p3 = math.exp(-abs(f - s) / 2)
            is_equal = {True: p1 * p2 * p3, False: 0}

            tab[s, f] = max(tab[s - 1][f - 1] + is_equal[first[f - 2] == second[s - 2]],
                            tab[s - 1][f] + d,
                            tab[s][f - 1] + d)
    return tab[-1, -1]


class BrodieModel(TfIdfBaseModel, HyperoptModel):
    def __init__(self, coder: SeqCoder, d: float):
        super().__init__(coder)
        self.d = d

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'BrodieModel':
        super().fit(sim_train_data, unsup_data)

        params_edges = {
            "d": (-1, 0),
        }
        self.find_params(sim_train_data, params_edges)

        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []
        anchor = self.tfidf_module.coder.to_seq(anchor_id)
        anchor_tfidf = self.tfidf_module.tfidf_vectorizer.transform(anchor_id)
        anchor_tfidf = {k: 1 - math.exp(1 - idf) for k, (_, idf) in anchor_tfidf.items()}
        for stack_id in stack_ids:
            stack = self.tfidf_module.coder.to_seq(stack_id)
            dist1 = needleman_wunsch_dist(anchor, stack, anchor_tfidf, self.d)
            # dist2 = needleman_wunsch(anchor, stack, anchor_tfidf, self.d)
            scores.append(dist1)
        return scores

    def name(self) -> str:
        return self.tfidf_module.name() + f"_brodie_{self.d}"
