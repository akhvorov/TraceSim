import itertools
import math
import numpy as np
from typing import List, Tuple, Dict, Iterable

from data.readers import sim_data_stack_ids
from evaluation.stack_sim import auc_model
from methods.hyperopt import HyperoptModel
from preprocess.seq_coder import SeqCoder


class RebucketModel(HyperoptModel):
    def __init__(self, coder: SeqCoder, c: float = None, o: float = None):
        self.coder = coder
        self.c = c
        self.o = o

    def grid_search(self, sim_train_data: List[Tuple[int, int, int]],
                    params_edges: Dict[str, Tuple[float, float]], step: float = 0.1):
        best_params = {}
        best_score = 0

        params = {name: np.arange(edges[0], edges[1], step) for name, edges in params_edges.items()}.items()
        names = [x[0] for x in params]
        print("Params", names)
        for i, param in enumerate(list(itertools.product(*[x[1] for x in params]))):
            print(f"{i}-th iter, {param}", end=' ')
            param_dict = {name: value for name, value in zip(names, param)}
            self._set_params(param_dict)
            score = auc_model(self, sim_train_data, full=False)[0]
            if score > best_score:
                best_params = param
                best_score = score
                print("New best", best_score, best_params)

        self._set_params({name: value for name, value in zip(names, best_params)})

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'RebucketModel':
        train_stacks = sim_data_stack_ids(sim_train_data)
        if unsup_data is not None:
            train_stacks = unsup_data

        self.coder.fit(train_stacks)

        params_edges = {
            "c": (0, 2),
            "o": (0, 2)
        }
        # self.find_params(sim_train_data, params_edges)
        self.grid_search(sim_train_data, params_edges, step=0.1)  # rebucket use grid search

        return self

    def dist(self, stack1: List[int], stack2: List[int]):
        stack_len1 = len(stack1)
        stack_len2 = len(stack2)

        M = [[0. for i in range(stack_len2 + 1)] for j in range(stack_len1 + 1)]

        for i in range(1, stack_len1 + 1):
            for j in range(1, stack_len2 + 1):
                if stack1[i - 1] == stack2[j - 1]:
                    x = math.exp(-self.c * min(i - 1, j - 1)) * math.exp(-self.o * abs(i - j))
                else:
                    x = 0.
                M[i][j] = max(M[i - 1][j - 1] + x, M[i - 1][j], M[i][j - 1])
        sig = 0.
        for i in range(min(stack_len1, stack_len2) + 1):
            sig += math.exp(-self.c * i)

        sim = M[stack_len1][stack_len2] / sig
        return sim

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []
        anchor = self.coder(anchor_id)
        for stack_id in stack_ids:
            stack = self.coder(stack_id)
            scores.append(self.dist(anchor, stack))
        return scores

    def name(self) -> str:
        return self.coder.name() + f"_rebucket_{self.c}_{self.o}"
