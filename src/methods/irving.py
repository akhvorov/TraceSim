import math
from typing import List, Tuple, Iterable

from methods.hyperopt import HyperoptModel
from methods.tfidf import TfIdfBaseModel
from preprocess.seq_coder import SeqCoder


class IrvingModel(TfIdfBaseModel, HyperoptModel):
    def __init__(self, coder: SeqCoder = None,
                 pos_coef: float = None, df_coef: float = None, diff_coef: float = None,
                 match_cost: float = None, gap_penalty: float = None, mismatch_penalty: float = None):
        super().__init__(coder)
        self.pos_coef = pos_coef
        self.df_coef = df_coef
        self.diff_coef = diff_coef

        self.match_cost = match_cost
        self.gap_penalty = gap_penalty
        self.mismatch_penalty = mismatch_penalty

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'IrvingModel':
        super().fit(sim_train_data, unsup_data)

        params_edges = {
            "pos_coef": (0, 1),
            "df_coef": (0, 15),
            "diff_coef": (0, 4),

            "match_cost": (0, 4),
            "gap_penalty": (0, 4),
            "mismatch_penalty": (0, 4)
        }
        self.find_params(sim_train_data, params_edges, max_evals=50)

        return self

    def weights(self, stack_id: int, coded_seq: List[str], w_p: float, w_df: float) -> List[float]:
        idfs = self.tfidf_module.tfidf_vectorizer.transform(stack_id)
        weights = []
        for i, word in enumerate(coded_seq):
            local_score = math.exp(-w_p * i)

            tf, idf = idfs[word]
            # idf = 1 + np.log(self.N / v)
            df = 1 / math.exp(idf - 1)
            global_score = math.exp(-w_df * df)
            weights.append(local_score * global_score)

        return weights

    def predict(self, anchor_id: int, stack_ids: List[int],
                pos_coef: float = None, df_coef: float = None, diff_coef: float = None) -> List[float]:
        pos_coef = self.pos_coef if pos_coef is None else pos_coef
        df_coef = self.df_coef if df_coef is None else df_coef
        diff_coef = self.diff_coef if diff_coef is None else diff_coef

        scores = []
        anchor_seq = self.tfidf_module.coder.to_seq(anchor_id)
        anchor_weights = self.weights(anchor_id, anchor_seq, pos_coef, pos_coef)
        for stack_id in stack_ids:
            stack_seq = self.tfidf_module.coder.to_seq(stack_id)
            stack_weights = self.weights(stack_id, stack_seq, pos_coef, df_coef)
            max_dist = sum(anchor_weights) + sum(stack_weights)
            dist = self.dist(anchor_seq, anchor_weights, stack_seq, stack_weights, diff_coef,
                             self.gap_penalty, self.match_cost, self.mismatch_penalty)
            # score = 0 if max_dist == 0 else 1 - dist / max_dist
            score = dist
            scores.append(score)
        return scores

    @staticmethod
    def dist(frames1: List[int], weights1: List[float],
             frames2: List[int], weights2: List[float],
             diff_coef: float, gap_penalty: float, match_cost: float, mismatch_penalty: float) -> float:
        matrix = [[0.0 for _ in range(len(frames1) + 1)] for _ in range(len(frames2) + 1)]

        prev_column = matrix[0]

        for i in range(len(frames1)):
            prev_column[i + 1] = prev_column[i] - gap_penalty * weights1[i]

        if len(frames1) == 0 or len(frames2) == 0:
            return 0.0

        curr_column = matrix[1]

        for i2 in range(len(frames2)):

            frame2 = frames2[i2]
            weight2 = weights2[i2]

            curr_column[0] = prev_column[0] - weight2

            for i1 in range(len(frames1)):

                frame1 = frames1[i1]
                weight1 = weights1[i1]

                if frame1 == frame2:
                    change_weight = match_cost * max(weight1, weight2) * math.exp(-diff_coef * abs(i1 - i2))
                else:
                    change_weight = mismatch_penalty * -max(weight1, weight2)

                change = prev_column[i1] + change_weight
                remove = prev_column[i1 + 1] - weight2
                insert = curr_column[i1] - weight1

                curr_column[i1 + 1] = max(change, remove, insert)

            if i2 != len(frames2) - 1:
                prev_column = curr_column
                curr_column = matrix[i2 + 2]

        return curr_column[-1]

    def name(self) -> str:
        return self.tfidf_module.name() + f"_irving_{self.pos_coef}_{self.df_coef}_{self.diff_coef}"
