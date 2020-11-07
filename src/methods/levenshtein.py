from typing import List, Tuple, Iterable

from data.readers import sim_data_stack_ids
from methods.base import SimStackModel
from preprocess.seq_coder import SeqCoder


def levenshtein_dist(frames1: List[int], weights1: List[float], frames2: List[int], weights2: List[float]) -> float:
    matrix = [[0.0 for _ in range(len(frames1) + 1)] for _ in range(len(frames2) + 1)]

    prev_column = matrix[0]

    for i in range(len(frames1)):
        prev_column[i + 1] = prev_column[i] + weights1[i]

    if len(frames1) == 0 or len(frames2) == 0:
        return 0.0

    curr_column = matrix[1]

    for i2 in range(len(frames2)):

        frame2 = frames2[i2]
        weight2 = weights2[i2]

        curr_column[0] = prev_column[0] + weight2

        for i1 in range(len(frames1)):

            frame1 = frames1[i1]
            weight1 = weights1[i1]

            if frame1 == frame2:
                curr_column[i1 + 1] = prev_column[i1]
            else:
                change = weight1 + weight2 + prev_column[i1]
                remove = weight2 + prev_column[i1 + 1]
                insert = weight1 + curr_column[i1]

                curr_column[i1 + 1] = min(change, remove, insert)

        if i2 != len(frames2) - 1:
            prev_column = curr_column
            curr_column = matrix[i2 + 2]

    return curr_column[-1]


class LevenshteinModel(SimStackModel):
    def __init__(self, coder: SeqCoder):
        self.coder = coder

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'LevenshteinModel':
        train_stacks = sim_data_stack_ids(sim_train_data)
        if unsup_data is not None:
            train_stacks = unsup_data

        self.coder.fit(train_stacks)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []
        anchor = self.coder(anchor_id)
        for stack_id in stack_ids:
            stack = self.coder(stack_id)
            d = levenshtein_dist(anchor, [1] * len(anchor), stack, [1] * len(stack))
            max_len = max(len(anchor), len(stack))
            scores.append(1 - d / max_len)
        return scores

    def name(self) -> str:
        return self.coder.name() + "_levenshtein"
