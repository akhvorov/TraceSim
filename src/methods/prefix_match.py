from typing import List, Tuple, Iterable

from data.readers import sim_data_stack_ids
from methods.base import SimStackModel
from preprocess.seq_coder import SeqCoder


class PrefixMatchModel(SimStackModel):
    def __init__(self, coder: SeqCoder, top_n: int = None):
        self.coder = coder
        self.top_n = top_n

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'PrefixMatchModel':
        train_stacks = sim_data_stack_ids(sim_train_data)
        if unsup_data is not None:
            train_stacks += unsup_data

        self.coder.fit(train_stacks)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []
        anchor = self.coder(anchor_id)
        for stack_id in stack_ids:
            prefix_len = None
            stack = self.coder(stack_id)
            for i, (fr1, fr2) in enumerate(zip(anchor, stack)):
                if fr1 != fr2:
                    prefix_len = i
            if prefix_len is None:
                prefix_len = min(len(anchor), len(stack))
            scores.append(prefix_len / max(len(anchor), len(stack)))
        return scores

    def name(self) -> str:
        return self.coder.name() + "_prefix_match" + (f"_{self.top_n}" if self.top_n is not None else "")
