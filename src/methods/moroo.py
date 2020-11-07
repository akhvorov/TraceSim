from typing import List, Tuple, Dict, Iterable

from methods.hyperopt import HyperoptModel
from methods.lerch import LerchModel
from methods.rebucket import RebucketModel
from preprocess.seq_coder import SeqCoder


class MorooModel(HyperoptModel):
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.rebucket = RebucketModel(coder, 0, 0)
        self.lerch = LerchModel(coder)
        self.alpha = 1.

    def _set_params(self, args: Dict[str, float]):
        self.rebucket.c = args['c']
        self.rebucket.o = args['o']
        self.alpha = args['alpha']

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'MorooModel':
        # self.rebucket.fit(sim_train_data, unsup_data)
        self.lerch.fit(sim_train_data, unsup_data)

        params_edges = {
            "c": (0, 3),
            "o": (0, 3),
            'alpha': (0, 1)
        }
        self.find_params(sim_train_data, params_edges, max_evals=20)

        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        lens = [len(set(self.coder(stack_id))) for stack_id in stack_ids]
        rebucket_score = self.rebucket.predict(anchor_id, stack_ids)
        lerch_score = self.lerch.predict(anchor_id, stack_ids)
        lerch_score = [score / (l ** 0.5) for score, l in zip(lerch_score, lens)]
        return [r * l / (self.alpha * r + (1 - self.alpha) * l) for r, l in zip(rebucket_score, lerch_score)]

    def name(self) -> str:
        return self.coder.name() + f"_moroo_{self.rebucket.c}_{self.rebucket.o}_{self.alpha}"

    def save(self, name: str = ""):
        self.lerch.save(name)

    def load(self, name: str = "") -> 'MorooModel':
        lerch = self.lerch.load(name)
        if lerch is not None:
            self.lerch = lerch
        return self
