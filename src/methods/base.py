from abc import abstractmethod, ABC
from typing import Tuple, List, Iterable


class SimStackModel(ABC):
    @abstractmethod
    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'SimStackModel':
        raise NotImplementedError

    @abstractmethod
    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        raise NotImplementedError

    def predict_pairs(self, sim_data: List[Tuple[int, int, int]]) -> List[float]:
        return [self.predict(st_id1, [st_id2])[0] for st_id1, st_id2, l in sim_data]

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class SimPairStackModel(SimStackModel):
    @abstractmethod
    def predict_pair(self, stack1: int, stack2: int) -> float:
        raise NotImplementedError

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        return [self.predict_pair(anchor_id, stack_id) for stack_id in stack_ids]
