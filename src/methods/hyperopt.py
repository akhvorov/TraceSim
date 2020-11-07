from abc import ABC
from typing import Dict, List, Tuple

from hyperopt import hp, fmin, tpe, space_eval

from evaluation.stack_sim import auc_model
from methods.base import SimStackModel


class HyperoptModel(SimStackModel, ABC):
    def _set_params(self, args: Dict[str, float]):
        for k, v in args.items():
            self.__dict__[k] = v

    def find_params(self, sim_train_data: List[Tuple[int, int, int]],
                    params_edges: Dict[str, Tuple[float, float]],
                    max_evals: int = 50):
        def objective(args):
            self._set_params(args)
            return 1 - auc_model(self, sim_train_data, full=False)[0]

        space = {name: hp.uniform(name, edges[0], edges[1]) for name, edges in params_edges.items()}
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
        print("Top params", space_eval(space, best))
        self._set_params(space_eval(space, best))
