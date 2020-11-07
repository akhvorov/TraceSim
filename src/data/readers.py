import json
import os
from typing import List, Tuple

import pandas as pd
from functools import lru_cache

from data.objects import Stack


def read_stack(path: str, frames_field: str = 'frames') -> Stack:
    with open(path) as f:
        dict = json.loads(f.read())
        return Stack(dict['id'], dict['timestamp'], dict['class'], dict.get(frames_field, dict["frames"])[0])


def read_supervised(path: str, have_train_indicator: bool = False, verbose: bool = False) -> \
        Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    df = pd.read_csv(path)
    target_tr, target_te = [], []
    for row in df.itertuples():
        t = (row.rid1, row.rid2, int(row.label))
        if have_train_indicator and row.train:
            target_tr.append(t)
        else:
            target_te.append(t)
    if verbose:
        print(f"Train pairs count: {len(target_tr)}")
        print(f"Test pairs count: {len(target_te)}")

    return target_tr, target_te


def sim_data_stack_ids(sim_data: List[Tuple[int, int, int]]) -> List[int]:
    return [p[0] for p in sim_data] + [p[1] for p in sim_data]


class StackLoader:
    def __init__(self, *dirs: str, frames_field: str = 'frames'):
        self.dirs = list(dirs)
        self.id_dir = {}
        self.frames_field = frames_field

    def init(self, *dirs: str):
        self.dirs += list(dirs)

    def add(self, directory: str):
        self.dirs.append(directory)

    def name(self) -> str:
        return ("rec" if self.frames_field == "frames" else "notrec") + "_loader"

    @lru_cache(maxsize=300_000)
    def __call__(self, id: int) -> Stack:
        if id not in self.id_dir:
            for d in self.dirs:
                if os.path.exists(f"{d}/{id}.json"):
                    self.id_dir[id] = d
                    break
        return read_stack(f"{self.id_dir[id]}/{id}.json", self.frames_field)
