from collections import Counter

import random
import numpy as np
from typing import List, Tuple

from data.readers import read_supervised, StackLoader
from evaluation.stack_sim import auc_model
from methods.base import SimStackModel
from methods.brodie import BrodieModel
from methods.cosine import CosineModel
from methods.irving import IrvingModel
from methods.lerch import LerchModel
from methods.levenshtein import LevenshteinModel
from methods.moroo import MorooModel
from methods.prefix_match import PrefixMatchModel
from methods.rebucket import RebucketModel
from methods.trace_sim import TraceSimModel
from preprocess.entry_coders import Exception2Seq, Stack2Seq, MultiEntry2Seq
from preprocess.seq_coder import SeqCoder
from preprocess.tokenizers import SimpleTokenizer


def set_seed(seed: int = 1):
    np.random.seed(seed)
    random.seed(seed)


def create_model(stack_loader: StackLoader, method: str = 'lerch',
                 use_ex: bool = False, max_len: int = None, trim_len: int = 0) -> SimStackModel:
    stack2seq = Stack2Seq(cased=False, trim_len=trim_len)
    if use_ex:
        stack2seq = MultiEntry2Seq([stack2seq, Exception2Seq(cased=False, trim_len=trim_len, throw=False, to_set=True)])
    coder = SeqCoder(stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len)
    if method == 'lerch':
        model = LerchModel(coder)
    elif method == 'cosine':
        model = CosineModel(coder)
    elif method == 'prefix':
        model = PrefixMatchModel(coder)
    elif method == 'rebucket':
        model = RebucketModel(coder)
    elif method == 'tracesim':
        model = TraceSimModel(coder)
    elif method == 'levenshtein':
        model = LevenshteinModel(coder)
    elif method == 'brodie':
        model = BrodieModel(coder, -1)
    elif method == 'moroo':
        model = MorooModel(coder)
    elif method == 'irving':
        model = IrvingModel(coder)
    else:
        raise ValueError("Method name is not match")
    return model


def split_train_test(sim_data1: List[Tuple[int, int, int]],
                     sim_data2: List[Tuple[int, int, int]],
                     rand_split: bool = False, train_size: float = 0.8):
    if rand_split:
        sim_data = np.random.permutation(sim_data1 + sim_data2)
    else:
        sim_data = sorted(sim_data1 + sim_data2, key=lambda t: max(t[0], t[1]))

    return sim_data[:int(train_size * len(sim_data))], sim_data[int(train_size * len(sim_data)):]


def stack_sim(stack_loader: StackLoader, target_path: str, method: str,
              rand_split: bool = False, split_ratio: float = 0.1):
    set_seed()

    sim_train_data, sim_test_data = read_supervised(target_path, have_train_indicator=False, verbose=False)
    sim_train_data, sim_test_data = split_train_test(sim_train_data, sim_test_data, rand_split, split_ratio)

    print("Train labels", dict(Counter([x[2] for x in sim_train_data])))
    print("Test labels", dict(Counter([x[2] for x in sim_test_data])))

    model = create_model(stack_loader, method)
    model.fit(sim_train_data, None)

    train_all_auc = auc_model(model, sim_train_data, full=False)
    test_all_auc = auc_model(model, sim_test_data, full=False)

    return test_all_auc[0]
