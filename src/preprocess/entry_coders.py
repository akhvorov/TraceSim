from abc import ABC, abstractmethod
from typing import List

from data.objects import Stack


class Entry2Seq(ABC):
    @abstractmethod
    def __call__(self, stack: Stack):
        pass

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def remove_equals(words: List[str]) -> List[str]:
    res = []
    for i, w in enumerate(words):
        if (i == 0 or words[i - 1] != w) and w.strip() != '':
            res.append(w)
    return res


class Entry2SeqHelper:
    def __init__(self, cased: bool = True, trim_len: int  = 0):
        self.cased = cased
        self.trim_len = trim_len
        self._name = ("" if cased else "un") + "cs" + (f"_tr{trim_len}" if trim_len > 0 else "")

    def __call__(self, seq: List[str]) -> List[str]:
        if self.trim_len > 0:
            seq = remove_equals([".".join(s.split('.')[:-self.trim_len]) for s in seq])
        seq = [s if self.cased else s.lower() for s in seq]
        return seq

    def name(self) -> str:
        return self._name


class Stack2Seq(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int  = 0):
        self.helper = Entry2SeqHelper(cased, trim_len)

    def __call__(self, stack: Stack) -> List[str]:
        return self.helper(stack.frames[::-1])

    def name(self) -> str:
        return "st_" + self.helper.name()


class Exception2Seq(Entry2Seq):
    def __init__(self, cased=True, trim_len=0, throw=False, to_set=True):
        self.helper = Entry2SeqHelper(cased, trim_len)
        self.throw = throw
        self.ex_transform = lambda x: (sorted(list(set(x)), reverse=True) if to_set else x)
        self._name = "ex_" + self.helper.name() + ("_" if throw else "_un") + "thr_" + ("st" if to_set else "lst")

    def __call__(self, stack: Stack) -> List[str]:
        exceptions = list(map(lambda x: x + (".throw" if self.throw else ""), self.ex_transform(stack.clazz)))
        return self.helper(exceptions)

    def name(self) -> str:
        return self._name


class MultiEntry2Seq(Entry2Seq):
    def __init__(self, e2ss: List[Entry2Seq]):
        self.e2ss = e2ss

    def __call__(self, stack: Stack) -> List[str]:
        res = []
        for e2s in self.e2ss:
            res += e2s(stack)
        return res

    def name(self) -> str:
        return "multe2s_" + "_".join(e2s.name() for e2s in self.e2ss)
