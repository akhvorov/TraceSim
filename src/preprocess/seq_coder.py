from functools import lru_cache
from typing import List, Iterable

from data.readers import StackLoader
from preprocess.entry_coders import remove_equals, Entry2Seq
from preprocess.tokenizers import Padding, Tokenizer


class VocabFreqController:
    def __init__(self, min_freq: int = 0, oov: str = 'OOV'):
        self.min_freq = min_freq
        self._freqs = {}
        self.frequent_words = set()
        self.oov = oov

    def fit(self, texts: Iterable[List[str]]) -> 'VocabFreqController':
        for text in texts:
            for word in text:
                self._freqs[word] = self._freqs.get(word, 0) + 1
        for word in self._freqs:
            if self._freqs[word] >= self.min_freq:
                self.frequent_words.add(word)
        return self

    def encode(self, text: List[str]) -> List[str]:
        if self.min_freq <= 0:
            return text
        return [w if w in self.frequent_words else self.oov for w in text]

    def __call__(self, text: List[str]) -> List[str]:
        return self.encode(text)

    def transform(self, texts: Iterable[List[str]]) -> List[List[str]]:
        return [self.encode(text) for text in texts]

    def fit_transform(self, texts: Iterable[List[str]]) -> List[List[str]]:
        return self.fit(texts).transform(texts)

    def __len__(self) -> int:
        return len(self.frequent_words) + 1

    def name(self) -> str:
        return "oov" + str(self.min_freq)


class CharFilter:
    def __init__(self):
        self.ok_symbols = set([chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.', ','])  # $

    def __call__(self, seq: List[str]) -> List[str]:
        return [s for s in ("".join(filter(lambda x: x.lower() in self.ok_symbols, word)) for word in seq) if s]


class SeqCoder:
    def __init__(self, stack_loader: StackLoader, entry_to_seq: Entry2Seq, tokenizer: Tokenizer,
                 min_freq: int = 0, max_len: int = None):
        self.stack_loader = stack_loader
        self.entry_to_seq = entry_to_seq
        self.char_filter = CharFilter()
        self.vocab_control = VocabFreqController(min_freq)
        self.tokenizer = Padding(tokenizer, max_len)
        self.fitted = False
        self._name = "_".join(
            filter(lambda x: x.strip(),
                   (self.stack_loader.name(), entry_to_seq.name(), self.tokenizer.name(), self.vocab_control.name())))

    def fit(self, stack_ids: Iterable[int]) -> 'SeqCoder':
        if self.fitted:
            print("SeqCoder already fitted, fit call skipped")
            return self
        seqs = [self.char_filter(self.entry_to_seq(self.stack_loader(stack_id))) for stack_id in stack_ids]
        if self.vocab_control:
            seqs = self.vocab_control.fit_transform(seqs)
        self.tokenizer.fit(seqs)
        self.fitted = True
        return self

    def _pre_call(self, stack_id: int):
        res = self.stack_loader(stack_id)
        for tr in [self.entry_to_seq, self.char_filter, self.vocab_control]:
            if tr is not None:
                res = tr(res)
        return remove_equals(res)

    @lru_cache(maxsize=200_000)
    def __call__(self, stack_id: int) -> List[int]:
        return self.tokenizer(self._pre_call(stack_id))

    @lru_cache(maxsize=200_000)
    def to_seq(self, stack_id: int) -> List[str]:
        return self.tokenizer.split(self._pre_call(stack_id))

    def __len__(self) -> int:
        return len(self.tokenizer)

    def name(self) -> str:
        return self._name

    def train(self, mode: bool = True):
        self.tokenizer.train(mode)
