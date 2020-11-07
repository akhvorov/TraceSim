from abc import ABC, abstractmethod
from typing import List, Union, Any


class IdCoder:
    def __init__(self):
        self.id2name = {}
        self.name2id = {}
        self.fixed = False

    def encode(self, word: Any) -> Union[int, None]:
        if not self.fixed and word not in self.name2id:
            self.name2id[word] = len(self.name2id)
            self.id2name[self.name2id[word]] = word
        return self.name2id.get(word, None)

    def __getitem__(self, item: Any) -> Union[int, None]:
        return self.encode(item)

    def encodes(self, words: List[Any]) -> List[Union[int, None]]:
        return [self.encode(word) for word in words]

    def decode(self, id: int) -> Any:
        return self.id2name[id]

    def decodes(self, ids: List[int]) -> List[Any]:
        return [self.decode(id) for id in ids]

    def fix(self):
        self.fixed = True

    def __len__(self) -> int:
        return len(self.name2id)


class Tokenizer(ABC):
    def __init__(self):
        self._train = False

    @abstractmethod
    def fit(self, texts: List[List[str]]) -> 'Tokenizer':
        return self

    @abstractmethod
    def encode(self, text: List[str]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def split(self, text: List[str]) -> List[str]:
        raise NotImplementedError

    def __call__(self, text: List[str], type: str = 'id') -> List[Union[int, str]]:
        if type == 'id':
            return self.encode(text)
        else:
            return self.split(text)

    @abstractmethod
    def to_str(self, id: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def train(self, mode: bool = True):
        self._train = mode


class Padding(Tokenizer):
    def __init__(self, tokenizer: 'Tokenizer', max_len: int = None,
                 sos: str = '[SOS]', eos: str = '[EOS]', pad: str = '[PAD]'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sos = sos
        self.eos = eos
        self.pad = pad

        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2

        self.offset = 3

    def fit(self, texts: List[List[str]]) -> 'Padding':
        self.tokenizer.fit(texts)
        return self

    def pad_seq(self, seq: List[Any], pad: Any):
        if self.max_len is not None:
            if len(seq) < self.max_len - 1:
                return [pad] * (self.max_len - 1 - len(seq)) + seq
            else:
                return seq[len(seq) - min(len(seq), self.max_len):]
        return seq

    def encode(self, text: List[str]) -> List[int]:
        encoded_seq = self.tokenizer.encode(text)
        encoded_seq = [x + self.offset for x in encoded_seq]
        return [self.sos_id] + self.pad_seq(encoded_seq, self.pad_id) + [self.eos_id]

    def split(self, text: List[str]) -> List[str]:
        return [self.sos] + self.pad_seq(self.tokenizer.split(text), self.pad) + [self.eos]

    def to_str(self, id: int) -> str:
        if id >= self.offset:
            return self.tokenizer.to_str(id - self.offset)
        elif id == self.sos_id:
            return self.sos
        elif id == self.eos_id:
            return self.eos
        elif id == self.pad_id:
            return self.pad
        else:
            raise ValueError("Unknown token id")

    def __len__(self) -> int:
        return len(self.tokenizer) + self.offset

    def name(self) -> str:
        return self.tokenizer.name() + (f"_pad{self.max_len}" if self.max_len is not None else "")

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.train(mode)


class SimpleTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.coder = IdCoder()

    def fit(self, texts: List[List[str]]):
        for text in texts:
            self.coder.encodes(text)
        self.coder.fix()
        return self

    def encode(self, text: List[str]) -> List[int]:
        return list(filter(lambda x: x is not None, self.coder.encodes(text)))

    def split(self, text: List[str]) -> List[str]:
        return text

    def to_str(self, id: int) -> str:
        return self.coder.decode(id)

    def __len__(self):
        return len(self.coder)

    def name(self) -> str:
        return "simple"
