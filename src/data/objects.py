from collections import namedtuple
from typing import List


class Stack:
    def __init__(self, id: int, timestamp: int, clazz: str, frames: List[str]):
        self.id = id
        self.ts = timestamp
        self.clazz = clazz
        self.frames = frames

    def eq_content(self, stack: 'Stack'):
        return self.clazz == stack.clazz and self.frames == stack.frames
