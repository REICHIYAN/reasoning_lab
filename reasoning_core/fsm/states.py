from __future__ import annotations
from enum import Enum, auto

class State(Enum):
    ASK = auto()
    RETRIEVE = auto()
    REASON = auto()
    TOOL = auto()
    ANSWER = auto()
    ERROR = auto()
