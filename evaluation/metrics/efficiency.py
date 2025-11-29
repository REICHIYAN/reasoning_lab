from __future__ import annotations
from typing import List
from reasoning_core.models.base import PredictionLog

def mean_tokens(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    return sum(l.tokens for l in logs) / len(logs)

def mean_steps(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    return sum(l.steps for l in logs) / len(logs)
