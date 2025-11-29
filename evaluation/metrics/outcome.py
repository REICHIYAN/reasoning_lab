from __future__ import annotations
from typing import List
from reasoning_core.models.base import PredictionLog

def exact_match(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    correct = sum(1 for l in logs if l.is_correct)
    return correct / len(logs)
