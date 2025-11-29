from __future__ import annotations
from typing import List
from reasoning_core.models.base import PredictionLog

def mean_step_accuracy(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    return sum(l.step_accuracy for l in logs) / len(logs)
