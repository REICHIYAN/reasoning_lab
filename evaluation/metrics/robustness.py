from __future__ import annotations
from typing import List
from reasoning_core.models.base import PredictionLog

def mean_self_consistency(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    return sum(l.self_consistency_score for l in logs) / len(logs)

def mean_paraphrase_accuracy(logs: List[PredictionLog]) -> float:
    if not logs:
        return 0.0
    return sum(l.paraphrase_correct for l in logs) / len(logs)
