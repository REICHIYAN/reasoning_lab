from __future__ import annotations
from typing import Iterable, List
from ..models.base import Sample, PredictionLog, ReasoningModel

def run_model_on_samples(model: ReasoningModel, samples: Iterable[Sample]) -> List[PredictionLog]:
    logs: List[PredictionLog] = []
    for s in samples:
        logs.append(model.predict(s))
    return logs
