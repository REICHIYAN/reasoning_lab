from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ReasoningMetrics:
    model_name: str
    # main metrics
    exact_match: float
    step_accuracy: float
    self_consistency: float
    paraphrase: float
    tokens_x: float
    steps_x: float
