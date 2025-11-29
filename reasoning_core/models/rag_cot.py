from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import random

from .base import ReasoningModel, Sample, PredictionLog
from .utils import toy_reason, noisy_boolean

@dataclass
class RagCotModel(ReasoningModel):
    name: str = "RAG×CoT"
    base_correct_prob: float = 0.83
    base_step_acc: float = 0.57
    seed: int = 1

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def predict(self, sample: Sample) -> PredictionLog:
        pred, cot, tokens, steps = toy_reason(sample.question)
        tokens = int(tokens * 1.8)
        steps = int(steps * 1.1)
        is_correct = noisy_boolean(self.base_correct_prob, self._rng)
        step_accuracy = self.base_step_acc
        self_consistency_score = 0.66
        paraphrase_correct = 0.66
        return PredictionLog(
            question=sample.question,
            gold_answer=sample.answer,
            prediction=pred,
            cot=cot,
            is_correct=is_correct,
            step_accuracy=step_accuracy,
            self_consistency_score=self_consistency_score,
            paraphrase_correct=paraphrase_correct,
            tokens=tokens,
            steps=steps,
            model_name=self.name,
        )
