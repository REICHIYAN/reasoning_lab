from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import random

from .base import ReasoningModel, Sample, PredictionLog
from .utils import toy_reason, noisy_boolean

@dataclass
class CoTModel(ReasoningModel):
    name: str = "CoT"
    base_correct_prob: float = 0.72
    base_step_acc: float = 0.48
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def predict(self, sample: Sample) -> PredictionLog:
        pred, cot, tokens, steps = toy_reason(sample.question)
        is_correct = noisy_boolean(self.base_correct_prob, self._rng)
        step_accuracy = self.base_step_acc
        self_consistency_score = 0.41
        paraphrase_correct = 0.41
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

@dataclass
class CoTSCModel(CoTModel):
    name: str = "CoT+SC"
    sc_samples: int = 5

    def predict(self, sample: Sample) -> PredictionLog:
        base_log = super().predict(sample)
        tokens = int(base_log.tokens * 2.4)  # self-consistency cost
        self_consistency_score = 0.63
        paraphrase_correct = 0.63
        step_accuracy = 0.51
        is_correct = noisy_boolean(0.77, self._rng)
        base_log.tokens = tokens
        base_log.self_consistency_score = self_consistency_score
        base_log.paraphrase_correct = paraphrase_correct
        base_log.step_accuracy = step_accuracy
        base_log.is_correct = is_correct
        base_log.model_name = self.name
        return base_log
