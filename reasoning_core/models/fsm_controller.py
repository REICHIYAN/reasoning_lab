from __future__ import annotations
from dataclasses import dataclass
import random

from .base import ReasoningModel, Sample, PredictionLog
from .utils import toy_reason, noisy_boolean

@dataclass
class FSMControllerModel(ReasoningModel):
    name: str = "FSM"
    base_correct_prob: float = 0.90
    base_step_acc: float = 0.64
    seed: int = 3

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def predict(self, sample: Sample) -> PredictionLog:
        pred, cot, tokens, steps = toy_reason(sample.question)
        tokens = int(tokens * 2.1)
        steps = int(steps * 1.3)
        cot = f"[STATE=RETRIEVE]->[STATE=REASON]->[STATE=ANSWER]\n{cot}"
        is_correct = noisy_boolean(0.897, self._rng)
        step_accuracy = self.base_step_acc
        self_consistency_score = 0.78
        paraphrase_correct = 0.78
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
