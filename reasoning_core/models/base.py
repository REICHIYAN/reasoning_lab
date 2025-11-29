from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List

@dataclass
class Sample:
    """Input example used for reasoning evaluation.

    Attributes
    ----------
    question:
        Natural-language question to be answered. In real experiments,
        this can be math word problems, QA, code prompts, etc.
    answer:
        Gold/reference answer used for outcome metrics (e.g., EM/Accuracy).
    meta:
        Arbitrary metadata such as ID, task type, difficulty, or source.
    """
    question: str
    answer: str
    meta: Dict[str, Any]

@dataclass
class PredictionLog:
    """Per-sample log containing both outcome and process information.

    This is the central structure consumed by the evaluation pipeline.
    A real system would populate these fields from actual LLM calls and
    process supervision signals.

    Attributes
    ----------
    question:
        Original input question.
    gold_answer:
        Reference answer.
    prediction:
        Model's final answer.
    cot:
        Chain-of-thought or reasoning trace (free-form text).
    is_correct:
        Whether prediction is counted as correct for EM/Accuracy.
    step_accuracy:
        Process-level score in [0,1] representing how many reasoning
        steps are judged correct on average.
    self_consistency_score:
        Aggregate score in [0,1] representing stability across multiple
        reasoning samples (e.g., majority-vote success).
    paraphrase_correct:
        Accuracy on paraphrased versions of the same question.
    tokens:
        Approximate token cost used for efficiency analysis.
    steps:
        Abstract “reasoning steps” used by the model (e.g., CoT length,
        number of tool calls, FSM transitions).
    model_name:
        Identifier of the model or reasoning strategy used.
    """
    question: str
    gold_answer: str
    prediction: str
    cot: str
    is_correct: bool
    step_accuracy: float
    self_consistency_score: float
    paraphrase_correct: float
    tokens: int
    steps: int
    model_name: str

class ReasoningModel(Protocol):
    """Interface for all reasoning strategies.

    Implementations wrap a specific reasoning style such as:

    * Plain Chain-of-Thought (CoT)
    * CoT + Self-consistency
    * RAG × CoT
    * ReAct (tool-augmented reasoning)
    * FSM-controlled multi-step reasoning

    The key requirement is that :meth:`predict` returns a :class:`PredictionLog`
    with both outcome and process information. This allows the same evaluation
    pipeline to be reused across models and datasets, which is exactly the
    structure you would want in a NeurIPS/ICLR reasoning benchmark.
    """

    name: str

    def predict(self, sample: Sample) -> PredictionLog:
        """Run reasoning for a single sample and return a detailed log.

        In a real system, this method would:
        1. Construct prompts / tool calls for an LLM.
        2. Execute one or more LLM calls.
        3. Parse the reasoning trace (CoT, tool outputs, FSM states).
        4. Compute process-level scores as needed.

        Here it is implemented with a toy backend, but the interface is
        designed to be stable even when swapping in real LLM clients.
        """
        ...
