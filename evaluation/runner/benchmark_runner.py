from __future__ import annotations
from typing import List, Dict
from dataclasses import dataclass

from reasoning_core.models.base import Sample, PredictionLog, ReasoningModel
from reasoning_core.pipelines.runner import run_model_on_samples
from evaluation.metrics.outcome import exact_match
from evaluation.metrics.process import mean_step_accuracy
from evaluation.metrics.robustness import mean_self_consistency, mean_paraphrase_accuracy
from evaluation.metrics.efficiency import mean_tokens, mean_steps
from evaluation.reporting.schema import ReasoningMetrics

def _normalize_efficiency(all_logs: Dict[str, List[PredictionLog]]) -> Dict[str, ReasoningMetrics]:
    # baseline: CoT tokens/steps
    base_logs = next(iter(all_logs.values()))
    base_tokens = mean_tokens(base_logs)
    base_steps = mean_steps(base_logs) or 1.0

    metrics: Dict[str, ReasoningMetrics] = {}
    for name, logs in all_logs.items():
        em = exact_match(logs)
        step_acc = mean_step_accuracy(logs)
        sc = mean_self_consistency(logs)
        para = mean_paraphrase_accuracy(logs)
        t = mean_tokens(logs) or base_tokens
        s = mean_steps(logs) or base_steps
        metrics[name] = ReasoningMetrics(
            model_name=name,
            exact_match=em,
            step_accuracy=step_acc,
            self_consistency=sc,
            paraphrase=para,
            tokens_x=t / base_tokens if base_tokens else 1.0,
            steps_x=s / base_steps if base_steps else 1.0,
        )
    return metrics

def run_benchmark(models: List[ReasoningModel], samples: List[Sample]) -> List[ReasoningMetrics]:
    all_logs: Dict[str, List[PredictionLog]] = {}
    for m in models:
        logs = run_model_on_samples(m, samples)
        all_logs[m.name] = logs
    metrics_dict = _normalize_efficiency(all_logs)
    return list(metrics_dict.values())
