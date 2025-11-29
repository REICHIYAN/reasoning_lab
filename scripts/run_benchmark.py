from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

from reasoning_core.models.base import Sample
from reasoning_core.models.cot import CoTModel, CoTSCModel
from reasoning_core.models.rag_cot import RagCotModel
from reasoning_core.models.react_agent import ReActModel
from reasoning_core.models.fsm_controller import FSMControllerModel
from evaluation.runner.benchmark_runner import run_benchmark
from evaluation.reporting.table_formatter import to_ascii_table

def load_samples(path: str) -> list[Sample]:
    p = Path(path)
    samples: list[Sample] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        samples.append(Sample(
            question=obj["question"],
            answer=obj["answer"],
            meta=obj.get("meta", {}),
        ))
    return samples

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # Fallback to a simple default if no config is present.
        return {
            "dataset": {"path": "data/tasks/sample_tasks.jsonl"},
            "models": [
                {"name": "CoT", "type": "cot"},
                {"name": "CoT+SC", "type": "cot_sc"},
                {"name": "RAG×CoT", "type": "rag_cot"},
                {"name": "ReAct", "type": "react"},
                {"name": "FSM", "type": "fsm"},
            ],
            "random_seed": 42,
        }
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def build_models(model_cfgs: List[Dict[str, Any]]):
    models = []
    for cfg in model_cfgs:
        t = cfg["type"]
        if t == "cot":
            models.append(CoTModel())
        elif t == "cot_sc":
            models.append(CoTSCModel())
        elif t == "rag_cot":
            models.append(RagCotModel())
        elif t == "react":
            models.append(ReActModel())
        elif t == "fsm":
            models.append(FSMControllerModel())
        else:
            raise ValueError(f"Unknown model type: {t}")
    return models

def main(config_path: str = "configs/experiment_default.yaml") -> None:
    cfg = load_config(config_path)
    dataset_path = cfg["dataset"]["path"]
    samples = load_samples(dataset_path)
    models = build_models(cfg["models"])
    metrics_list = run_benchmark(models, samples)
    # sort by model name for stable output
    table = to_ascii_table(sorted(metrics_list, key=lambda m: m.model_name))
    print(table)

if __name__ == "__main__":
    main()
