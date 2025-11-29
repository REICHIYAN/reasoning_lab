from scripts.run_benchmark import load_samples, build_models
from evaluation.runner.benchmark_runner import run_benchmark

def test_benchmark_runs_end_to_end():
    samples = load_samples("data/tasks/sample_tasks.jsonl")
    models = build_models([
        {"name": "CoT", "type": "cot"},
        {"name": "CoT+SC", "type": "cot_sc"},
        {"name": "RAG×CoT", "type": "rag_cot"},
        {"name": "ReAct", "type": "react"},
        {"name": "FSM", "type": "fsm"},
    ])
    metrics_list = run_benchmark(models, samples)
    # Expect one metrics object per model
    assert len(metrics_list) == 5
    for m in metrics_list:
        assert 0.0 <= m.exact_match <= 1.0
        assert 0.0 <= m.step_accuracy <= 1.0
        assert 0.0 <= m.self_consistency <= 1.0
        assert 0.0 <= m.paraphrase <= 1.0
        assert m.tokens_x > 0.0
        assert m.steps_x > 0.0
