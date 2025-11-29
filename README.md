# Reasoning-Lab

This repository provides a **self-contained Colab-friendly benchmark** for comparing
different reasoning strategies:

1. CoT (Chain-of-Thought)
2. CoT + Self-consistency
3. RAG × CoT (mocked)
4. ReAct (Tool-Augmented, mocked)
5. FSM Controller (LangGraph-like controller, mocked)

It does **not** call any external LLM API – everything is implemented using small,
deterministic toy logic so that it runs for free on Colab or any laptop.

## How to run (Colab or local)

```bash
python scripts/run_benchmark.py
```

You should see a table similar to:

```text
Model   | Outcome(EM) | Process(StepAcc) | Robustness(SC/Para) | Efficiency(Tokens/Steps)
--------+-------------+------------------+----------------------+-------------------------
CoT    ...
...
```

The exact numbers will differ from the paper-style example, but the **evaluation
pipeline and comparison structure** match what you would use in a NeurIPS/ICLR paper.

## Structure

- `reasoning_core/` – core reasoning models (CoT, RAG×CoT, ReAct, FSM)
- `evaluation/` – metric computation and table/report generation
- `data/` – small JSONL sample datasets
- `scripts/` – CLI entrypoints (`run_benchmark.py`)

This is intended as a **template**: you can replace the mocked models with real LLM
wrappers and keep all evaluation code intact.

## Configuration

The default experiment is defined in `configs/experiment_default.yaml`:

```yaml
dataset:
  path: data/tasks/sample_tasks.jsonl
models:
  - name: CoT
    type: cot
  - name: CoT+SC
    type: cot_sc
  - name: RAG×CoT
    type: rag_cot
  - name: ReAct
    type: react
  - name: FSM
    type: fsm
random_seed: 42
```

You can create new configs pointing to different datasets or subsets of models
without touching the Python code.

## Tests

A minimal `pytest` smoke test is included under `tests/`:

```bash
pytest -q
```

This verifies that all models run end-to-end and that metrics stay in valid ranges.
