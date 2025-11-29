from __future__ import annotations
from typing import List
from .schema import ReasoningMetrics

def to_ascii_table(metrics: List[ReasoningMetrics]) -> str:
    headers = [
        "Model",
        "Outcome(EM)",
        "Process(StepAcc)",
        "Robustness(SC/Para)",
        "Efficiency(Tokens/Steps)",
    ]
    rows = []
    for m in metrics:
        row = [
            m.model_name,
            f"{m.exact_match*100:.1f}",
            f"{m.step_accuracy:.2f}",
            f"{m.self_consistency:.2f}/{m.paraphrase:.2f}",
            f"{m.tokens_x:.2f}×/{m.steps_x:.2f}×",
        ]
        rows.append(row)

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(cells):
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = []
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-"*w for w in col_widths))
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)
