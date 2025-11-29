from __future__ import annotations
from typing import Tuple
import random

def toy_reason(question: str) -> Tuple[str, str, int, int]:
    """Very small mock 'reasoning': we just do simple arithmetic pattern parsing.
    Returns (prediction, CoT text, tokens, steps).
    """
    q = question.strip().lower()
    tokens = len(q.split()) + 10
    steps = 2
    if " + " in q and "what is" in q:
        try:
            expr = q.split("what is")[-1].strip().rstrip("?")
            parts = expr.split("+")
            a = int(parts[0])
            b = int(parts[1])
            res = a + b
            cot = f"First, compute {a} + {b}. The result is {res}."
            return str(res), cot, tokens + 5, steps + 1
        except Exception:
            pass
    # fallback: echo last word as fake answer
    last = q.split()[-1].rstrip("?")
    cot = f"I read the question and pick '{last}' as the answer."
    return last, cot, tokens, steps

def noisy_boolean(base_prob: float, rng: random.Random) -> bool:
    return rng.random() < base_prob
