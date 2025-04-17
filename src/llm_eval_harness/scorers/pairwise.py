"""Pairwise comparison — head-to-head scoring with significance testing."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness.core.errors import ScorerError
from llm_eval_harness.core.models import Verdict
from llm_eval_harness.scorers.judge import Judge, JudgeVerdict


@dataclass
class PairwiseResult:
    a_wins: int
    b_wins: int
    ties: int
    win_rate: float
    ci_low: float
    ci_high: float
    bradley_terry_a: float
    bradley_terry_b: float

    def summary(self) -> str:
        return (
            f"A wins: {self.a_wins} | B wins: {self.b_wins} | ties: {self.ties}\n"
            f"Win rate (A): {self.win_rate:.3f}  95% CI [{self.ci_low:.3f}, {self.ci_high:.3f}]\n"
            f"B-T score: A={self.bradley_terry_a:.3f}  B={self.bradley_terry_b:.3f}"
        )


async def compare_pairwise(
    judge: Judge,
    pairs: Sequence[tuple[str, str, str, str | None]],
    *,
    concurrency: int = 8,
) -> PairwiseResult:
    """`pairs`: iterable of (prompt, a, b, reference). Returns aggregate result."""
    if not pairs:
        raise ScorerError("compare_pairwise requires at least one pair")

    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def _decide(prompt: str, a: str, b: str, ref: str | None) -> JudgeVerdict:
        async with semaphore:
            verdict_prompt = (
                "Compare A and B. Reply JSON: {\"winner\": \"A\"|\"B\"|\"tie\", \"score_a\": 0..1, \"score_b\": 0..1}"
                f"\nPROMPT: {prompt}\nA: {a}\nB: {b}\nREFERENCE: {ref or '-'}"
            )
            return await judge.judge(verdict_prompt, a, reference=ref)

    verdicts = await asyncio.gather(*(_decide(p, a, b, r) for p, a, b, r in pairs))
    counter_obj = Counter()
    score_a = score_b = 0.0
    for verdict in verdicts:
        label = (verdict.label or "").upper()
        if label not in {"A", "B"}:
            label = "A" if verdict.score >= 0.5 else "B"
        if label == "A":
            counter_obj["A"] += 1
            score_a += 1.0
        elif label == "B":
            counter_obj["B"] += 1
            score_b += 1.0
        else:
            counter_obj["tie"] += 1

    a_wins = counter_obj["A"]
    b_wins = counter_obj["B"]
    ties = counter_obj["tie"]
    n = a_wins + b_wins + ties
    win_rate = (a_wins + 0.5 * ties) / n
    ci_low, ci_high = _wilson_ci(a_wins, n)
    bt_a, bt_b = _bradley_terry(a_wins, b_wins, ties)
    return PairwiseResult(
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        bradley_terry_a=bt_a,
        bradley_terry_b=bt_b,
    )


def _wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _bradley_terry(a_wins: int, b_wins: int, ties: int) -> tuple[float, float]:
    total = a_wins + b_wins + ties
    if total == 0:
        return 0.5, 0.5
    p_a = (a_wins + 0.5 * ties) / total
    p_b = 1.0 - p_a
    return _normalise(p_a, p_b)


def _normalise(a: float, b: float) -> tuple[float, float]:
    s = a + b
    if s <= 0:
        return 0.5, 0.5
    return a / s, b / s