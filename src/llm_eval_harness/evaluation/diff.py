"""Diff two runs to detect regressions and improvements."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from llm_eval_harness.core.errors import EvalHarnessError
from llm_eval_harness.core.models import RunReport


@dataclass
class RegressionDiff:
    base_id: str
    head_id: str
    scorer_deltas: dict[str, float]
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    latency_delta_ms: float = 0.0
    cost_delta_usd: float = 0.0

    def is_regression(self, *, threshold: float = 0.02) -> bool:
        return any(delta <= -threshold for delta in self.scorer_deltas.values())

    def to_dict(self) -> dict:
        return {
            "base_id": self.base_id,
            "head_id": self.head_id,
            "scorer_deltas": self.scorer_deltas,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "latency_delta_ms": self.latency_delta_ms,
            "cost_delta_usd": self.cost_delta_usd,
        }


def diff(base: RunReport | str | Path, head: RunReport | str | Path) -> RegressionDiff:
    base_report = _coerce(base)
    head_report = _coerce(head)

    base_scores = _aggregate(base_report)
    head_scores = _aggregate(head_report)
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    for name in sorted(set(base_scores) | set(head_scores)):
        b = base_scores.get(name, 0.0)
        h = head_scores.get(name, 0.0)
        delta = h - b
        deltas[name] = round(delta, 4)
        if delta <= -0.02:
            regressions.append(f"{name}: {b:.3f} -> {h:.3f} (Δ={delta:+.3f})")
        elif delta >= 0.02:
            improvements.append(f"{name}: {b:.3f} -> {h:.3f} (Δ={delta:+.3f})")

    base_lat = _median_latency(base_report)
    head_lat = _median_latency(head_report)
    base_cost = sum(r.cost_usd for r in base_report.results)
    head_cost = sum(r.cost_usd for r in head_report.results)

    return RegressionDiff(
        base_id=base_report.run_id,
        head_id=head_report.run_id,
        scorer_deltas=deltas,
        regressions=regressions,
        improvements=improvements,
        latency_delta_ms=round(head_lat - base_lat, 2),
        cost_delta_usd=round(head_cost - base_cost, 6),
    )


def _coerce(report: RunReport | str | Path) -> RunReport:
    if isinstance(report, RunReport):
        return report
    p = Path(report)
    if not p.exists():
        raise EvalHarnessError(f"Run report not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return RunReport.model_validate(raw)


def _aggregate(report: RunReport) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in report.results:
        for s in r.scorer_outputs:
            totals[s.name] = totals.get(s.name, 0.0) + s.score
            counts[s.name] = counts.get(s.name, 0) + 1
    return {name: totals[name] / max(counts[name], 1) for name in totals}


def _median_latency(report: RunReport) -> float:
    lat = sorted(r.latency_ms for r in report.results)
    if not lat:
        return 0.0
    return lat[len(lat) // 2]


def merge(diffs: Iterable[RegressionDiff]) -> RegressionDiff:
    items = list(diffs)
    if not items:
        raise EvalHarnessError("merge requires at least one RegressionDiff")
    head = items[-1]
    return RegressionDiff(
        base_id=items[0].base_id,
        head_id=head.head_id,
        scorer_deltas=head.scorer_deltas,
        regressions=head.regressions,
        improvements=head.improvements,
        latency_delta_ms=head.latency_delta_ms,
        cost_delta_usd=head.cost_delta_usd,
    )