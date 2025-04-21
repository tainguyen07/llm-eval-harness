"""Gate conditions for CI — fail runs that drop below thresholds."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from llm_eval_harness.core.models import RunReport


@dataclass
class GateReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "measurements": self.measurements,
        }


def evaluate_gates(report: RunReport, gates: Mapping[str, float]) -> GateReport:
    """`gates` keys: `min_<scorer>` (e.g. `min_exact_match`), `max_p99_ms`, `max_cost_usd`."""
    scores = _aggregate_scores(report)
    measurements: dict[str, float] = {}
    failures: list[str] = []

    for key, threshold in gates.items():
        if key.startswith("min_"):
            name = key.removeprefix("min_")
            value = scores.get(name, 0.0)
            measurements[key] = round(value, 4)
            if value < threshold:
                failures.append(f"{name} {value:.3f} < {threshold}")
        elif key.startswith("max_"):
            name = key.removeprefix("max_")
            value = _aggregate_other(report, name)
            measurements[key] = round(value, 4)
            if value > threshold:
                failures.append(f"{name} {value:.3f} > {threshold}")
        else:
            raise ValueError(f"Unknown gate key: {key!r}")

    return GateReport(passed=not failures, failures=failures, measurements=measurements)


def iter_failures(gate_reports: Iterable[GateReport]) -> list[str]:
    out: list[str] = []
    for report in gate_reports:
        out.extend(report.failures)
    return out


def _aggregate_scores(report: RunReport) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in report.results:
        for s in r.scorer_outputs:
            totals[s.name] = totals.get(s.name, 0.0) + s.score
            counts[s.name] = counts.get(s.name, 0) + 1
    return {n: totals[n] / max(counts[n], 1) for n in totals}


def _aggregate_other(report: RunReport, name: str) -> float:
    if name == "p99_ms":
        lat = sorted(r.latency_ms for r in report.results)
        if not lat:
            return 0.0
        return lat[int(len(lat) * 0.99)]
    if name == "median_ms":
        lat = sorted(r.latency_ms for r in report.results)
        if not lat:
            return 0.0
        return lat[len(lat) // 2]
    if name == "cost_usd":
        return sum(r.cost_usd for r in report.results)
    if name == "error_rate":
        if not report.results:
            return 0.0
        return report.n_failures / len(report.results)
    return 0.0