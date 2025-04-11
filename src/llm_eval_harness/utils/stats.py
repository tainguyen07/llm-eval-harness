"""Statistical helpers — percentiles, latency summaries, win-rate intervals."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return ordered[floor]
    return ordered[floor] + (ordered[ceil] - ordered[floor]) * (k - floor)


@dataclass
class LatencySummary:
    n: int
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "median_ms": round(self.median_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
        }


def summarize_latency(latencies_ms: Iterable[float]) -> LatencySummary:
    values = list(latencies_ms)
    if not values:
        return LatencySummary(0, 0, 0, 0, 0, 0, 0)
    return LatencySummary(
        n=len(values),
        median_ms=percentile(values, 0.5),
        p90_ms=percentile(values, 0.9),
        p95_ms=percentile(values, 0.95),
        p99_ms=percentile(values, 0.99),
        max_ms=max(values),
        mean_ms=sum(values) / len(values),
    )


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)