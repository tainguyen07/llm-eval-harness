"""Typed data models passed between runner, scorers, and reporters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    inputs: Mapping[str, Any]
    expected: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ScorerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float
    passed: bool | None = None
    details: Mapping[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    label: str | None = None
    reasoning: str | None = None


class ExampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    prediction: str
    latency_ms: float
    cost_usd: float = 0.0
    scorer_outputs: list[ScorerOutput] = Field(default_factory=list)
    verdict: Verdict | None = None
    error: str | None = None
    trace: Mapping[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    finished_at: datetime | None = None
    prompt_id: str | None = None
    prompt_template: str | None = None
    model: str
    dataset_path: Path
    dataset_size: int
    concurrency: int
    results: list[ExampleResult] = Field(default_factory=list)
    scorer_names: tuple[str, ...] = Field(default_factory=tuple)
    artifacts_dir: Path | None = None

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_failures(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def summary(self) -> str:
        scorer_avg: dict[str, float] = {}
        for r in self.results:
            for s in r.scorer_outputs:
                scorer_avg[s.name] = scorer_avg.get(s.name, 0.0) + s.score
        for name, total in scorer_avg.items():
            scorer_avg[name] = round(total / max(self.n, 1), 4)

        lat = sorted(r.latency_ms for r in self.results)
        median = lat[len(lat) // 2] if lat else 0.0
        p99 = lat[int(len(lat) * 0.99)] if lat else 0.0

        lines = [
            f"Run '{self.name}' finished: {self.n} examples, {self.n_failures} failures",
        ]
        for name, score in scorer_avg.items():
            lines.append(f"  {name}: {score}")
        lines.append(f"  median latency: {median:.0f} ms")
        lines.append(f"  p99 latency: {p99:.0f} ms")
        cost = sum(r.cost_usd for r in self.results)
        lines.append(f"  cost: ${cost:.3f}")
        if self.artifacts_dir:
            lines.append(f"Report: {self.artifacts_dir}")
        return "\n".join(lines)