"""Tests for regression diffing and gates."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llm_eval_harness.core.models import ExampleResult, RunReport, ScorerOutput
from llm_eval_harness.evaluation.diff import diff
from llm_eval_harness.evaluation.gates import evaluate_gates


def _make_report(
    name: str,
    scores: dict[str, float],
    *,
    n: int = 4,
    latency_ms: float = 100.0,
    cost_usd: float = 0.001,
    errors: int = 0,
) -> RunReport:
    results = []
    for i in range(n):
        outputs = [ScorerOutput(name=name, score=scores[name]) for name in scores]
        results.append(
            ExampleResult(
                example_id=f"e{i}",
                prediction="ok",
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                scorer_outputs=outputs,
                error="boom" if i < errors else None,
            )
        )
    return RunReport(
        name=name,
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
        model="m",
        dataset_path=None,
        dataset_size=n,
        concurrency=1,
        results=results,
        scorer_names=tuple(scores),
    )


def test_diff_no_regression() -> None:
    base = _make_report("a", {"x": 0.8})
    head = _make_report("b", {"x": 0.82})
    diff_obj = diff(base, head)
    assert not diff_obj.regressions
    assert diff_obj.improvements


def test_diff_detects_regression() -> None:
    base = _make_report("a", {"x": 0.9})
    head = _make_report("b", {"x": 0.7})
    diff_obj = diff(base, head)
    assert diff_obj.regressions
    assert diff_obj.is_regression()


def test_diff_latency_and_cost() -> None:
    base = _make_report("a", {"x": 0.8}, latency_ms=100, cost_usd=0.001)
    head = _make_report("b", {"x": 0.8}, latency_ms=150, cost_usd=0.002)
    diff_obj = diff(base, head)
    assert diff_obj.latency_delta_ms == 50.0
    assert diff_obj.cost_delta_usd == pytest.approx(0.001, rel=1e-3)


def test_gates_pass() -> None:
    report = _make_report("x", {"x": 0.8})
    gate_report = evaluate_gates(report, {"min_x": 0.5, "max_p99_ms": 5000.0})
    assert gate_report.passed
    assert not gate_report.failures


def test_gates_fail() -> None:
    report = _make_report("x", {"x": 0.4})
    gate_report = evaluate_gates(report, {"min_x": 0.5})
    assert not gate_report.passed
    assert gate_report.failures


def test_gates_unknown_key_raises() -> None:
    report = _make_report("x", {"x": 0.4})
    with pytest.raises(ValueError):
        evaluate_gates(report, {"bogus_key": 0.5})


def test_gates_aggregate_metrics() -> None:
    report = _make_report("x", {"x": 0.9}, n=10, errors=1, latency_ms=200)
    gate_report = evaluate_gates(report, {"max_error_rate": 0.2, "max_median_ms": 500.0})
    assert gate_report.passed


def test_diff_to_dict_round_trip() -> None:
    base = _make_report("a", {"x": 0.8})
    head = _make_report("b", {"x": 0.85})
    diff_obj = diff(base, head)
    d = diff_obj.to_dict()
    assert d["base_id"] == base.run_id
    assert d["head_id"] == head.run_id
    assert "scorer_deltas" in d