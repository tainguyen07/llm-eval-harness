"""Tests for reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_eval_harness.core.models import ExampleResult, RunReport, ScorerOutput
from llm_eval_harness.reporting.html import render_html, write_html
from llm_eval_harness.reporting.markdown import render_markdown, write_markdown
from llm_eval_harness.reporting.serialize import (
    write_artifacts,
    write_csv,
    write_json,
    write_junit,
)


def _make_report(*, with_failures: bool = False) -> RunReport:
    results = [
        ExampleResult(
            example_id="e1",
            prediction="Paris",
            latency_ms=120.0,
            cost_usd=0.001,
            scorer_outputs=[ScorerOutput(name="exact_match", score=1.0, passed=True)],
        ),
        ExampleResult(
            example_id="e2",
            prediction="London",
            latency_ms=80.0,
            cost_usd=0.0008,
            scorer_outputs=[ScorerOutput(name="exact_match", score=0.0, passed=False)],
            error="boom" if with_failures else None,
        ),
    ]
    return RunReport(
        name="test",
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
        model="m",
        dataset_path=Path("/tmp/data.jsonl"),
        dataset_size=2,
        concurrency=4,
        results=results,
        scorer_names=("exact_match",),
        artifacts_dir=Path("/tmp/run"),
    )


def test_render_html_contains_run_name() -> None:
    html = render_html(_make_report())
    assert "Run" in html
    assert "exact_match" in html


def test_render_html_no_embed() -> None:
    html = render_html(_make_report(), embed=False)
    assert "Raw report JSON" in html


def test_render_markdown() -> None:
    md = render_markdown(_make_report())
    assert "Run `test`" in md
    assert "exact_match" in md
    assert "Latency" in md


def test_write_html(tmp_path: Path) -> None:
    report = _make_report()
    report.artifacts_dir = tmp_path
    out = write_html(report, tmp_path / "report.html")
    assert out.exists()
    assert "<html" in out.read_text(encoding="utf-8")


def test_write_json(tmp_path: Path) -> None:
    report = _make_report()
    out = write_json(report, tmp_path / "report.json")
    payload = out.read_text(encoding="utf-8")
    assert "exact_match" in payload


def test_write_csv(tmp_path: Path) -> None:
    report = _make_report()
    out = write_csv(report, tmp_path / "report.csv")
    text = out.read_text(encoding="utf-8")
    assert "e1,Paris" in text or "e2,London" in text


def test_write_junit(tmp_path: Path) -> None:
    report = _make_report()
    out = write_junit(report, tmp_path / "junit.xml")
    text = out.read_text(encoding="utf-8")
    assert "<testsuite" in text
    assert "tests=" in text


def test_write_artifacts_all(tmp_path: Path) -> None:
    report = _make_report()
    report.artifacts_dir = tmp_path
    written = write_artifacts(report)
    assert "html" in written
    assert "json" in written
    assert "csv" in written
    assert "markdown" in written
    assert "junit" in written


def test_write_artifacts_subset(tmp_path: Path) -> None:
    report = _make_report()
    report.artifacts_dir = tmp_path
    written = write_artifacts(report, formats=("json", "csv"))
    assert set(written) == {"json", "csv"}


def test_markdown_includes_failure_block() -> None:
    report = _make_report(with_failures=True)
    md = render_markdown(report)
    assert "Failures" in md
    assert "boom" in md


def test_summary_format() -> None:
    report = _make_report()
    summary = report.summary()
    assert "exact_match" in summary
    assert "ms" in summary


def test_summary_with_zero_examples() -> None:
    report = RunReport(
        name="empty",
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
        model="m",
        dataset_path=Path("/tmp/data.jsonl"),
        dataset_size=0,
        concurrency=1,
    )
    summary = report.summary()
    assert "0 examples" in summary