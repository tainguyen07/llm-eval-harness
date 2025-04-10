"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_harness.config import EvalConfig, JudgeConfig, RetryConfig, RunSpec
from llm_eval_harness.core.errors import EvalHarnessError


def test_run_spec_requires_prompt(tmp_path: Path) -> None:
    with pytest.raises(EvalHarnessError):
        RunSpec(name="x", dataset_path=tmp_path / "d.jsonl")


def test_judge_config_defaults() -> None:
    cfg = JudgeConfig(model="gpt-4o-mini")
    assert cfg.temperature == 0.0
    assert cfg.adapter == "stub"


def test_retry_config_validates() -> None:
    with pytest.raises(Exception):
        RetryConfig(attempts=0)


def test_eval_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
run:
  name: qa
  prompt_template: "Q: {q}"
  dataset_path: /tmp/d.jsonl
  concurrency: 4
  scorers: [exact_match]
gates:
  min_exact_match: 0.5
report:
  html: true
  markdown: false
""",
        encoding="utf-8",
    )
    cfg = EvalConfig.from_yaml(path)
    assert cfg.run.name == "qa"
    assert cfg.run.concurrency == 4
    assert cfg.gates["min_exact_match"] == 0.5
    assert cfg.report["markdown"] is False


def test_eval_config_rejects_unknown_report_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "run:\n  name: x\n  prompt_template: 'x'\n  dataset_path: /tmp/d.jsonl\nreport:\n  nonsense: true\n",
        encoding="utf-8",
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvalConfig.from_yaml(path)