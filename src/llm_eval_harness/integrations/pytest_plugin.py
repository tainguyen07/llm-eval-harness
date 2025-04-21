"""Pytest plugin — `llm_eval` fixture and `--llm-eval-config` option."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_eval_harness.config import EvalConfig
from llm_eval_harness.core.run import run as run_eval


@pytest.fixture(scope="session")
def llm_eval_config(request: pytest.FixtureRequest) -> EvalConfig:
    config_path = request.config.getoption("--llm-eval-config") or "config/default.yaml"
    return EvalConfig.from_yaml(config_path)


@pytest.fixture
def llm_eval_report(request: pytest.FixtureRequest, llm_eval_config: EvalConfig) -> Iterator:
    completed: list = []

    async def _factory(**overrides):  # type: ignore[no-untyped-def]
        run_spec = llm_eval_config.run.model_copy(update=overrides)
        report = await run_eval(
            name=run_spec.name,
            prompt=run_spec.prompt_template or "",
            dataset=run_spec.dataset_path,
            predict=lambda ex: _stub_predict(ex),
            scorers=run_spec.scorers,
            concurrency=run_spec.concurrency,
            output_dir=run_spec.output_dir,
        )
        completed.append(report)
        return report

    yield _factory


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-eval-config",
        action="store",
        default="config/default.yaml",
        help="Path to llm-eval-harness YAML config.",
    )


async def _stub_predict(example):  # type: ignore[no-untyped-def]
    return (str(example.inputs.get("expected", "")) or "", 0.0)