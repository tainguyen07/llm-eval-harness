"""Top-level `run()` facade — wires dataset, runner, scorers, and reporters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from llm_eval_harness.core.errors import DatasetError, EvalHarnessError
from llm_eval_harness.core.models import Example, ExampleResult, RunReport
from llm_eval_harness.datasets import load_dataset
from llm_eval_harness.prompts import render_template
from llm_eval_harness.reporting import write_artifacts
from llm_eval_harness.runners.async_runner import AsyncRunner
from llm_eval_harness.runners.thread_runner import ThreadRunner
from llm_eval_harness.scorers import apply_scorers

PredictFn = Callable[[Example], Awaitable[tuple[str, float]]]


async def run(
    name: str,
    *,
    prompt: str,
    dataset: Sequence[Example] | str | Path,
    predict: PredictFn,
    scorers: Sequence[str] = (),
    concurrency: int = 8,
    judge: Any | None = None,
    output_dir: str | Path = "runs",
    use_threads: bool = False,
) -> RunReport:
    """Run an evaluation end-to-end and return a typed `RunReport`."""
    if not name:
        raise EvalHarnessError("`name` must be a non-empty string")

    examples = _coerce_dataset(dataset)
    template = prompt

    started_at = datetime.now(tz=timezone.utc)

    async def _wrapped_predict(example: Example) -> tuple[str, float]:
        rendered = render_template(template, dict(example.inputs))
        return await predict(_synthetic_example(example, rendered))

    runner: AsyncRunner | ThreadRunner = (
        ThreadRunner(concurrency=concurrency) if use_threads else AsyncRunner(concurrency=concurrency)
    )
    results: list[ExampleResult] = []
    async for result in runner.stream(examples, _wrapped_predict):
        results.append(result)

    if scorers:
        results = await apply_scorers(results, examples, scorers)

    report = RunReport(
        name=name,
        started_at=started_at,
        finished_at=datetime.now(tz=timezone.utc),
        prompt_template=template,
        model=getattr(predict, "__name__", "predict"),
        dataset_path=Path(str(dataset)) if isinstance(dataset, (str, Path)) else Path("<inline>"),
        dataset_size=len(examples),
        concurrency=concurrency,
        results=results,
        scorer_names=tuple(scorers),
    )

    artifacts = Path(output_dir) / report.run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    report.artifacts_dir = artifacts
    write_artifacts(report)
    return report


def _coerce_dataset(dataset: Sequence[Example] | str | Path) -> list[Example]:
    if isinstance(dataset, (str, Path)):
        return load_dataset(dataset)
    return list(dataset)


def _synthetic_example(example: Example, rendered_prompt: str) -> Example:
    return example.model_copy(update={"inputs": {"prompt": rendered_prompt, **dict(example.inputs)}})