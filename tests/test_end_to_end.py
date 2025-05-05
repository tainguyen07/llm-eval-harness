"""End-to-end smoke test for `run()`."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_harness import run


@pytest.mark.asyncio
async def test_run_end_to_end(tmp_path: Path) -> None:
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        '{"id": "1", "inputs": {"question": "q1"}, "expected": "a1"}\n'
        '{"id": "2", "inputs": {"question": "q2"}, "expected": "a2"}\n',
        encoding="utf-8",
    )

    async def predict(example):
        return (str(example.inputs["question"]).replace("q", "a"), 0.0)

    report = await run(
        name="e2e",
        prompt="Answer: {question}",
        dataset=dataset,
        predict=predict,
        scorers=["exact_match"],
        concurrency=2,
        output_dir=tmp_path / "runs",
    )
    assert report.n == 2
    assert report.n_failures == 0
    assert report.summary()
    assert report.artifacts_dir is not None
    assert (report.artifacts_dir / "report.html").exists()
    assert (report.artifacts_dir / "report.json").exists()


@pytest.mark.asyncio
async def test_run_with_inline_examples() -> None:
    from llm_eval_harness import Example

    examples = [Example(id=str(i), inputs={"q": "q"}, expected="a") for i in range(3)]

    async def predict(example):
        return ("a", 0.0)

    report = await run(
        name="inline",
        prompt="Answer: {q}",
        dataset=examples,
        predict=predict,
        scorers=["exact_match"],
    )
    assert report.n == 3
    assert all(r.error is None for r in report.results)


@pytest.mark.asyncio
async def test_run_recovers_from_predict_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"id": "1", "inputs": {"q": "q"}, "expected": "a"}\n', encoding="utf-8")

    async def predict(example):
        raise RuntimeError("fail")

    report = await run(
        name="fail",
        prompt="x",
        dataset=dataset,
        predict=predict,
        scorers=[],
        concurrency=1,
        output_dir=tmp_path / "runs",
    )
    assert report.n_failures == 1