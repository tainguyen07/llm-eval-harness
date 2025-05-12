"""Quickstart: run an evaluation against a local JSONL dataset."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_eval_harness import run


async def echo_predict(example):  # type: ignore[no-untyped-def]
    """Echo back the expected answer (good for a smoke test)."""
    return (str(example.inputs.get("expected", "")), 0.0)


async def main() -> None:
    report = await run(
        name="qa-smoke",
        prompt="Answer concisely: {question}",
        dataset=Path("examples/data/qa.jsonl"),
        predict=echo_predict,
        scorers=["exact_match", "token_f1"],
        concurrency=4,
        output_dir="runs",
    )
    print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())