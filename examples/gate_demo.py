"""Run, then enforce gates, then exit non-zero on regression."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_eval_harness import run
from llm_eval_harness.evaluation.gates import evaluate_gates


async def echo_predict(example):  # type: ignore[no-untyped-def]
    return (str(example.inputs.get("expected", "")), 0.0)


async def main() -> None:
    report = await run(
        name="qa-gated",
        prompt="Answer: {question}",
        dataset=Path("examples/data/qa.jsonl"),
        predict=echo_predict,
        scorers=["exact_match"],
        output_dir="runs",
    )
    gates = evaluate_gates(report, {"min_exact_match": 0.6, "max_p99_ms": 5000.0})
    print(gates.to_dict())
    if not gates.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())