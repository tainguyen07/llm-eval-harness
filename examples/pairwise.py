"""Pairwise comparison example using the StubJudge."""

from __future__ import annotations

import asyncio

from llm_eval_harness.scorers.compare_pairwise import compare_pairwise
from llm_eval_harness.scorers.judge import StubJudge


async def main() -> None:
    judge = StubJudge()
    pairs = [
        ("What is 2+2?", "4", "four", "4"),
        ("Capital of France?", "Paris", "London", "Paris"),
        ("Author of Hamlet?", "Shakespeare", "William Shakespeare", "William Shakespeare"),
    ]
    result = await compare_pairwise(judge, pairs, concurrency=2)
    print(result.summary())


if __name__ == "__main__":
    asyncio.run(main())