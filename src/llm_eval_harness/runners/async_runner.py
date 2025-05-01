"""Asyncio-based runner with bounded concurrency and per-example retries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from llm_eval_harness.core.errors import EvalHarnessError
from llm_eval_harness.core.models import Example, ExampleResult
from llm_eval_harness.runners.retry import RetryPolicy


class AsyncRunner:
    def __init__(self, *, concurrency: int = 8, retry: RetryPolicy | None = None) -> None:
        if concurrency < 1:
            raise EvalHarnessError("concurrency must be >= 1")
        self._sem = asyncio.Semaphore(concurrency)
        self._retry = retry or RetryPolicy()

    async def stream(
        self,
        examples: Sequence[Example],
        predict: Callable[[Example], Awaitable[tuple[str, float]]],
    ) -> AsyncIterator[ExampleResult]:
        tasks = [asyncio.create_task(self._run_one(example, predict)) for example in examples]
        for task in asyncio.as_completed(tasks):
            yield await task

    async def gather(
        self,
        examples: Sequence[Example],
        predict: Callable[[Example], Awaitable[tuple[str, float]]],
    ) -> list[ExampleResult]:
        return [result async for result in self.stream(examples, predict)]

    async def _run_one(
        self,
        example: Example,
        predict: Callable[[Example], Awaitable[tuple[str, float]]],
    ) -> ExampleResult:
        async with self._sem:
            attempt = 0
            last_error: str | None = None
            while attempt < self._retry.max_attempts:
                attempt += 1
                started = time.perf_counter()
                try:
                    prediction, cost = await predict(example)
                    latency = (time.perf_counter() - started) * 1000.0
                    return ExampleResult(
                        example_id=example.id,
                        prediction=prediction,
                        latency_ms=latency,
                        cost_usd=cost,
                    )
                except Exception as exc:
                    last_error = repr(exc)
                    if attempt >= self._retry.max_attempts:
                        break
                    await asyncio.sleep(self._retry.backoff_for(attempt))
            return ExampleResult(
                example_id=example.id,
                prediction="",
                latency_ms=0.0,
                cost_usd=0.0,
                error=last_error,
            )