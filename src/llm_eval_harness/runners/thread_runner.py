"""Thread-pool runner — same surface as AsyncRunner for sync predictors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llm_eval_harness.core.errors import EvalHarnessError
from llm_eval_harness.core.models import Example, ExampleResult
from llm_eval_harness.runners.retry import RetryPolicy


class ThreadRunner:
    def __init__(self, *, concurrency: int = 8, retry: RetryPolicy | None = None) -> None:
        if concurrency < 1:
            raise EvalHarnessError("concurrency must be >= 1")
        self._concurrency = concurrency
        self._retry = retry or RetryPolicy()

    def gather(
        self,
        examples: Sequence[Example],
        predict: Callable[[Example], tuple[str, float]],
    ) -> list[ExampleResult]:
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            return list(pool.map(lambda ex: self._run_sync(ex, predict), examples))

    def _run_sync(
        self,
        example: Example,
        predict: Callable[[Example], tuple[str, float]],
    ) -> ExampleResult:
        attempt = 0
        last_error: str | None = None
        while attempt < self._retry.max_attempts:
            attempt += 1
            started = time.perf_counter()
            try:
                prediction, cost = predict(example)
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
                time.sleep(self._retry.backoff_for(attempt))
        return ExampleResult(
            example_id=example.id,
            prediction="",
            latency_ms=0.0,
            cost_usd=0.0,
            error=last_error,
        )


async def _to_async_result(result: ExampleResult) -> ExampleResult:
    return result


async def run_in_executor(runner: ThreadRunner, examples: Sequence[Example], predict: Callable[..., Any]) -> list[ExampleResult]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, runner.gather, examples, predict)