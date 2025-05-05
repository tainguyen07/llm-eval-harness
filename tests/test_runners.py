"""Tests for runners and retry policy."""

from __future__ import annotations

import asyncio

import pytest

from llm_eval_harness.core.errors import EvalHarnessError
from llm_eval_harness.runners import AsyncRunner, RetryPolicy, ThreadRunner


@pytest.mark.asyncio
async def test_async_runner_returns_all_results(examples) -> None:
    async def predict(example):
        return (str(example.inputs["x"]), 0.0)

    runner = AsyncRunner(concurrency=2)
    results = await runner.gather(examples, predict)
    assert len(results) == 3
    assert {r.example_id for r in results} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_async_runner_records_failure(examples) -> None:
    async def predict(example):
        if example.id == "b":
            raise RuntimeError("boom")
        return ("ok", 0.0)

    runner = AsyncRunner(concurrency=2, retry=RetryPolicy(max_attempts=1))
    results = await runner.gather(examples, predict)
    failed = [r for r in results if r.error]
    assert len(failed) == 1
    assert failed[0].example_id == "b"


@pytest.mark.asyncio
async def test_async_runner_retries_then_succeeds(examples) -> None:
    state = {"calls": 0}

    async def predict(example):
        if example.id == "a":
            state["calls"] += 1
            if state["calls"] < 2:
                raise RuntimeError("transient")
        return ("ok", 0.0)

    runner = AsyncRunner(concurrency=1, retry=RetryPolicy(max_attempts=3, initial_backoff=0.0, jitter=0.0))
    results = await runner.gather(examples, predict)
    assert all(r.error is None for r in results)
    assert state["calls"] == 2


def test_async_runner_invalid_concurrency() -> None:
    with pytest.raises(EvalHarnessError):
        AsyncRunner(concurrency=0)


def test_thread_runner_gathers(examples) -> None:
    runner = ThreadRunner(concurrency=2)
    results = runner.gather(examples, lambda ex: (str(ex.inputs["x"]), 0.0))
    assert len(results) == 3


def test_retry_policy_backoff_grows() -> None:
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=8.0, jitter=0.0)
    assert policy.backoff_for(1) == 1.0
    assert policy.backoff_for(2) == 2.0
    assert policy.backoff_for(3) == 4.0
    assert policy.backoff_for(4) == 8.0
    assert policy.backoff_for(10) == 8.0


def test_retry_policy_validates() -> None:
    with pytest.raises(Exception):
        RetryPolicy(max_attempts=0)


@pytest.mark.asyncio
async def test_async_runner_stream_preserves_set(examples) -> None:
    async def predict(example):
        return ("ok", 0.0)

    runner = AsyncRunner(concurrency=2)
    collected = []
    async for result in runner.stream(examples, predict):
        collected.append(result)
    assert {r.example_id for r in collected} == {"a", "b", "c"}