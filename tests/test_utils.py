"""Tests for utilities."""

from __future__ import annotations

import pytest

from llm_eval_harness.utils.hashing import content_hash, short_hash, stable_id
from llm_eval_harness.utils.stats import percentile, summarize_latency, wilson_interval
from llm_eval_harness.utils.timing import Timer, time_block
from llm_eval_harness.utils.tokens import count_tokens, estimate_cost, register_pricing


def test_content_hash_deterministic() -> None:
    a = content_hash({"b": 2, "a": 1})
    b = content_hash({"a": 1, "b": 2})
    assert a == b


def test_short_hash_length() -> None:
    assert len(short_hash({"x": 1})) == 8


def test_stable_id_changes_with_input() -> None:
    assert stable_id("a", "b") != stable_id("a", "c")


def test_percentile_basic() -> None:
    values = list(range(101))
    assert percentile(values, 0.5) == pytest.approx(50.0)
    assert percentile(values, 0.99) == pytest.approx(99.0)


def test_percentile_empty() -> None:
    assert percentile([], 0.5) == 0.0


def test_percentile_invalid_q() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_summarize_latency_empty() -> None:
    summary = summarize_latency([])
    assert summary.n == 0
    assert summary.median_ms == 0


def test_summarize_latency_values() -> None:
    summary = summarize_latency(range(100, 200))
    assert summary.n == 100
    assert 100 <= summary.median_ms <= 200


def test_wilson_interval() -> None:
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high


def test_wilson_interval_zero_n() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_timer_elapsed() -> None:
    timer = Timer()
    timer.start()
    elapsed = timer.elapsed_ms()
    assert elapsed >= 0


def test_time_block_records() -> None:
    with time_block() as timer:
        _ = sum(range(100))
    assert timer.elapsed_ms() >= 0


def test_count_tokens_zero() -> None:
    assert count_tokens("") == 0


def test_count_tokens_positive() -> None:
    assert count_tokens("hello world") > 0


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost(1000, 1000, "gpt-4o-mini")
    assert cost > 0


def test_estimate_cost_unknown_model() -> None:
    assert estimate_cost(1000, 1000, "unknown-model") == 0.0


def test_register_pricing() -> None:
    register_pricing("custom", 1.0, 2.0)
    assert estimate_cost(1000, 1000, "custom") == pytest.approx(3.0, rel=1e-3)