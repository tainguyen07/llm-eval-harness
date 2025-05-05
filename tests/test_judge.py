"""Tests for LLM-as-judge adapters and pairwise comparison."""

from __future__ import annotations

import pytest

from llm_eval_harness.scorers.judge import JudgeVerdict, StubJudge, judge_batch
from llm_eval_harness.scorers.pairwise import compare_pairwise


@pytest.mark.asyncio
async def test_stub_judge_basic() -> None:
    judge = StubJudge()
    verdict = await judge.judge("p", "Some long answer text")
    assert isinstance(verdict, JudgeVerdict)
    assert 0.0 <= verdict.score <= 1.0


@pytest.mark.asyncio
async def test_stub_judge_empty() -> None:
    judge = StubJudge()
    verdict = await judge.judge("p", "")
    assert verdict.label == "empty"
    assert verdict.score == 0.0


@pytest.mark.asyncio
async def test_stub_judge_with_reference() -> None:
    judge = StubJudge()
    verdict = await judge.judge("p", "Paris is the capital", reference="Paris is the capital")
    assert verdict.score > 0.95


@pytest.mark.asyncio
async def test_judge_batch_concurrency() -> None:
    judge = StubJudge()
    items = [("p", f"answer {i}", None) for i in range(8)]
    results = await judge_batch(judge, items, concurrency=4)
    assert len(results) == 8


@pytest.mark.asyncio
async def test_pairwise_basic() -> None:
    judge = StubJudge()
    pairs = [("p", "4", "four", "4"), ("p", "Paris", "London", "Paris")]
    result = await compare_pairwise(judge, pairs)
    assert result.a_wins + result.b_wins + result.ties == 2
    assert 0.0 <= result.win_rate <= 1.0


@pytest.mark.asyncio
async def test_pairwise_empty_raises() -> None:
    from llm_eval_harness.core.errors import ScorerError

    judge = StubJudge()
    with pytest.raises(ScorerError):
        await compare_pairwise(judge, [])


def test_judge_registry_has_adapters() -> None:
    from llm_eval_harness.registry import registry

    judges = registry.list_judges()
    assert "stub" in judges
    assert "openai" in judges
    assert "anthropic" in judges