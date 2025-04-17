"""LLM-as-judge scorers — abstract Judge + adapters."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_eval_harness.core.errors import JudgeError
from llm_eval_harness.core.models import Verdict
from llm_eval_harness.registry import registry


@dataclass
class JudgeVerdict:
    score: float
    label: str | None = None
    reasoning: str | None = None

    def to_verdict(self) -> Verdict:
        return Verdict(score=self.score, label=self.label, reasoning=self.reasoning)


class Judge(ABC):
    """Abstract judge. Implementations live in adapters/."""

    name: str = "abstract"

    @abstractmethod
    async def judge(
        self,
        prompt: str,
        candidate: str,
        reference: str | None = None,
    ) -> JudgeVerdict:
        raise NotImplementedError


@registry.judge("stub")
class StubJudge(Judge):
    """Heuristic judge used for tests and offline runs."""

    name = "stub"

    async def judge(
        self,
        prompt: str,
        candidate: str,
        reference: str | None = None,
    ) -> JudgeVerdict:
        if not candidate:
            return JudgeVerdict(score=0.0, label="empty", reasoning="Empty prediction")
        score = min(1.0, len(candidate) / 200.0)
        if reference:
            from difflib import SequenceMatcher

            score = max(score, SequenceMatcher(None, candidate, reference).ratio())
        label = "ok" if score >= 0.5 else "weak"
        return JudgeVerdict(score=score, label=label, reasoning="stub judge")


@registry.judge("openai")
class OpenAIJudge(Judge):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise JudgeError("Install `openai` extra to use OpenAIJudge") from exc
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def judge(self, prompt: str, candidate: str, reference: str | None = None) -> JudgeVerdict:
        system = "You are a strict evaluator. Reply with JSON: {score: 0..1, label: str, reasoning: str}"
        user = f"PROMPT:\n{prompt}\n\nCANDIDATE:\n{candidate}\n\nREFERENCE:\n{reference or '-'}"
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        import json

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return JudgeVerdict(
            score=float(data.get("score", 0.0)),
            label=str(data.get("label", "")) or None,
            reasoning=str(data.get("reasoning", "")) or None,
        )


@registry.judge("anthropic")
class AnthropicJudge(Judge):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", *, api_key: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise JudgeError("Install `anthropic` extra to use AnthropicJudge") from exc
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def judge(self, prompt: str, candidate: str, reference: str | None = None) -> JudgeVerdict:
        msg = await self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Reply with JSON {{score, label, reasoning}}. Score 0..1.\n"
                        f"PROMPT: {prompt}\nCANDIDATE: {candidate}\nREFERENCE: {reference or '-'}"
                    ),
                }
            ],
        )
        import json

        text = msg.content[0].text if msg.content else "{}"
        data = json.loads(text)
        return JudgeVerdict(
            score=float(data.get("score", 0.0)),
            label=data.get("label"),
            reasoning=data.get("reasoning"),
        )


class LLMJudgeScorer:
    """Wrap a Judge so it can be referenced by name in run() / config."""

    def __init__(self, judge: Judge, *, rubric: str | None = None) -> None:
        self._judge = judge
        self._rubric = rubric

    async def __call__(self, prompt: str, candidate: str, reference: str | None = None) -> JudgeVerdict:
        verdict = await self._judge.judge(prompt, candidate, reference)
        if self._rubric:
            verdict.reasoning = (verdict.reasoning or "") + f"\nrubric: {self._rubric}"
        return verdict


async def judge_batch(
    judge: Judge,
    items: Sequence[tuple[str, str, str | None]],
    *,
    concurrency: int = 8,
) -> list[JudgeVerdict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(prompt: str, candidate: str, reference: str | None) -> JudgeVerdict:
        async with semaphore:
            return await judge.judge(prompt, candidate, reference)

    return await asyncio.gather(*(_one(p, c, r) for p, c, r in items))