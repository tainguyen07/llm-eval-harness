"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from llm_eval_harness import Example, PromptRegistry


@pytest.fixture
def dataset_path(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        '{"id": "q1", "inputs": {"question": "Capital of France?"}, "expected": "Paris"}\n'
        '{"id": "q2", "inputs": {"question": "2 + 2?"}, "expected": "4"}\n'
        '{"id": "q3", "inputs": {"question": "Boiling point of water?"}, "expected": "100"}\n',
        encoding="utf-8",
    )
    yield path


@pytest.fixture
def examples() -> list[Example]:
    return [
        Example(id="a", inputs={"x": 1}, expected="1"),
        Example(id="b", inputs={"x": 2}, expected="2"),
        Example(id="c", inputs={"x": 3}, expected="3"),
    ]


@pytest.fixture
def registry() -> PromptRegistry:
    reg = PromptRegistry()
    reg.register("summarize", "Summarize: {{text}}")
    reg.register("answer", "Answer: {question}")
    return reg


@pytest.fixture
async def async_iter_helper() -> AsyncIterator[int]:
    for i in range(3):
        yield i