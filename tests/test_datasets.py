"""Tests for dataset loaders and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_harness.core.errors import DatasetError
from llm_eval_harness.datasets import (
    from_csv,
    from_dicts,
    from_jsonl,
    load_dataset,
    split_dataset,
)
from llm_eval_harness.datasets.preprocess import (
    lowercase,
    normalize_whitespace,
    preprocess,
    truncate_by_tokens,
)


def test_load_jsonl(dataset_path: Path) -> None:
    examples = load_dataset(dataset_path)
    assert len(examples) == 3
    assert examples[0].inputs == {"question": "Capital of France?"}
    assert examples[0].expected == "Paris"


def test_load_jsonl_with_limit(dataset_path: Path) -> None:
    examples = load_dataset(dataset_path, limit=2)
    assert len(examples) == 2


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope.jsonl")


def test_load_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "foo.bin"
    path.write_bytes(b"")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_invalid_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        from_jsonl(path)


def test_csv_loader(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(
        "id,question,expected\n" "q1,Capital?,Paris\n" "q2,2+2?,4\n",
        encoding="utf-8",
    )
    rows = from_csv(path)
    assert len(rows) == 2
    assert rows[1].expected == "4"


def test_from_dicts() -> None:
    rows = from_dicts([
        {"id": "a", "inputs": {"x": 1}, "expected": "1"},
        {"inputs": {"y": 2}, "expected": "2"},
    ])
    assert len(rows) == 2
    assert rows[1].id == "row-2"


def test_split_dataset(examples: list) -> None:
    train, test = split_dataset(examples, ratio=0.34, seed=42)
    assert len(test) == 1
    assert len(train) == 2


def test_split_invalid_ratio(examples: list) -> None:
    with pytest.raises(DatasetError):
        split_dataset(examples, ratio=1.5)


def test_lowercase() -> None:
    ex = examples := next(iter([__import__("llm_eval_harness").Example(id="x", inputs={"q": "WHY?"}, expected="YES")]))
    out = lowercase(ex)
    assert out.inputs["q"] == "why?"
    assert out.expected == "yes"


def test_normalize_whitespace() -> None:
    from llm_eval_harness import Example

    ex = Example(id="x", inputs={"q": "  hello   world  "}, expected="  ok  ")
    out = normalize_whitespace(ex)
    assert out.inputs["q"] == "hello world"
    assert out.expected == "ok"


def test_truncate_by_tokens() -> None:
    from llm_eval_harness import Example

    ex = Example(id="x", inputs={"q": "one two three four five"}, expected=None)
    trunc = truncate_by_tokens(max_tokens=3)
    out = trunc(ex)
    assert out.inputs["q"] == "one two three"


def test_truncate_zero_raises() -> None:
    with pytest.raises(ValueError):
        truncate_by_tokens(max_tokens=0)


def test_preprocess_compose() -> None:
    from llm_eval_harness import Example

    ex = Example(id="x", inputs={"q": "  HELLO  "}, expected="  WORLD  ")
    out = preprocess(ex, [lowercase, normalize_whitespace])
    assert out.inputs["q"] == "hello"
    assert out.expected == "world"