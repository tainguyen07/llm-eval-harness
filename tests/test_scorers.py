"""Tests for deterministic scorers."""

from __future__ import annotations

import pytest

from llm_eval_harness.core.errors import ScorerError
from llm_eval_harness.core.models import Example, ExampleResult
from llm_eval_harness.registry import registry
from llm_eval_harness.scorers.apply import score_example
from llm_eval_harness.scorers.deterministic import (
    contains,
    exact_match,
    fuzzy_match,
    json_schema,
    levenshtein,
    numeric_tolerance,
    regex_match,
    token_f1,
)


@pytest.fixture
def pair() -> tuple[Example, ExampleResult]:
    example = Example(id="x", inputs={}, expected="Paris")
    result = ExampleResult(example_id="x", prediction="Paris", latency_ms=1.0)
    return example, result


def test_exact_match_registered() -> None:
    assert "exact_match" in registry.list_scorers()


def test_exact_match(pair) -> None:
    score, _ = exact_match("Paris", "Paris")
    assert score == 1.0
    score, _ = exact_match("Paris.", "Paris")
    assert score == 0.0


def test_contains(pair) -> None:
    score, _ = contains("The capital is Paris, France", "Paris")
    assert score == 1.0


def test_regex_match() -> None:
    score, _ = regex_match(r"^\d+$", "12345")
    assert score == 1.0
    score, _ = regex_match(r"^\d+$", "abc")
    assert score == 0.0


def test_fuzzy_match_threshold() -> None:
    score, details = fuzzy_match("Paris", "Paris", threshold=0.85)
    assert score == 1.0
    assert details["ratio"] == 1.0


def test_levenshtein() -> None:
    score, details = levenshtein("kitten", "sitting")
    assert 0.0 < score < 1.0
    assert details["distance"] == 3


def test_levenshtein_identical() -> None:
    score, details = levenshtein("abc", "abc")
    assert score == 1.0
    assert details["distance"] == 0


def test_token_f1_perfect() -> None:
    score, details = token_f1("the cat sat", "the cat sat")
    assert score == 1.0
    assert details["precision"] == 1.0


def test_token_f1_zero_overlap() -> None:
    score, _ = token_f1("apple", "orange")
    assert score == 0.0


def test_token_f1_both_empty() -> None:
    score, _ = token_f1("", "")
    assert score == 1.0


def test_numeric_tolerance_passes() -> None:
    score, _ = numeric_tolerance("3.1416", "3.1415", atol=1e-3)
    assert score == 1.0


def test_numeric_tolerance_fails() -> None:
    score, _ = numeric_tolerance("3.2", "3.5", atol=1e-3)
    assert score == 0.0


def test_numeric_tolerance_invalid() -> None:
    score, _ = numeric_tolerance("not a number", "5")
    assert score == 0.0


def test_score_example_applies_named_scorers(pair) -> None:
    outputs = score_example(pair[0], pair[1], ["exact_match"])
    assert len(outputs) == 1
    assert outputs[0].name == "exact_match"
    assert outputs[0].score == 1.0


def test_score_example_unknown_scorer_raises(pair) -> None:
    with pytest.raises(ScorerError):
        score_example(pair[0], pair[1], ["nope"])


def test_json_schema_valid() -> None:
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    score, _ = json_schema(schema, '{"name": "Ada"}')
    assert score == 1.0


def test_json_schema_invalid_json() -> None:
    score, _ = json_schema({"type": "object"}, "not json")
    assert score == 0.0