"""Deterministic scorers — exact match, contains, regex, Levenshtein, etc."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Mapping

from llm_eval_harness.registry import registry


def _norm(value: str | None) -> str:
    return (value or "").strip()


@registry.scorer("exact_match")
def exact_match(prediction: str, expected: str) -> tuple[float, dict[str, bool]]:
    """Exact equality after stripping."""
    return (1.0 if _norm(prediction) == _norm(expected) else 0.0), {"equal": _norm(prediction) == _norm(expected)}


@registry.scorer("contains")
def contains(prediction: str, expected: str) -> tuple[float, dict[str, bool]]:
    needle = _norm(expected)
    return (1.0 if needle and needle in _norm(prediction) else 0.0), {"found": needle in _norm(prediction)}


@registry.scorer("regex")
def regex_match(pattern: str, prediction: str) -> tuple[float, dict[str, bool]]:
    return (1.0 if re.search(pattern, prediction or "") else 0.0), {"matched": bool(re.search(pattern, prediction or ""))}


@registry.scorer("fuzzy_match")
def fuzzy_match(prediction: str, expected: str, *, threshold: float = 0.85) -> tuple[float, dict[str, float]]:
    ratio = SequenceMatcher(None, _norm(prediction), _norm(expected)).ratio()
    return (1.0 if ratio >= threshold else ratio), {"ratio": round(ratio, 4)}


@registry.scorer("levenshtein")
def levenshtein(prediction: str, expected: str) -> tuple[float, dict[str, int]]:
    return _lev_score(prediction or "", expected or "")


@registry.scorer("token_f1")
def token_f1(prediction: str, expected: str) -> tuple[float, dict[str, float]]:
    pred_tokens = _norm(prediction).split()
    exp_tokens = _norm(expected).split()
    if not pred_tokens and not exp_tokens:
        return 1.0, {"precision": 1.0, "recall": 1.0}
    common = Counter(pred_tokens) & Counter(exp_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, {"precision": 0.0, "recall": 0.0}
    precision = num_same / len(pred_tokens)
    recall = num_same / len(exp_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, {"precision": round(precision, 4), "recall": round(recall, 4)}


@registry.scorer("numeric_tolerance")
def numeric_tolerance(prediction: str, expected: str, *, atol: float = 1e-3) -> tuple[float, dict[str, float]]:
    try:
        a = float(_norm(prediction))
        b = float(_norm(expected))
    except ValueError:
        return 0.0, {"delta": float("inf")}
    delta = abs(a - b)
    return (1.0 if delta <= atol else 0.0), {"delta": delta}


@registry.scorer("json_schema")
def json_schema(schema: Mapping[str, Any], prediction: str) -> tuple[float, dict[str, Any]]:
    try:
        import jsonschema

    except ImportError:
        return 0.0, {"error": "jsonschema not installed"}
    try:
        instance = json.loads(prediction)
    except json.JSONDecodeError:
        return 0.0, {"error": "prediction is not valid JSON"}
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if not errors:
        return 1.0, {"valid": True}
    return 0.0, {"errors": [e.message for e in errors]}


def _lev_score(a: str, b: str) -> tuple[float, dict[str, int]]:
    if a == b:
        return 1.0, {"distance": 0}
    if not a or not b:
        return 0.0, {"distance": max(len(a), len(b))}
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    distance = prev[-1]
    score = 1.0 - distance / max(len(a), len(b))
    return score, {"distance": distance}