"""Preprocessing helpers applied to model inputs and references."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from llm_eval_harness.core.models import Example


def preprocess(example: Example, transforms: Iterable[Callable[[Example], Example]]) -> Example:
    current = example
    for transform in transforms:
        current = transform(current)
    return current


def lowercase(example: Example) -> Example:
    inputs = {key: _stringify(value).lower() for key, value in example.inputs.items()}
    return example.model_copy(
        update={
            "inputs": inputs,
            "expected": example.expected.lower() if example.expected else None,
        }
    )


def normalize_whitespace(example: Example) -> Example:
    pattern = re.compile(r"\s+")
    inputs = {key: pattern.sub(" ", _stringify(value)).strip() for key, value in example.inputs.items()}
    return example.model_copy(
        update={
            "inputs": inputs,
            "expected": pattern.sub(" ", example.expected).strip() if example.expected else None,
        }
    )


def truncate_by_tokens(max_tokens: int, tokenizer: Callable[[str], list[str]] | None = None) -> Callable[[Example], Example]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    def _truncate(example: Example) -> Example:
        out: dict[str, Any] = {}
        for key, value in example.inputs.items():
            text = _stringify(value)
            tokens = (tokenizer or _naive_tokens)(text)
            if len(tokens) > max_tokens:
                sep = getattr(tokenizer, "_sep", " ")
                if sep in {" "}:
                    text = sep.join(tokens[:max_tokens])
                else:
                    text = "".join(tokens[:max_tokens])
            out[key] = text
        return example.model_copy(update={"inputs": out})

    return _truncate


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _naive_tokens(text: str) -> list[str]:
    return text.split()