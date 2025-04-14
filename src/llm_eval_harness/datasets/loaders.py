"""Dataset loaders for JSONL, CSV, dicts, and Hugging Face datasets."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_eval_harness.core.errors import DatasetError
from llm_eval_harness.core.models import Example


def load_dataset(path: str | Path, *, limit: int | None = None) -> list[Example]:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = from_jsonl(p, limit=limit)
    elif suffix == ".csv":
        rows = from_csv(p, limit=limit)
    elif suffix in {".json"}:
        rows = from_jsonl(p, limit=limit)
    else:
        raise DatasetError(f"Unsupported dataset format: {suffix!r}")
    return rows


def from_jsonl(path: str | Path, *, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSONL on line {line_no} of {path}: {exc}") from exc
            examples.append(_to_example(record, line_no))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def from_csv(path: str | Path, *, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, start=1):
            examples.append(_to_example(row, row_no))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def from_dicts(records: Iterable[Mapping[str, Any]]) -> list[Example]:
    return [_to_example(dict(record), i) for i, record in enumerate(records, start=1)]


def from_huggingface(dataset: Any, *, split: str | None = None) -> list[Example]:
    try:
        ds = dataset[split] if split else dataset
    except (KeyError, TypeError) as exc:
        raise DatasetError(f"Hugging Face dataset lookup failed: {exc}") from exc

    out: list[Example] = []
    for i, row in enumerate(ds):
        out.append(_to_example(dict(row), i))
    return out


def split_dataset(
    examples: Sequence[Example], *, ratio: float = 0.2, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    if not 0 < ratio < 1:
        raise DatasetError("ratio must be in (0, 1)")
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    cut = int(len(examples) * ratio)
    test = [examples[i] for i in indices[:cut]]
    train = [examples[i] for i in indices[cut:]]
    return train, test


def _to_example(record: Mapping[str, Any], index: int) -> Example:
    record = dict(record)
    if "id" not in record:
        record["id"] = f"row-{index}"
    inputs = record.get("inputs") or {
        k: v for k, v in record.items() if k not in {"id", "expected", "metadata"}
    }
    metadata = record.get("metadata", {})
    return Example(
        id=str(record["id"]),
        inputs=dict(inputs),
        expected=record.get("expected"),
        metadata=dict(metadata),
    )