"""Content hashing utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def short_hash(payload: Mapping[str, Any], *, length: int = 8) -> str:
    return content_hash(payload)[:length]


def stable_id(*parts: Any) -> str:
    encoded = "|".join(str(p) for p in parts)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()  # noqa: S324 — short, non-security ID