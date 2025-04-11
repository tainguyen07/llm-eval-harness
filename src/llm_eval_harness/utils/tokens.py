"""Token counting and cost estimation."""

from __future__ import annotations

_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005 / 1000, 0.015 / 1000),
    "gpt-4o-mini": (0.00015 / 1000, 0.0006 / 1000),
    "claude-opus-4-1": (0.015 / 1000, 0.075 / 1000),
    "claude-sonnet-4-5": (0.003 / 1000, 0.015 / 1000),
    "claude-haiku-4-5": (0.0008 / 1000, 0.004 / 1000),
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Best-effort tokenizer using a public regex; avoids the tiktoken hard dep.

    ponytail: real token counts need tiktoken; fall back to a 4-chars/token heuristic.
    Upgrade: pass a tiktoken encoder for exact counts.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0
    in_cost, out_cost = pricing
    return prompt_tokens * in_cost + completion_tokens * out_cost


def register_pricing(model: str, prompt_per_1k: float, completion_per_1k: float) -> None:
    _PRICING[model] = (prompt_per_1k / 1000, completion_per_1k / 1000)