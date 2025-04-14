"""Apply registered scorers to example results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_eval_harness.core.errors import ScorerError
from llm_eval_harness.core.models import Example, ExampleResult, ScorerOutput
from llm_eval_harness.registry import registry


def score_example(
    example: Example,
    result: ExampleResult,
    scorer_names: Sequence[str],
) -> list[ScorerOutput]:
    outputs: list[ScorerOutput] = []
    for name in scorer_names:
        scorer = registry.get_scorer(name)
        try:
            score, details = _invoke(scorer, example, result)
        except ScorerError:
            raise
        except Exception as exc:
            raise ScorerError(f"Scorer {name!r} failed: {exc}") from exc
        outputs.append(
            ScorerOutput(
                name=name,
                score=float(score),
                passed=bool(score >= 0.5) if score is not None else None,
                details=dict(details or {}),
            )
        )
    return outputs


async def apply_scorers(
    results: Sequence[ExampleResult],
    examples: Sequence[Example],
    scorer_names: Sequence[str],
) -> list[ExampleResult]:
    lookup = {ex.id: ex for ex in examples}
    out: list[ExampleResult] = []
    for result in results:
        example = lookup.get(result.example_id)
        if example is None:
            out.append(result)
            continue
        scored = score_example(example, result, scorer_names)
        out.append(result.model_copy(update={"scorer_outputs": scored}))
    return out


def _invoke(scorer: Any, example: Example, result: ExampleResult) -> tuple[float | int, dict[str, Any] | None]:
    sig = _signature(scorer)
    if "example" in sig and "result" in sig:
        return scorer(example=example, result=result)
    if "prediction" in sig and "expected" in sig:
        return scorer(prediction=result.prediction, expected=example.expected or "")
    if "prediction" in sig:
        return scorer(prediction=result.prediction)
    raise ScorerError(f"Cannot determine scorer signature for {scorer!r}")


def _signature(callable_obj: Any) -> set[str]:
    import inspect

    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return set()
    return set(params)