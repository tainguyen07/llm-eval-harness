"""Webhook notifier — Slack-compatible JSON payload + retries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from llm_eval_harness.core.models import RunReport
from llm_eval_harness.evaluation.diff import RegressionDiff


class WebhookNotifier:
    def __init__(self, url: str, *, timeout: float = 10.0, max_retries: int = 3) -> None:
        self._url = url
        self._timeout = timeout
        self._max_retries = max_retries

    def post(self, payload: Mapping[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = httpx.post(self._url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc

    def send_diff(self, diff_result: RegressionDiff, *, channel: str | None = None) -> httpx.Response:
        payload = dict(diff_result.to_dict())
        if channel:
            payload["channel"] = channel
        return self.post(payload)


def notify_slack(
    url: str,
    report: RunReport,
    diff_result: RegressionDiff | None = None,
    *,
    title: str | None = None,
) -> httpx.Response:
    """Format a Slack-compatible payload (works with most incoming-webhook relays)."""
    title = title or f"Run {report.name} — {report.summary().splitlines()[0]}"
    fields = [
        {"title": "Examples", "value": str(report.n), "short": True},
        {"title": "Failures", "value": str(report.n_failures), "short": True},
    ]
    if diff_result is not None:
        fields.append({"title": "Regressions", "value": str(len(diff_result.regressions)), "short": True})
        fields.append({"title": "Improvements", "value": str(len(diff_result.improvements)), "short": True})
    payload = {
        "text": title,
        "attachments": [{"fields": fields}],
    }
    notifier = WebhookNotifier(url)
    return notifier.post(payload)


@dataclass
class RegressionThresholds:
    min_accuracy: float | None = None
    max_p99_ms: float | None = None
    max_cost_usd: float | None = None

    def to_gates(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.min_accuracy is not None:
            out["min_overall"] = self.min_accuracy
        if self.max_p99_ms is not None:
            out["max_p99_ms"] = self.max_p99_ms
        if self.max_cost_usd is not None:
            out["max_cost_usd"] = self.max_cost_usd
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> RegressionThresholds:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**(raw or {})) if isinstance(raw, dict) else cls()

    def to_json(self) -> str:
        return json.dumps(
            {
                "min_accuracy": self.min_accuracy,
                "max_p99_ms": self.max_p99_ms,
                "max_cost_usd": self.max_cost_usd,
            }
        )