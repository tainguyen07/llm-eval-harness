"""HTML reporter — self-contained single-file report with embedded JSON."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_eval_harness.core.models import RunReport

_TEMPLATE_NAME = "report.html.j2"


def render_html(report: RunReport, *, embed: bool = True) -> str:
    template = _load_template()
    payload = {
        "report": report,
        "scores": _aggregate(report),
        "summary": report.summary(),
        "data_json": report.model_dump_json() if embed else "",
    }
    return template.render(**payload)


def _aggregate(report: RunReport) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in report.results:
        for s in r.scorer_outputs:
            slot = out.setdefault(s.name, {"sum": 0.0, "n": 0, "passes": 0})
            slot["sum"] += s.score
            slot["n"] += 1
            if s.passed:
                slot["passes"] += 1
    final: dict[str, dict[str, float]] = {}
    for name, slot in out.items():
        n = max(slot["n"], 1)
        final[name] = {
            "mean": round(slot["sum"] / n, 4),
            "passes": slot["passes"],
            "n": slot["n"],
        }
    return final


def _load_template() -> Any:
    base = resources.files("llm_eval_harness") / "templates"
    env = Environment(loader=FileSystemLoader(str(base)), autoescape=select_autoescape(["html"]))
    return env.get_template(_TEMPLATE_NAME)


def write_html(report: RunReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(report), encoding="utf-8")
    return p