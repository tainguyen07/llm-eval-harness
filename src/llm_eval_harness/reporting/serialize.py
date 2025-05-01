"""CSV, JSON, and JUnit serializers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_eval_harness.core.models import RunReport


def write_json(report: RunReport, path: str | Path | None = None) -> Path:
    p = Path(path) if path else (report.artifacts_dir or Path("runs")) / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p


def write_csv(report: RunReport, path: str | Path | None = None) -> Path:
    p = Path(path) if path else (report.artifacts_dir or Path("runs")) / "report.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["example_id", "prediction", "latency_ms", "cost_usd", "error", "scorers"]
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.results:
            writer.writerow(
                {
                    "example_id": r.example_id,
                    "prediction": r.prediction,
                    "latency_ms": round(r.latency_ms, 3),
                    "cost_usd": round(r.cost_usd, 6),
                    "error": r.error or "",
                    "scorers": json.dumps({s.name: s.score for s in r.scorer_outputs}),
                }
            )
    return p


def write_junit(report: RunReport, path: str | Path | None = None) -> Path:
    p = Path(path) if path else (report.artifacts_dir or Path("runs")) / "junit.xml"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="{report.name}" tests="{report.n}" failures="{report.n_failures}">',
    ]
    for r in report.results:
        status = "failure" if r.error else "success"
        name = r.example_id
        lines.append(f'  <testcase classname="run" testname="{name}">')
        if r.error:
            lines.append(f'    <failure message="{_xml_escape(r.error)}" />')
        lines.append(f'    <system-out>{_xml_escape(r.prediction[:200])}</system-out>')
        lines.append("  </testcase>")
        _ = status
    lines.append("</testsuite>")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_artifacts(report: RunReport, *, formats: tuple[str, ...] = ("html", "markdown", "json", "csv", "junit")) -> dict[str, Path]:
    from llm_eval_harness.reporting.html import write_html
    from llm_eval_harness.reporting.markdown import write_markdown

    base = report.artifacts_dir or Path("runs") / report.run_id
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "html" in formats:
        written["html"] = write_html(report, base / "report.html")
    if "markdown" in formats:
        written["markdown"] = write_markdown(report, base / "report.md")
    if "json" in formats:
        written["json"] = write_json(report, base / "report.json")
    if "csv" in formats:
        written["csv"] = write_csv(report, base / "report.csv")
    if "junit" in formats:
        written["junit"] = write_junit(report, base / "junit.xml")
    return written


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )