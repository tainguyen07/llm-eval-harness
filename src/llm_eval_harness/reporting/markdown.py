"""Markdown summary writer."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from llm_eval_harness.core.models import RunReport


def render_markdown(report: RunReport) -> str:
    out = StringIO()
    out.write(f"# Run `{report.name}`\n\n")
    out.write(f"- Run ID: `{report.run_id}`\n")
    out.write(f"- Model: `{report.model}`\n")
    out.write(f"- Dataset: `{report.dataset_path}` ({report.dataset_size} examples)\n")
    out.write(f"- Concurrency: {report.concurrency}\n")
    if report.prompt_template:
        out.write(f"- Prompt template:\n\n  ```\n  {report.prompt_template}\n  ```\n")
    out.write("\n## Scorers\n\n")
    out.write("| Scorer | Mean | Pass rate | n |\n")
    out.write("|---|---|---|---|\n")
    for name, stats in _aggregate(report).items():
        pass_rate = (stats["passes"] / stats["n"]) if stats["n"] else 0.0
        out.write(f"| {name} | {stats['mean']:.3f} | {pass_rate:.1%} | {stats['n']} |\n")

    latencies = sorted(r.latency_ms for r in report.results)
    if latencies:
        out.write("\n## Latency\n\n")
        out.write(f"- median: {latencies[len(latencies) // 2]:.0f} ms\n")
        out.write(f"- p99: {latencies[int(len(latencies) * 0.99)]:.0f} ms\n")
        out.write(f"- max: {latencies[-1]:.0f} ms\n")

    total_cost = sum(r.cost_usd for r in report.results)
    out.write(f"\n## Cost\n\n- ${total_cost:.4f} total\n")

    failures = [r for r in report.results if r.error]
    if failures:
        out.write("\n## Failures\n\n")
        out.write(f"{len(failures)} examples failed:\n\n")
        for r in failures[:10]:
            out.write(f"- `{r.example_id}`: {r.error}\n")
        if len(failures) > 10:
            out.write(f"- …and {len(failures) - 10} more\n")
    return out.getvalue()


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


def write_markdown(report: RunReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(report), encoding="utf-8")
    return p