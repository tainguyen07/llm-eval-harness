"""Command-line interface built on Click."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from llm_eval_harness import __version__
from llm_eval_harness.config import EvalConfig
from llm_eval_harness.core.errors import EvalHarnessError
from llm_eval_harness.core.models import RunReport
from llm_eval_harness.core.run import run as run_eval
from llm_eval_harness.evaluation.diff import diff
from llm_eval_harness.evaluation.gates import evaluate_gates
from llm_eval_harness.prompts.registry import PromptRegistry
from llm_eval_harness.registry import registry
from llm_eval_harness.reporting.serialize import write_artifacts
from llm_eval_harness.utils.logging import configure_logging

console = Console()


@click.group()
@click.version_option(__version__, prog_name="llm-eval")
@click.option("--config", default=None, help="Path to YAML config.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--quiet", is_flag=True, help="Suppress non-error output.")
def main(config: str | None, verbose: bool, quiet: bool) -> None:
    """LLM evaluation harness."""
    level = "debug" if verbose else "error" if quiet else "info"
    configure_logging(level)


@main.command()
@click.option("--name", required=True, help="Run name.")
@click.option("--prompt", "prompt_text", default=None, help="Inline prompt template.")
@click.option("--prompt-id", default=None, help="Registered prompt id.")
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--model", default="stub", show_default=True)
@click.option("--scorer", "scorers", multiple=True)
@click.option("--concurrency", default=8, show_default=True)
@click.option("--output-dir", default="runs", show_default=True)
@click.option("--predict", default="echo", help="Predictor: echo | uppercase | stub.")
def run(
    name: str,
    prompt_text: str | None,
    prompt_id: str | None,
    dataset_path: str,
    model: str,
    scorers: tuple[str, ...],
    concurrency: int,
    output_dir: str,
    predict: str,
) -> None:
    """Run an evaluation against a dataset."""
    if not prompt_text and not prompt_id:
        raise click.UsageError("Either --prompt or --prompt-id is required.")
    template = prompt_text or PromptRegistry().by_name(prompt_id or "").template

    async def _predict(example):  # type: ignore[no-untyped-def]
        rendered = template.format(**example.inputs) if "{" in template else template
        if predict == "echo":
            return rendered, 0.0
        if predict == "uppercase":
            return rendered.upper(), 0.0
        return (rendered[: min(len(rendered), 64)], 0.0)

    report = asyncio.run(
        run_eval(
            name=name,
            prompt=template,
            dataset=Path(dataset_path),
            predict=_predict,
            scorers=scorers,
            concurrency=concurrency,
            output_dir=output_dir,
        )
    )
    console.print(report.summary())


@main.command("diff")
@click.argument("base", type=click.Path(exists=True))
@click.argument("head", type=click.Path(exists=True))
def diff_cmd(base: str, head: str) -> None:
    """Compare two previous runs."""
    result = diff(Path(base), Path(head))
    console.print_json(data=result.to_dict())
    if result.is_regression():
        sys.exit(2)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--format", "formats", multiple=True, default=("html", "markdown", "json", "csv", "junit"))
def report(run_dir: str, formats: tuple[str, ...]) -> None:
    """Re-render reports from a previous run."""
    json_path = Path(run_dir) / "report.json"
    if not json_path.exists():
        raise click.UsageError(f"No report.json in {run_dir}")
    report_obj = RunReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    written = write_artifacts(report_obj, formats=formats)
    for name, path in written.items():
        console.print(f"[green]wrote[/green] {name}: {path}")


@main.command()
@click.option("--name", required=True)
@click.option("--template", required=True)
@click.option("--version", "version", default="v1", show_default=True)
@click.option("--base-dir", default=".llm-eval/prompts", show_default=True)
def register(name: str, template: str, version: str, base_dir: str) -> None:
    """Register a prompt version."""
    reg = PromptRegistry(base_dir=Path(base_dir))
    prompt = reg.register(name, template, version=version)
    console.print(f"[green]registered[/green] {prompt.name}@{prompt.version} -> {prompt.content_hash[:8]}")


@main.command()
@click.argument("kind", type=click.Choice(["prompts", "scorers", "judges"]))
def list_cmd(kind: str) -> None:
    """List registered prompts, scorers, or judges."""
    table = Table(title=f"Registered {kind}")
    table.add_column("name")
    if kind == "scorers":
        for name in registry.list_scorers():
            table.add_row(name)
    elif kind == "judges":
        for name in registry.list_judges():
            table.add_row(name)
    else:
        for name in PromptRegistry()._prompts:
            table.add_row(name)
    console.print(table)


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def gates_cmd(config_path: str) -> None:
    """Evaluate gate conditions for a run."""
    cfg = EvalConfig.from_yaml(config_path)
    json_path = cfg.run.output_dir / "report.json"
    if not json_path.exists():
        raise click.UsageError(f"Run report not found at {json_path}; run `llm-eval run` first")
    report_obj = RunReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    gate_report = evaluate_gates(report_obj, cfg.gates)
    console.print_json(data=gate_report.to_dict())
    if not gate_report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main(obj={})