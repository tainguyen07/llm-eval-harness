"""llm-eval-harness — production-grade evaluation harness for LLM applications."""

from llm_eval_harness.config import EvalConfig, JudgeConfig, RetryConfig, RunSpec
from llm_eval_harness.core.errors import (
    DatasetError,
    EvalHarnessError,
    JudgeError,
    PromptError,
    ScorerError,
)
from llm_eval_harness.core.models import (
    Example,
    ExampleResult,
    RunReport,
    ScorerOutput,
    Verdict,
)
from llm_eval_harness.core.run import run
from llm_eval_harness.datasets import load_dataset
from llm_eval_harness.evaluation import gates
from llm_eval_harness.evaluation.diff import diff
from llm_eval_harness.prompts import Prompt, PromptRegistry
from llm_eval_harness.registry import registry
from llm_eval_harness.reporting import render_html, render_markdown
from llm_eval_harness.runners.async_runner import AsyncRunner

__version__ = "0.4.2"

__all__ = [
    "__version__",
    "AsyncRunner",
    "DatasetError",
    "EvalConfig",
    "EvalHarnessError",
    "Example",
    "ExampleResult",
    "JudgeConfig",
    "JudgeError",
    "Prompt",
    "PromptError",
    "PromptRegistry",
    "RetryConfig",
    "RunReport",
    "RunSpec",
    "ScorerError",
    "ScorerOutput",
    "Verdict",
    "diff",
    "gates",
    "load_dataset",
    "registry",
    "render_html",
    "render_markdown",
    "run",
]