"""Domain-specific exceptions used across the harness."""

from __future__ import annotations


class EvalHarnessError(Exception):
    """Base class for all exceptions raised by the harness."""


class DatasetError(EvalHarnessError):
    """Dataset loading, parsing, or validation failure."""


class PromptError(EvalHarnessError):
    """Prompt template resolution or rendering failure."""


class ScorerError(EvalHarnessError):
    """Scorer execution failure (not a low score, but a hard error)."""


class JudgeError(EvalHarnessError):
    """LLM-as-judge adapter or transport failure."""


class ConfigError(EvalHarnessError):
    """Invalid configuration supplied by the user."""