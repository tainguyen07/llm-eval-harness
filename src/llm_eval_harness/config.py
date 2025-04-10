"""Configuration models for the evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_eval_harness.core.errors import EvalHarnessError


class RetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int = Field(default=3, ge=1, le=10)
    backoff: float = Field(default=0.5, ge=0.0, le=30.0)
    max_backoff: float = Field(default=10.0, ge=0.0, le=120.0)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    rubric: str | None = None
    adapter: str = Field(default="stub")


class RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    prompt_id: str | None = None
    prompt_template: str | None = None
    dataset_path: Path
    dataset_format: str = Field(default="jsonl")
    concurrency: int = Field(default=8, ge=1, le=512)
    scorers: tuple[str, ...] = Field(default_factory=tuple)
    judge: JudgeConfig | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    output_dir: Path = Field(default=Path("runs"))

    @model_validator(mode="after")
    def _one_prompt_source(self) -> RunSpec:
        if self.prompt_id is None and self.prompt_template is None:
            raise EvalHarnessError("RunSpec requires prompt_id or prompt_template")
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunSpec
    gates: dict[str, float] = Field(default_factory=dict)
    report: dict[str, bool] = Field(
        default_factory=lambda: {"html": True, "markdown": True, "csv": True, "json": True}
    )

    @field_validator("report")
    @classmethod
    def _report_keys(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"html", "markdown", "csv", "json", "junit"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown report formats: {sorted(unknown)}")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise EvalHarnessError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)