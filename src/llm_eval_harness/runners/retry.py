"""Retry policy with exponential backoff and jitter."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff: float = Field(default=0.5, ge=0.0, le=30.0)
    max_backoff: float = Field(default=10.0, ge=0.0, le=120.0)
    multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: float = Field(default=0.1, ge=0.0, le=1.0)

    def backoff_for(self, attempt: int) -> float:
        delay = min(self.initial_backoff * (self.multiplier ** (attempt - 1)), self.max_backoff)
        if self.jitter > 0:
            delay += random.uniform(-self.jitter * delay, self.jitter * delay)
        return max(0.0, delay)