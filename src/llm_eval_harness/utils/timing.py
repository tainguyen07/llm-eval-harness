"""Timing helpers — context manager and decorator."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    """Tiny reusable timer. Reads elapsed_ms() at any point."""

    started_at: float | None = None
    stopped_at: float | None = None

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.stopped_at = None

    def stop(self) -> None:
        self.stopped_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at or time.perf_counter()
        return (end - self.started_at) * 1000.0


@contextmanager
def time_block() -> Iterator[Timer]:
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def timed(fn: Callable[..., object]) -> Callable[..., tuple[object, float]]:
    def wrapper(*args: object, **kwargs: object) -> tuple[object, float]:
        with time_block() as timer:
            result = fn(*args, **kwargs)
        return result, timer.elapsed_ms()

    return wrapper