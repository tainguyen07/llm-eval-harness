"""Lightweight in-process registry for prompts, scorers, and judge adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class Registry:
    _prompts: dict[str, Any] = field(default_factory=dict)
    _scorers: dict[str, Callable[..., Any]] = field(default_factory=dict)
    _judges: dict[str, type] = field(default_factory=dict)
    _datasets: dict[str, Any] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def prompt(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            with self._lock:
                self._prompts[name] = obj
            return obj

        return decorator

    def scorer(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            with self._lock:
                self._scorers[name] = fn
            return fn

        return decorator

    def judge(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            with self._lock:
                self._judges[name] = cls
            return cls

        return decorator

    def dataset(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            with self._lock:
                self._datasets[name] = fn
            return fn

        return decorator

    def get_scorer(self, name: str) -> Callable[..., Any]:
        with self._lock:
            try:
                return self._scorers[name]
            except KeyError as exc:
                raise KeyError(f"Scorer not registered: {name!r}") from exc

    def get_judge(self, name: str) -> type:
        with self._lock:
            try:
                return self._judges[name]
            except KeyError as exc:
                raise KeyError(f"Judge adapter not registered: {name!r}") from exc

    def list_scorers(self) -> list[str]:
        with self._lock:
            return sorted(self._scorers)

    def list_judges(self) -> list[str]:
        with self._lock:
            return sorted(self._judges)

    def list_prompts(self) -> list[str]:
        with self._lock:
            return sorted(self._prompts)


registry = Registry()