"""Prompt registry — content-addressed versions, render, diff, rollback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any

from llm_eval_harness.core.errors import PromptError
from llm_eval_harness.utils.hashing import content_hash


@dataclass(frozen=True)
class Prompt:
    id: str
    name: str
    version: str
    template: str
    variables: tuple[str, ...]
    created_at: datetime
    content_hash: str

    def render(self, variables: Mapping[str, Any]) -> str:
        try:
            return render_template(self.template, dict(variables))
        except KeyError as exc:
            raise PromptError(
                f"Missing variable {exc.args[0]!r} for prompt {self.name}@{self.version}"
            ) from exc

    def diff(self, other: Prompt) -> str:
        import difflib

        a = self.template.splitlines(keepends=True)
        b = other.template.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(a, b, fromfile=f"{self.name}@{self.version}", tofile=f"{other.name}@{other.version}")
        )


def render_template(template: str, variables: Mapping[str, Any]) -> str:
    if "{{" in template or "{%" in template:
        return _render_jinja(template, variables)
    return _render_python(template, variables)


def _render_python(template: str, variables: Mapping[str, Any]) -> str:
    return Template(template).safe_substitute(variables)


def _render_jinja(template: str, variables: Mapping[str, Any]) -> str:
    from jinja2 import Environment, StrictUndefined, meta

    env = Environment(undefined=StrictUndefined, autoescape=False)
    parsed = env.parse(template)
    required = sorted(meta.find_undeclared_variables(parsed))
    missing = [v for v in required if v not in variables]
    if missing:
        raise PromptError(f"Missing Jinja variables: {missing}")
    return env.from_string(template).render(**variables)


def extract_variables(template: str) -> tuple[str, ...]:
    if "{{" in template or "{%" in template:
        from jinja2 import Environment, meta

        env = Environment()
        return tuple(sorted(meta.find_undeclared_variables(env.parse(template))))
    return tuple(sorted(Template(template).get_identifiers()))


@dataclass
class PromptRegistry:
    base_dir: Path | None = None
    _prompts: dict[str, Prompt] = field(default_factory=dict)

    def register(
        self,
        name: str,
        template: str,
        *,
        version: str = "v1",
        metadata: Mapping[str, Any] | None = None,
    ) -> Prompt:
        variables = extract_variables(template)
        digest = content_hash({"name": name, "version": version, "template": template})
        prompt = Prompt(
            id=digest,
            name=name,
            version=version,
            template=template,
            variables=variables,
            created_at=datetime.now(tz=timezone.utc),
            content_hash=digest,
        )
        self._prompts[digest] = prompt
        self._prompts[f"{name}@{version}"] = prompt
        if self.base_dir is not None:
            self._persist(prompt, dict(metadata or {}))
        return prompt

    def get(self, key: str) -> Prompt:
        try:
            return self._prompts[key]
        except KeyError as exc:
            raise PromptError(f"Prompt not found: {key!r}") from exc

    def by_name(self, name: str, version: str = "v1") -> Prompt:
        return self.get(f"{name}@{version}")

    def rollback(self, name: str, version: str) -> Prompt:
        prior = self.by_name(name, version)
        return self.register(name, prior.template, version=_next_version(version, self._versions(name)))

    def _versions(self, name: str) -> Iterable[str]:
        return [
            p.version
            for p in self._prompts.values()
            if p.name == name and p.version.startswith("v") and p.version[1:].isdigit()
        ]

    def _persist(self, prompt: Prompt, metadata: Mapping[str, Any]) -> None:
        assert self.base_dir is not None
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{prompt.name}__{prompt.version}.json"
        payload = {
            "id": prompt.id,
            "name": prompt.name,
            "version": prompt.version,
            "template": prompt.template,
            "variables": list(prompt.variables),
            "created_at": prompt.created_at.isoformat(),
            "content_hash": prompt.content_hash,
            "metadata": dict(metadata),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _next_version(current: str, existing: Iterable[str]) -> str:
    nums = [int(v[1:]) for v in existing if v[1:].isdigit()]
    nxt = max(nums, default=0) + 1
    return f"v{nxt}"


def content_hash(payload: Mapping[str, Any]) -> str:  # re-export for convenience
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()