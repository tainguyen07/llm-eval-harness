"""Tests for the prompt registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_harness.core.errors import PromptError
from llm_eval_harness.prompts import Prompt, PromptRegistry


def test_register_creates_prompt(registry: PromptRegistry) -> None:
    prompt = registry.register("qa", "Answer: {q}", version="v1")
    assert isinstance(prompt, Prompt)
    assert prompt.name == "qa"
    assert prompt.variables == ("q",)


def test_render_python_template(registry: PromptRegistry) -> None:
    prompt = registry.by_name("answer")
    assert prompt.render({"question": "Capital?"}) == "Answer: Capital?"


def test_render_jinja_template(registry: PromptRegistry) -> None:
    prompt = registry.register("explain", "Explain {{ topic }} to {{ audience }}.")
    assert prompt.render({"topic": "DNA", "audience": "a 5yo"}) == "Explain DNA to a 5yo."


def test_missing_variable_raises(registry: PromptRegistry) -> None:
    prompt = registry.register("needs", "Hi {{ name }}, age {{ age }}")
    with pytest.raises(PromptError):
        prompt.render({"name": "Ada"})


def test_extract_variables() -> None:
    from llm_eval_harness.prompts.registry import extract_variables

    assert extract_variables("Hello {x} and {y}") == ("x", "y")
    assert extract_variables("Hi {{ user }} from {{ place }}") == ("place", "user")


def test_diff_two_versions(registry: PromptRegistry) -> None:
    a = registry.register("v", "Hello {name}", version="v1")
    b = registry.register("v", "Hello, {name}!", version="v2")
    diff = a.diff(b)
    assert "Hello {name}" in diff
    assert "Hello, {name}!" in diff


def test_rollback_creates_new_version(registry: PromptRegistry, tmp_path: Path) -> None:
    reg = PromptRegistry(base_dir=tmp_path / "prompts")
    reg.register("x", "first {y}", version="v1")
    reg.register("x", "second {y}", version="v2")
    rolled = reg.rollback("x", "v1")
    assert rolled.version == "v3"
    assert rolled.template == "first {y}"


def test_persistence(tmp_path: Path) -> None:
    reg = PromptRegistry(base_dir=tmp_path / "prompts")
    prompt = reg.register("perma", "echo {x}", version="v1")
    on_disk = list((tmp_path / "prompts").glob("*.json"))
    assert len(on_disk) == 1
    import json

    payload = json.loads(on_disk[0].read_text(encoding="utf-8"))
    assert payload["content_hash"] == prompt.content_hash


def test_get_missing_raises(registry: PromptRegistry) -> None:
    with pytest.raises(PromptError):
        registry.by_name("nope")