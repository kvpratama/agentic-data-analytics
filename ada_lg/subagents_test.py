"""Tests for the ada_lg subagent specs."""

from __future__ import annotations

from ada.config import Settings
from ada_lg.subagents import get_subagent_specs


def test_returns_three_named_specs() -> None:
    """The pure-LangGraph port has the same three subagents."""
    specs = get_subagent_specs(Settings())
    names = [spec["name"] for spec in specs]
    assert names == ["profiler", "cleaner", "analyst"]


def test_each_spec_has_required_fields() -> None:
    """Each spec carries the prompt and skill directory."""
    specs = get_subagent_specs(Settings())
    for spec in specs:
        assert set(spec).issuperset({"name", "system_prompt", "skill_dir"})
        assert spec["skill_dir"].startswith("/skills/")
        assert spec["system_prompt"].strip()
