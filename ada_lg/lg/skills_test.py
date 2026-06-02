"""Tests for the skills loader and read_skill tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend

from ada_lg.lg.skills import build_skills


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    """Create a fake profiler skills tree on disk."""
    base = tmp_path / "profiler_skills"
    (base / "profile_basics").mkdir(parents=True)
    (base / "profile_basics" / "SKILL.md").write_text(
        "---\nname: profile-basics\ndescription: Run basic profiling\n---\n# Basics\nDo X then Y.\n"
    )
    (base / "advanced_stats").mkdir(parents=True)
    (base / "advanced_stats" / "SKILL.md").write_text(
        "---\nname: advanced-stats\ndescription: Compute extra stats\n---\n# Advanced\nUse scipy.\n"
    )
    return tmp_path


def _backend(root: Path) -> FilesystemBackend:
    """Build a virtual filesystem backend for tests."""
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)


def test_build_skills_produces_index_text(skills_root: Path) -> None:
    """build_skills creates a prompt index from frontmatter."""
    backend = _backend(skills_root)
    prompt_index, _tool = build_skills(backend, skill_dir="/profiler_skills/")

    assert "profile-basics" in prompt_index
    assert "Run basic profiling" in prompt_index
    assert "advanced-stats" in prompt_index


async def test_read_skill_returns_body(skills_root: Path) -> None:
    """read_skill returns content below frontmatter."""
    backend = _backend(skills_root)
    _index, read_skill = build_skills(backend, skill_dir="/profiler_skills/")

    body = await read_skill.ainvoke({"name": "profile-basics"})
    assert "# Basics" in body
    assert "Do X then Y" in body


async def test_read_skill_returns_error_for_unknown(skills_root: Path) -> None:
    """Unknown skill names return an error string."""
    backend = _backend(skills_root)
    _index, read_skill = build_skills(backend, skill_dir="/profiler_skills/")

    body = await read_skill.ainvoke({"name": "does-not-exist"})
    assert "Error" in body or "unknown" in body.lower()
