"""Tests for the path-prefix write-deny wrapper."""

from __future__ import annotations

from langchain.tools import tool

from ada_lg.lg.permissions import wrap_write_deny


@tool
async def fake_write_file(path: str, content: str) -> str:
    """Fake write tool."""
    del content
    return f"wrote {path}"


async def test_allows_write_outside_denied_prefix() -> None:
    """Writes outside denied prefixes are delegated."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    result = await wrapped.ainvoke({"path": "/workspace/foo.txt", "content": "x"})
    assert result == "wrote /workspace/foo.txt"


async def test_blocks_write_under_denied_prefix() -> None:
    """Writes under denied prefixes return an error."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    result = await wrapped.ainvoke({"path": "/skills/profiler/SKILL.md", "content": "x"})
    assert "denied" in result.lower()
    assert "/skills/" in result


async def test_blocks_write_with_double_leading_slash() -> None:
    """Leading-slash variants normalize to the denied prefix."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    result = await wrapped.ainvoke({"path": "//skills/profiler/SKILL.md", "content": "x"})
    assert "denied" in result.lower()


async def test_blocks_write_at_denied_prefix_root() -> None:
    """The prefix root itself is denied even without a trailing slash."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    result = await wrapped.ainvoke({"path": "/skills", "content": "x"})
    assert "denied" in result.lower()


async def test_blocks_write_with_dot_segments() -> None:
    """Paths containing ``.`` segments normalize before deny matching."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    result = await wrapped.ainvoke({"path": "/skills/./profiler/SKILL.md", "content": "x"})
    assert "denied" in result.lower()


def test_wrapped_tool_keeps_name() -> None:
    """Wrapped tools preserve their public name."""
    wrapped = wrap_write_deny(fake_write_file, prefixes=("/skills/",))
    assert wrapped.name == "fake_write_file"
