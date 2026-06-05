"""Path-prefix write-deny wrapper for filesystem tools."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from typing import Any

from langchain.tools import BaseTool, tool


def _normalize(path: str) -> str:
    """Collapse leading slashes and posix-normalize ``path``.

    Args:
        path: Raw path argument supplied by the model.

    Returns:
        A path with a single leading slash and ``..``/``.`` segments resolved.
        Non-absolute inputs are returned with a single leading slash prepended.
    """
    collapsed = "/" + path.lstrip("/")
    return posixpath.normpath(collapsed)


def _is_denied(path: str, prefixes: Sequence[str]) -> str | None:
    """Return the matching prefix if ``path`` falls under any denied prefix."""
    normalized = _normalize(path)
    for prefix in prefixes:
        norm_prefix = _normalize(prefix).rstrip("/") or "/"
        if normalized == norm_prefix or normalized.startswith(norm_prefix + "/"):
            return prefix
    return None


def wrap_write_deny(inner: BaseTool, *, prefixes: Sequence[str]) -> BaseTool:
    """Wrap a write/edit tool so denied path prefixes return an error.

    Path matching normalizes leading slashes and ``..``/``.`` segments so that
    inputs like ``//skills/x`` or ``/skills/./x`` are treated equivalently to
    ``/skills/x``. Both the prefix root (``/skills``) and any descendant
    (``/skills/...``) are denied, mirroring the reference ``/skills/**``
    semantics.

    Args:
        inner: Tool with a ``path`` argument.
        prefixes: Absolute path prefixes to deny.

    Returns:
        A guarded tool with the same public name and schema.
    """

    @tool(inner.name, description=inner.description, args_schema=inner.args_schema)
    async def guarded(**kwargs: Any) -> Any:
        path = kwargs.get("path")
        if isinstance(path, str):
            matched = _is_denied(path, prefixes)
            if matched is not None:
                return f"Error: write denied for path under {matched}"
        return await inner.ainvoke(kwargs)

    return guarded
