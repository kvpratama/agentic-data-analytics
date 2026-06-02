"""Path-prefix write-deny wrapper for filesystem tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.tools import BaseTool, tool


def wrap_write_deny(inner: BaseTool, *, prefixes: Sequence[str]) -> BaseTool:
    """Wrap a write/edit tool so denied path prefixes return an error.

    Args:
        inner: Tool with a path argument.
        prefixes: Absolute path prefixes to deny.

    Returns:
        A guarded tool with the same public name and schema.
    """

    @tool(inner.name, description=inner.description, args_schema=inner.args_schema)
    async def guarded(**kwargs: Any) -> Any:
        path = kwargs.get("path")
        if isinstance(path, str):
            for prefix in prefixes:
                if path.startswith(prefix):
                    return f"Error: write denied for path under {prefix}"
        return await inner.ainvoke(kwargs)

    return guarded
