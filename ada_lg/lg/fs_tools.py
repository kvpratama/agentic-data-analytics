"""LangChain tools over a Deep Agents BackendProtocol."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import cast

from deepagents.backends.protocol import BackendProtocol, ExecuteResponse
from langchain.tools import BaseTool, tool


def _content_to_text(content: object) -> str:
    """Normalize backend file content to text."""
    if isinstance(content, list):
        return "\n".join(str(line) for line in content)
    return str(content)


def _entry_path(entry: Mapping[str, object]) -> str:
    """Return a display path/name from backend file info."""
    value = entry.get("path") or entry.get("name")
    return str(value)


def build_fs_tools(backend: BackendProtocol) -> list[BaseTool]:
    """Build filesystem tools bound to a backend instance.

    Args:
        backend: BackendProtocol implementation used for filesystem operations.

    Returns:
        LangChain tools for ls/read/write/edit/glob/grep plus execute when present.
    """

    @tool
    async def ls(path: str = "/") -> str:
        """List files and directories at the given absolute path."""
        result = await backend.als(path)
        if result.error:
            return f"Error: {result.error}"
        lines = []
        for entry in result.entries or []:
            item = cast("Mapping[str, object]", entry)
            marker = "/" if item.get("is_dir") is True or item.get("type") == "directory" else ""
            lines.append(f"{_entry_path(item)}{marker}\t{item.get('size', 0)}")
        return "\n".join(lines) if lines else "(empty)"

    @tool
    async def read_file(path: str) -> str:
        """Read a text file at the given absolute path and return its content."""
        result = await backend.aread(path)
        if result.error:
            return f"Error: {result.error}"
        if result.file_data is None:
            return ""
        return _content_to_text(result.file_data["content"])

    @tool
    async def write_file(path: str, content: str) -> str:
        """Create a file at the given absolute path with text content."""
        result = await backend.awrite(path, content)
        if result.error:
            return f"Error: {result.error}"
        return f"Wrote {result.path}"

    @tool
    async def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace old_string with new_string in a file."""
        result = await backend.aedit(path, old_string, new_string, replace_all)
        if result.error:
            return f"Error: {result.error}"
        return f"Edited {result.path} ({result.occurrences} replacement(s))"

    @tool
    async def glob(pattern: str, path: str = "/") -> str:
        """Find files whose paths match the given glob pattern."""
        result = await backend.aglob(pattern, path)
        if result.error:
            return f"Error: {result.error}"
        lines = [_entry_path(cast("Mapping[str, object]", match)) for match in result.matches or []]
        return "\n".join(lines) if lines else "(no matches)"

    @tool
    async def grep(pattern: str, path: str | None = None, glob: str | None = None) -> str:
        """Search for a literal text pattern across files."""
        result = await backend.agrep(pattern, path, glob)
        if result.error:
            return f"Error: {result.error}"
        lines = [
            f"{item.get('path') or item.get('file')}:"
            f"{item.get('line') or item.get('line_number')}: "
            f"{item.get('text') or item.get('line')}"
            for match in result.matches or []
            for item in [cast("Mapping[str, object]", match)]
        ]
        return "\n".join(lines) if lines else "(no matches)"

    tools: list[BaseTool] = [ls, read_file, write_file, edit_file, glob, grep]

    execute_fn = getattr(backend, "execute", None)
    if callable(execute_fn):
        sandbox_execute = cast("Callable[[str], ExecuteResponse]", execute_fn)

        @tool
        async def execute(command: str) -> str:
            """Run a shell command inside the sandbox and return its output."""
            response = await asyncio.to_thread(sandbox_execute, command)
            head = f"exit_code={response.exit_code}"
            if response.truncated:
                head += " (truncated)"
            return f"{head}\n{response.output}"

        tools.append(execute)

    return tools
