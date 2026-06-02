"""Tests for the filesystem tool layer."""

from __future__ import annotations

from unittest.mock import AsyncMock

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)

from ada_lg.lg.fs_tools import build_fs_tools


async def test_ls_calls_backend_als_and_formats_entries() -> None:
    """ls delegates to BackendProtocol.als and renders file names."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.als.return_value = LsResult(
        entries=[
            {
                "path": "/workspace/dataset.csv",
                "is_dir": False,
                "size": 1024,
                "modified_at": "2026-06-01T00:00:00",
            },
            {
                "path": "/workspace/plots",
                "is_dir": True,
                "size": 0,
                "modified_at": "2026-06-01T00:00:00",
            },
        ]
    )
    tools = build_fs_tools(backend)
    ls = next(t for t in tools if t.name == "ls")

    result = await ls.ainvoke({"path": "/workspace"})

    backend.als.assert_awaited_once_with("/workspace")
    assert "dataset.csv" in result
    assert "plots" in result


async def test_read_file_returns_content() -> None:
    """read_file returns text content from BackendProtocol.aread."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.aread.return_value = ReadResult(
        file_data={
            "content": "line1\nline2",
            "encoding": "utf-8",
            "created_at": "x",
            "modified_at": "x",
        }
    )
    tools = build_fs_tools(backend)
    read_file = next(t for t in tools if t.name == "read_file")

    result = await read_file.ainvoke({"path": "/workspace/dataset.csv"})

    backend.aread.assert_awaited_once_with("/workspace/dataset.csv")
    assert "line1" in result
    assert "line2" in result


async def test_read_file_returns_error_message() -> None:
    """read_file surfaces backend errors to the model."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.aread.return_value = ReadResult(error="file_not_found")
    tools = build_fs_tools(backend)
    read_file = next(t for t in tools if t.name == "read_file")

    result = await read_file.ainvoke({"path": "/nope"})

    assert "Error" in result
    assert "file_not_found" in result


async def test_write_file_delegates_to_awrite() -> None:
    """write_file delegates to BackendProtocol.awrite."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.awrite.return_value = WriteResult(path="/workspace/foo.txt")
    tools = build_fs_tools(backend)
    write_file = next(t for t in tools if t.name == "write_file")

    result = await write_file.ainvoke({"path": "/workspace/foo.txt", "content": "hello"})

    backend.awrite.assert_awaited_once_with("/workspace/foo.txt", "hello")
    assert "/workspace/foo.txt" in result


async def test_edit_file_passes_replace_all_flag() -> None:
    """edit_file passes through the replace_all flag."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.aedit.return_value = EditResult(path="/x", occurrences=3)
    tools = build_fs_tools(backend)
    edit_file = next(t for t in tools if t.name == "edit_file")

    result = await edit_file.ainvoke(
        {"path": "/x", "old_string": "a", "new_string": "b", "replace_all": True}
    )

    backend.aedit.assert_awaited_once_with("/x", "a", "b", True)
    assert "3" in result


async def test_glob_returns_matches() -> None:
    """glob renders matched file names."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.aglob.return_value = GlobResult(
        matches=[
            {"path": "/workspace/a.csv", "is_dir": False, "size": 10, "modified_at": "x"},
            {"path": "/workspace/b.csv", "is_dir": False, "size": 20, "modified_at": "x"},
        ]
    )
    tools = build_fs_tools(backend)
    glob_tool = next(t for t in tools if t.name == "glob")

    result = await glob_tool.ainvoke({"pattern": "*.csv"})

    backend.aglob.assert_awaited_once_with("*.csv", "/")
    assert "a.csv" in result
    assert "b.csv" in result


async def test_grep_includes_pattern_path_glob() -> None:
    """grep forwards pattern, path, and glob to the backend."""
    backend = AsyncMock(spec=BackendProtocol)
    backend.agrep.return_value = GrepResult(
        matches=[
            {"path": "/workspace/notes.md", "line": 3, "text": "TODO fix later"},
        ]
    )
    tools = build_fs_tools(backend)
    grep_tool = next(t for t in tools if t.name == "grep")

    result = await grep_tool.ainvoke({"pattern": "TODO", "path": "/workspace", "glob": "*.md"})

    backend.agrep.assert_awaited_once_with("TODO", "/workspace", "*.md")
    assert "notes.md" in result
    assert "TODO" in result


async def test_execute_tool_appears_when_backend_supports_sandbox() -> None:
    """execute is included for sandbox backends."""
    backend = AsyncMock(spec=SandboxBackendProtocol)
    backend.execute.return_value = ExecuteResponse(output="hello\n", exit_code=0, truncated=False)

    tools = build_fs_tools(backend)
    execute = next(t for t in tools if t.name == "execute")

    result = await execute.ainvoke({"command": "echo hello"})

    backend.execute.assert_called_once_with("echo hello")
    assert "hello" in result
    assert "exit_code=0" in result


async def test_execute_tool_omitted_when_backend_is_plain() -> None:
    """execute is omitted for plain filesystem backends."""
    backend = AsyncMock(spec=BackendProtocol)
    tools = build_fs_tools(backend)
    assert all(t.name != "execute" for t in tools)
