"""Persistent IPython kernel for the code-execution agent variant.

Wraps ``jupyter_client.KernelManager`` to give a long-lived Python REPL with
state that persists across calls. Exposes a single ``execute_python`` tool.
"""

from __future__ import annotations

import queue
import re

from jupyter_client.manager import KernelManager
from langchain_core.tools import tool as _tool_decorator

_MAX_OUTPUT_CHARS = 10_000

# Some model providers (e.g. Google Gemini/Gemma) strictly validate the JSON
# payload of the request and reject the non-standard tokens ``NaN`` /
# ``Infinity`` that pandas and numpy emit by default. Lower-casing them turns
# them into ordinary identifier-looking strings that survive JSON encoding.
_NON_JSON_FLOAT_RE = re.compile(r"\b(NaN|-?Infinity)\b")


def _sanitize_for_json(text: str) -> str:
    """Replace ``NaN``/``Infinity`` tokens with JSON-safe lowercase variants."""
    return _NON_JSON_FLOAT_RE.sub(lambda m: m.group(0).lower(), text)


class KernelSession:
    """A long-lived IPython kernel rooted at ``work_dir``.

    State (variables, imports, DataFrames) persists across ``execute`` calls.
    The kernel's working directory is ``work_dir`` so relative paths like
    ``'dataset.csv'`` resolve to the per-run sandbox.

    Args:
        work_dir: Directory the kernel runs in (its CWD).
        timeout: Per-cell wall-clock timeout in seconds.
    """

    def __init__(self, work_dir: str, timeout: int = 60):
        self._work_dir = work_dir
        self._timeout = timeout
        self._km = KernelManager()
        self._km.start_kernel(cwd=work_dir)
        self._client = self._km.client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=30)

    def execute(self, code: str) -> str:
        """Execute ``code`` in the kernel and return collected output.

        Returns a single string containing stdout, the final result repr,
        and any traceback. Never raises — every failure mode is a string
        the agent can read.
        """
        if not self._km.is_alive():
            return "KernelDeadError: restart required"
        msg_id = self._client.execute(code)
        outputs: list[str] = []
        while True:
            try:
                msg = self._client.get_iopub_msg(timeout=self._timeout)
            except queue.Empty:
                self._km.interrupt_kernel()
                try:
                    self._client.get_shell_msg(timeout=self._timeout)
                except queue.Empty:
                    pass
                return f"TimeoutError: cell exceeded {self._timeout}s"
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            msg_type = msg["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                outputs.append(content["text"])
            elif msg_type == "execute_result":
                outputs.append(content["data"].get("text/plain", ""))
            elif msg_type == "error":
                outputs.append("\n".join(content["traceback"]))
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
        try:
            self._client.get_shell_msg(timeout=self._timeout)
        except queue.Empty:
            pass
        text = _sanitize_for_json("".join(outputs))
        if len(text) > _MAX_OUTPUT_CHARS:
            extra = len(text) - _MAX_OUTPUT_CHARS
            text = text[:_MAX_OUTPUT_CHARS] + f"\n... [truncated, {extra} more chars]"
        return text

    def shutdown(self) -> None:
        """Stop channels and shut down the kernel."""
        try:
            self._client.stop_channels()
        finally:
            self._km.shutdown_kernel(now=True)


def make_execute_python_tool(session: KernelSession):
    """Return an ``execute_python`` LangChain tool bound to ``session``."""

    @_tool_decorator
    def execute_python(code: str) -> str:
        """Execute Python in a persistent IPython kernel.

        State (imports, variables, DataFrames) persists across calls.
        The kernel's CWD is the per-run work directory; use relative paths
        like 'dataset.csv', 'profile.json', 'plots/foo.png'.

        Errors are returned as strings (full traceback). Output is truncated
        at ~10 KB. Long-running cells (>60s by default) are interrupted.

        Args:
            code: Python source to execute. May span multiple statements.

        Returns:
            Combined stdout, the final expression repr, and any traceback.
        """
        return session.execute(code)

    return execute_python
