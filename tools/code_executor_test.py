"""Tests for the persistent IPython kernel session."""

import pytest

from tools.code_executor import KernelSession, make_execute_python_tool


@pytest.fixture
def session(tmp_path):
    s = KernelSession(work_dir=str(tmp_path), timeout=10)
    yield s
    s.shutdown()


def test_execute_returns_stdout(session):
    """print() output is captured and returned as a string."""
    result = session.execute("print('hi')")
    assert "hi" in result


def test_state_persists(session):
    """Variables defined in one call survive into the next."""
    session.execute("x = 41")
    result = session.execute("print(x + 1)")
    assert "42" in result


def test_traceback_returned_as_string(session):
    """Errors don't raise out of execute(); the traceback is returned."""
    result = session.execute("1/0")
    assert "ZeroDivisionError" in result


def test_output_truncated(session):
    """Huge output is capped with a clear suffix."""
    result = session.execute("print('a' * 100_000)")
    assert "[truncated" in result
    assert len(result) < 20_000


def test_timeout_interrupts(tmp_path):
    """Long-running code is interrupted and returns a timeout string."""
    s = KernelSession(work_dir=str(tmp_path), timeout=1)
    try:
        result = s.execute("while True: pass")
        assert "TimeoutError" in result
    finally:
        s.shutdown()


def test_kernel_recovers_after_timeout(tmp_path):
    """After a timeout, the next execute call still works."""
    s = KernelSession(work_dir=str(tmp_path), timeout=1)
    try:
        s.execute("while True: pass")
        result = s.execute("print('alive')")
        assert "alive" in result
    finally:
        s.shutdown()


def test_cwd_is_work_dir(session, tmp_path):
    """Files written via relative paths land in work_dir."""
    session.execute("open('marker.txt', 'w').write('here')")
    assert (tmp_path / "marker.txt").read_text() == "here"


def test_kernel_dead_returns_string(tmp_path):
    """If the kernel process is gone, execute returns a clear message."""
    s = KernelSession(work_dir=str(tmp_path), timeout=5)
    s._km.shutdown_kernel(now=True)  # simulate crash
    result = s.execute("print('hi')")
    assert "KernelDeadError" in result
    s.shutdown()


def test_execute_python_tool_runs_code(session):
    """The LangChain tool wrapper invokes the underlying session."""
    tool = make_execute_python_tool(session)
    result = tool.invoke({"code": "print('via tool')"})
    assert "via tool" in result
