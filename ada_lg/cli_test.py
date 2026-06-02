"""Tests that the cli shim rebinds the agent factory."""

from __future__ import annotations

import ada_lg.cli
from ada import cli as ada_cli


def test_rebind_swaps_factory_symbol() -> None:
    """Importing ada_lg.cli swaps ada.cli's factory symbol."""
    assert ada_cli.create_analytics_agent is ada_lg.cli.create_analytics_agent
    assert ada_cli.create_analytics_agent.__module__.startswith("ada_lg")


def test_main_is_exported() -> None:
    """The sibling CLI exports the shared main entrypoint."""
    assert callable(ada_lg.cli.main)
