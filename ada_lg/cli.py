"""Sibling CLI that drives the pure-LangGraph implementation of ada."""

from __future__ import annotations

from typing import Any, cast

from ada import cli as _ada_cli
from ada_lg.agent import create_analytics_agent

cast(Any, _ada_cli).create_analytics_agent = create_analytics_agent

main = _ada_cli.main
