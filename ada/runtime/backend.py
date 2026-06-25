"""Backend auto-detection for Modal vs local sandboxing."""

from __future__ import annotations

import os


def has_modal_credentials() -> bool:
    """Return True if both Modal token env vars are set."""
    return bool(os.environ.get("MODAL_TOKEN_ID")) and bool(os.environ.get("MODAL_TOKEN_SECRET"))
