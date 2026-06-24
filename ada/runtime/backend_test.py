"""Tests for backend detection logic."""

import os
from unittest.mock import patch

from ada.runtime.backend import has_modal_credentials


def test_has_modal_credentials_returns_true_when_both_set() -> None:
    with patch.dict(
        os.environ,
        {"MODAL_TOKEN_ID": "id-123", "MODAL_TOKEN_SECRET": "sec-456"},
        clear=True,
    ):
        assert has_modal_credentials() is True


def test_has_modal_credentials_returns_false_when_id_missing() -> None:
    with patch.dict(os.environ, {"MODAL_TOKEN_SECRET": "sec-456"}, clear=True):
        assert has_modal_credentials() is False


def test_has_modal_credentials_returns_false_when_secret_missing() -> None:
    with patch.dict(os.environ, {"MODAL_TOKEN_ID": "id-123"}, clear=True):
        assert has_modal_credentials() is False


def test_has_modal_credentials_returns_false_when_both_missing() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert has_modal_credentials() is False
