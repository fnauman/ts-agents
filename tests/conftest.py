"""Shared pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_working_directory(tmp_path, monkeypatch):
    """Run each test from its own temporary directory.

    CLI commands default to relative output roots such as ``outputs/``; running
    from the checkout would leave run directories behind. Repository files are
    resolved from the checkout root, not the working directory.
    """
    monkeypatch.chdir(tmp_path)
