"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tools.registry import REGISTRY, load_all_tools


@pytest.fixture(scope="session")
def registry():
    """The global registry with every tool module loaded."""
    load_all_tools()
    return REGISTRY
