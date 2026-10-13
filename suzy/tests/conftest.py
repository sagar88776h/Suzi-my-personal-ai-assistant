"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tools.registry import REGISTRY, load_all_tools

API_KEY_VARS = ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_API_KEY")


@pytest.fixture(scope="session")
def registry():
    """The global registry with every tool module loaded."""
    load_all_tools()
    return REGISTRY


@pytest.fixture(autouse=True)
def _isolate_api_keys(monkeypatch):
    """Every test starts without API keys, and keys loaded from a .env during a test
    (load_dotenv writes os.environ directly) are removed again afterwards."""
    for var in API_KEY_VARS:
        monkeypatch.setenv(var, "placeholder")  # records the original value for restoring
        monkeypatch.delenv(var)
