"""Configuration loading for Suzy.

Settings live in two places:

* ``config.yaml`` - every user-tunable setting (voice, LLM provider, ...).
* ``.env``        - secrets such as API keys, never committed to git.

Missing or broken config files never stop Suzy from starting: any value that
cannot be read falls back to the built-in defaults below.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from core.paths import CONFIG_PATH, ENV_PATH

log = logging.getLogger("suzy.config")

# Built-in defaults. config.yaml only needs to contain the values you change.
DEFAULT_CONFIG: dict[str, Any] = {
    "user": {
        "name": "Sagar",
    },
    "assistant": {
        "name": "Suzy",
        "max_context_turns": 20,   # conversation turns sent to the LLM
        "max_tool_rounds": 5,      # tool-call rounds allowed per request
    },
    "llm": {
        "provider": "groq",
        "fallback_provider": "gemini",
        "temperature": 0.6,
        "max_tokens": 1024,
        "request_timeout": 20,     # seconds
        "groq": {
            "model": "llama-3.3-70b-versatile",
        },
        "gemini": {
            "model": "gemini-2.5-flash",
            # 0 disables "thinking" on Gemini 2.5 Flash for snappier replies.
            # Set to null for models that do not support a thinking budget.
            "thinking_budget": 0,
        },
    },
    "tools": {
        "apps": {
            "fuzzy_threshold": 80,     # score needed to open an app directly
            "suggest_threshold": 60,   # score needed to suggest "did you mean"
            "extra_search_dirs": [],   # more folders to scan for .exe/.lnk
        },
    },
    "logging": {
        "level": "DEBUG",
        "max_bytes": 2_000_000,
        "backup_count": 5,
    },
}

# Environment variables that hold each provider's API key, in priority order.
API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is recursively merged onto ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Thread-safe access to Suzy's settings using dotted keys (``"llm.provider"``)."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self._data = deep_merge(DEFAULT_CONFIG, data or {})
        self._path = path
        self._lock = threading.RLock()
        self.load_errors: list[str] = []

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path = CONFIG_PATH, env_path: Path = ENV_PATH) -> Config:
        """Load config.yaml and .env, falling back to defaults on any problem."""
        errors: list[str] = []
        if env_path.exists():
            load_dotenv(env_path, override=False)

        user_data: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    user_data = loaded
                elif loaded is not None:
                    errors.append(f"{path.name} is not a mapping; using defaults.")
            except (OSError, yaml.YAMLError) as exc:
                errors.append(f"Could not read {path.name} ({exc}); using defaults.")
        else:
            errors.append(f"{path.name} not found; using defaults.")

        config = cls(user_data, path)
        config.load_errors = errors
        for message in errors:
            log.warning(message)
        return config

    # ------------------------------------------------------------------ access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a value such as ``config.get("llm.groq.model")``."""
        with self._lock:
            node: Any = self._data
            for part in dotted_key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a value in memory (call :meth:`save` to persist it)."""
        with self._lock:
            parts = dotted_key.split(".")
            node = self._data
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[parts[-1]] = copy.deepcopy(value)

    def save(self) -> None:
        """Write the current settings back to config.yaml."""
        if self._path is None:
            raise ValueError("This Config has no file path to save to.")
        with self._lock:
            text = yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True)
        self._path.write_text(text, encoding="utf-8")

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    # ------------------------------------------------------------------ secrets
    @staticmethod
    def api_key(provider: str) -> str | None:
        """Return the API key for ``provider`` from the environment, or None.

        Empty values and the placeholder text from .env.example count as missing.
        """
        for var in API_KEY_ENV_VARS.get(provider, ()):
            value = os.environ.get(var, "").strip().strip('"').strip("'")
            if value and not value.lower().startswith("your"):
                return value
        return None
