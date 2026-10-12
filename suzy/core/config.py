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
import re
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
        "quick_replies": True,     # say simple action results directly (skips a 2nd AI call)
    },
    "llm": {
        "provider": "groq",
        "fallback_provider": "gemini",
        "temperature": 0.6,
        "max_tokens": 2048,
        "request_timeout": 20,     # seconds
        # Models are tried in order; if one is retired, the next one is used.
        "groq": {
            "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
            "reasoning_effort": "low",   # gpt-oss only: low | medium | high
        },
        "gemini": {
            "models": ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash"],
            "thinking_budget": None,     # None = model default; 0 disables thinking on 2.5 models
        },
    },
    "voice": {
        "enabled": True,             # microphone + push-to-talk
        "hotkey": "ctrl+alt+s",      # works from any window
        "input_device": None,        # None = Windows default mic; or part of a mic's name
        "energy_threshold": 0.006,   # how loud counts as speech; lower = more sensitive
        "max_record_seconds": 15,
        "silence_seconds": 1.2,      # pause that ends what you're saying
        "no_speech_timeout": 6,      # stop listening if you say nothing
        "sound_cues": True,          # chime when listening starts/stops
        "wake_word": {
            "enabled": True,         # always listen for "Suzy" / "Hi Suzy" / "Hello Suzy"
            "model": "tiny",         # Whisper model that checks for her name (tiny = fastest)
            "followup_seconds": 8,   # after answering, the next request needs no wake word (0 = off)
            "command_timeout": 6,    # after "Suzy", how long to wait for the request
            "vad_threshold": 0.5,    # 0-1: higher = needs clearer speech (fewer false triggers)
            "end_silence": 0.7,      # pause that ends a sentence
        },
    },
    "tts": {
        "enabled": True,
        "voice": "en-IN-NeerjaExpressiveNeural",  # natural, expressive Indian female voice
        "fallback_voice": "en-IN-NeerjaNeural",
        "hindi_voice": "hi-IN-SwaraNeural",  # speaks replies containing Hindi (Devanagari) words
        "rate": "+0%",               # speed, e.g. "+15%" faster, "-10%" slower
        "pitch": "+0Hz",
        "volume": "+0%",
        "offline_fallback": True,    # Windows built-in voice when there's no internet
    },
    "stt": {
        "engine": "groq",            # groq = fast cloud Whisper (free, same key); local = on this PC only
        "cloud_model": "whisper-large-v3-turbo",
        "model": "small",            # local model, also the backup when offline            # tiny | base | small | medium (bigger = slower, more accurate)
        "device": "cpu",
        "compute_type": "int8",
        "language": None,            # None = auto (English/Hindi); "en" to force English
        "allowed_languages": ["en", "hi"],
        "beam_size": 1,              # 1 = fastest; 5 = a little more accurate
        "models_dir": "models",
    },
    "self_improvement": {
        "enabled": True,             # Suzy may change her own code (always asks before installing)
        "auto_fix_errors": True,     # start fixing new errors from the log on her own (still asks to install)
        "max_steps": 40,             # coding steps per change
        "context_tokens": 4500,      # per-request size; fits Groq's free 8,000 tokens/minute
        "max_output_tokens": 3000,
        "keep_backups": 10,
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


_YAML_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*)$")


def replace_yaml_value(text: str, keys: list[str], value: Any) -> str | None:
    """Replace the value of ``a.b.c`` in YAML text, keeping indentation and the line's comment.

    Returns None if the key isn't in the text (or holds a nested block).
    """
    stack: list[tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _YAML_LINE.match(line.rstrip("\r\n"))
        if not match:
            continue
        indent = len(match["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, match["key"]))
        if [key for _indent, key in stack] != keys:
            continue
        rest = match["rest"]
        comment_at = re.search(r"\s+#", rest)
        old_value = rest[:comment_at.start()] if comment_at else rest
        if not old_value.strip():
            return None  # a nested section, not a single value
        comment = rest[comment_at.start():] if comment_at else ""
        dumped = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
        dumped = dumped.removesuffix("\n...").removesuffix("...").strip()
        new_value = f" {dumped}"
        if comment:  # keep the comment roughly where it was
            new_value = new_value.ljust(len(old_value))
        ending = "\n" if line.endswith("\n") else ""
        lines[index] = f"{match['indent']}{match['key']}:{new_value}{comment}{ending}"
        return "".join(lines)
    return None


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

    def persist(self, dotted_key: str, value: Any) -> None:
        """Set a value and write just that line of config.yaml, keeping its comments.

        Falls back to rewriting the whole file if the key isn't in it yet.
        """
        self.set(dotted_key, value)
        if self._path is None:
            return
        with self._lock:
            try:
                text = self._path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            updated = replace_yaml_value(text, dotted_key.split("."), value)
            if updated is None:
                self.save()
            else:
                self._path.write_text(updated, encoding="utf-8")

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
