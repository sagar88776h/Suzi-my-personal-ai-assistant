"""Tests for configuration loading and API key handling."""

from __future__ import annotations

import pytest

from core.config import DEFAULT_CONFIG, Config, deep_merge


def test_defaults():
    config = Config()
    assert config.get("llm.provider") == "groq"
    assert config.get("llm.groq.model") == "llama-3.3-70b-versatile"
    assert config.get("missing.key", "fallback") == "fallback"
    assert config.get("llm.provider.deeper") is None


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}, "d": 4})
    assert merged == {"a": {"b": 9, "c": 2}, "d": 4} and base == {"a": {"b": 1, "c": 2}}


def test_load_yaml_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("user:\n  name: Sagar K\nllm:\n  provider: gemini\n", encoding="utf-8")
    config = Config.load(path, tmp_path / ".env")
    assert config.get("user.name") == "Sagar K"
    assert config.get("llm.provider") == "gemini"
    assert config.get("llm.fallback_provider") == "gemini"  # default kept
    assert config.load_errors == []


@pytest.mark.parametrize("content", ["llm: [unclosed", "- just\n- a list\n"])
def test_broken_yaml_falls_back_to_defaults(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    config = Config.load(path, tmp_path / ".env")
    assert config.get("llm.provider") == DEFAULT_CONFIG["llm"]["provider"]
    assert config.load_errors


def test_missing_yaml(tmp_path):
    config = Config.load(tmp_path / "nope.yaml", tmp_path / ".env")
    assert config.get("assistant.name") == "Suzy" and "not found" in config.load_errors[0]


def test_set_and_save_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    config = Config({}, path)
    config.set("tts.voice", "en-US-AriaNeural")
    config.set("llm.provider", "gemini")
    config.save()
    reloaded = Config.load(path, tmp_path / ".env")
    assert reloaded.get("tts.voice") == "en-US-AriaNeural" and reloaded.get("llm.provider") == "gemini"


def test_save_without_path():
    with pytest.raises(ValueError):
        Config().save()


def test_get_returns_copies():
    config = Config()
    config.get("tools.apps.extra_search_dirs").append("X")
    assert config.get("tools.apps.extra_search_dirs") == []


def test_api_keys_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY="gsk_abc"\nGEMINI_API_KEY=\n', encoding="utf-8")
    Config.load(tmp_path / "config.yaml", env)
    assert Config.api_key("groq") == "gsk_abc"
    assert Config.api_key("gemini") is None
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_alt")
    assert Config.api_key("gemini") == "AIza_alt"
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_key_here")
    assert Config.api_key("groq") is None
    assert Config.api_key("unknown") is None
