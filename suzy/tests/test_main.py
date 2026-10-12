"""Tests for the terminal entry point helpers."""

from __future__ import annotations

from datetime import datetime

import main
from core.config import Config
from core.llm import GeminiProvider, GroqProvider, LLMRouter


def test_greeting_by_time_of_day():
    config = Config()
    assert main.greeting(config, datetime(2026, 1, 1, 8)).startswith("Good morning, Sagar.")
    assert main.greeting(config, datetime(2026, 1, 1, 14)).startswith("Good afternoon")
    assert main.greeting(config, datetime(2026, 1, 1, 22)).startswith("Good evening")


def test_startup_report():
    config = Config()
    none = LLMRouter([GroqProvider(None, "m"), GeminiProvider(None, "m")])
    assert any("No API key found" in p for p in main.startup_report(config, none))

    groq_only = LLMRouter([GroqProvider("k", "m"), GeminiProvider(None, "m")])
    assert main.startup_report(config, groq_only) == ["No Gemini API key, so I have no Gemini backup brain."]

    both = LLMRouter([GroqProvider("k", "m"), GeminiProvider("k", "m")])
    assert main.startup_report(config, both) == []


def test_once_mode_without_keys(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")
    assert main.main(["--once", "hello"]) == 0
    out = capsys.readouterr().out
    assert "Heads up" in out and "API key" in out


def test_terminal_chat_commands(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")
    inputs = iter(["", "/help", "/tools", "/reset", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    assert main.main([]) == 0
    out = capsys.readouterr().out
    assert "/reset" in out and "open_app" in out and "Fresh start" in out and "Bye for now" in out


def test_terminal_chat_handles_ctrl_c(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")

    def interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert main.main([]) == 0
    assert "Bye for now" in capsys.readouterr().out
