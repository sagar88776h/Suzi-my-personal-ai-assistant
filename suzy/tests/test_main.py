"""Tests for the terminal entry point helpers."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

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
    assert main.main(["--once", "hello", "--mute"]) == 0
    out = capsys.readouterr().out
    assert "Heads up" in out and "API key" in out


def test_terminal_chat_commands(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")
    inputs = iter(["", "/help", "/tools", "/mute", "/mute", "/reset", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    assert main.main(["--mute", "--no-voice"]) == 0
    out = capsys.readouterr().out
    assert "/reset" in out and "open_app" in out and "Fresh start" in out and "Bye for now" in out
    assert "Voice on." in out and "Voice off." in out


def test_terminal_chat_handles_ctrl_c(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")

    def interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert main.main(["--mute", "--no-voice"]) == 0
    assert "Bye for now" in capsys.readouterr().out


class StubRouter:
    """Stands in for LLMRouter in startup tests."""

    def __init__(self, statuses):
        self._statuses = statuses
        self.providers = [GroqProvider("k", "m")]

    @property
    def configured_providers(self):
        return ["groq"]

    def check(self):
        return self._statuses


def test_brain_check_output(capsys):
    from core.llm import LLMError, LLMErrorKind, ProviderStatus

    bad = ProviderStatus("groq", False, "m", LLMError("groq", LLMErrorKind.MODEL_UNAVAILABLE, "404 gone"))
    main.run_brain_check(StubRouter([bad]))
    out = capsys.readouterr().out
    assert "Groq   : NOT working - none of the configured models are available" in out
    assert "groq: 404 gone" in out

    main.run_brain_check(StubRouter([ProviderStatus("groq", True, "openai/gpt-oss-120b")]))
    out = capsys.readouterr().out
    assert "working (openai/gpt-oss-120b)" in out and "Technical details" not in out


def test_check_flag_exit_code(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")
    assert main.main(["--check"]) == 1


def test_print_reply_shows_detail(capsys):
    main.print_reply(main.AssistantReply("Oops.", error=True, detail="groq: auth - 401"), show_actions=True)
    assert "(details: groq: auth - 401)" in capsys.readouterr().out
    main.print_reply(main.AssistantReply("Oops.", error=True, detail="x"), show_actions=False)
    assert "details" not in capsys.readouterr().out


def test_terminal_ui_redraws_prompt_after_background_event(monkeypatch, capsys):
    ui = main.TerminalUI()
    ui._at_prompt = True  # main thread is waiting in input()
    ui.on_event("listening", None)
    ui.on_event("heard", "open notepad")
    ui.on_event("reply", main.AssistantReply("Opening Notepad."))
    ui.on_event("idle", None)
    out = capsys.readouterr().out
    assert out.startswith("\nSuzy: Listening")
    assert "You (voice): open notepad" in out and "Suzy: Opening Notepad." in out
    assert out.endswith("You: ")


def test_build_voice_without_devices(monkeypatch):
    """No sound card at all: Suzy still starts, explains, and runs typing-only."""
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    config = Config()
    assistant, _router, _registry = main.build_assistant(config)
    voice = main.build_voice(config, assistant, main.TerminalUI())
    assert not voice.session.can_listen and voice.hotkey is None
    assert any("audio system" in p for p in voice.problems)


def test_build_voice_with_fake_devices(monkeypatch, tmp_path):
    import sys
    import types

    from tests.fake_sounddevice import install_fake_sounddevice
    from tests.test_hotkey import FakeHotKeys

    install_fake_sounddevice(monkeypatch)
    keyboard = types.SimpleNamespace(GlobalHotKeys=FakeHotKeys)
    package = types.ModuleType("pynput")
    package.keyboard = keyboard
    monkeypatch.setitem(sys.modules, "pynput", package)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    monkeypatch.setattr("core.stt.SpeechToText.load_async", lambda self: None)
    config = Config({"stt": {"models_dir": str(tmp_path)}})
    assistant, _router, _registry = main.build_assistant(config)
    ui = main.TerminalUI()
    voice = main.build_voice(config, assistant, ui)
    assert voice.session.can_listen and voice.hotkey is not None and ui.hotkey_label == "Ctrl+Alt+S"
    assert any("downloading my speech model" in n for n in voice.notes)
    assert any("Fake Mic" in n for n in voice.notes) and voice.problems == []


def _mic_test(monkeypatch, capsys, loudness, config=None):
    import numpy as np

    from tests.fake_sounddevice import install_fake_sounddevice, speech_signal

    # loudness 0 = exact digital silence, which is what a privacy-blocked Windows mic delivers
    signal = np.zeros(16_000, np.float32) if loudness == 0 else speech_signal(0.1, 1.0, 0.0, loudness=loudness)
    install_fake_sounddevice(monkeypatch, mic_signal=signal, realtime=0.005)
    config = config or Config()
    if config.get("stt.models_dir") == "models":
        # Never load the real speech model in tests (it exists on a PC where Suzy already ran).
        config.set("stt.models_dir", str(Path(tempfile.mkdtemp()) / "no-models"))
    code = main.run_mic_test(config, seconds=0.3)
    return code, capsys.readouterr().out


def test_mic_test_good(monkeypatch, capsys):
    code, out = _mic_test(monkeypatch, capsys, loudness=0.2)
    assert code == 0 and "[0] Fake Mic   <- Windows default" in out and "sounds good" in out


def test_mic_test_silent(monkeypatch, capsys):
    code, out = _mic_test(monkeypatch, capsys, loudness=0.0)
    assert code == 1 and "pure silence" in out and "Privacy" in out


def test_mic_test_quiet(monkeypatch, capsys):
    code, out = _mic_test(monkeypatch, capsys, loudness=0.005)
    assert code == 1 and "very quiet" in out and "voice.energy_threshold:" in out


def test_mic_test_chosen_device(monkeypatch, capsys):
    code, out = _mic_test(monkeypatch, capsys, 0.2, Config({"voice": {"input_device": "headset"}}))
    assert "Testing: USB Headset Microphone" in out


def test_mic_test_flag(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert main.main(["--mic-test"]) == 1
    assert "audio system" in capsys.readouterr().out


def test_build_voice_bad_input_device(monkeypatch, tmp_path):
    from tests.fake_sounddevice import install_fake_sounddevice

    install_fake_sounddevice(monkeypatch)
    monkeypatch.setattr("core.stt.SpeechToText.load_async", lambda self: None)
    monkeypatch.setitem(__import__("sys").modules, "pynput", None)
    config = Config({"voice": {"input_device": "Bluetooth"}, "stt": {"models_dir": str(tmp_path)}})
    assistant, _router, _registry = main.build_assistant(config)
    voice = main.build_voice(config, assistant, main.TerminalUI())
    assert any("Bluetooth" in p and "default microphone" in p for p in voice.problems)
    assert voice.session.can_listen and voice.session.recorder.device is None


def test_mic_test_runs_speech_recognition(monkeypatch, capsys, tmp_path):
    from core.stt import Transcript

    (tmp_path / "faster-whisper-small").mkdir()
    (tmp_path / "faster-whisper-small" / "model.bin").write_bytes(b"x")
    monkeypatch.setattr("core.stt.SpeechToText.transcribe",
                        lambda self, audio: Transcript("Hey Suzy, open Notepad.", "en", 0.9, 0.8))
    code, out = _mic_test(monkeypatch, capsys, 0.2, Config({"stt": {"models_dir": str(tmp_path)}}))
    assert code == 0 and 'I heard: "Hey Suzy, open Notepad."' in out


def test_mic_test_without_model(monkeypatch, capsys, tmp_path):
    code, out = _mic_test(monkeypatch, capsys, 0.2, Config({"stt": {"models_dir": str(tmp_path)}}))
    assert code == 0 and "not downloaded yet" in out
