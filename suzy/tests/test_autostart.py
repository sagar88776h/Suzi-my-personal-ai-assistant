"""Tests for starting with Windows and the "welcome back" greeting."""

from __future__ import annotations

import subprocess

import pytest

import guard
import main
from core.config import Config
from tools import autostart, winutils


# ----------------------------------------------------------------- greeting
def test_startup_greeting():
    assert main.greeting(Config(), startup=True) == "Hello Sagar, welcome back."
    custom = Config({"user": {"name": "Sagar"}, "assistant": {"startup_greeting": "Namaste {name}!"}})
    assert main.greeting(custom, startup=True) == "Namaste Sagar!"
    broken = Config({"assistant": {"startup_greeting": "Hi {nme}"}})
    assert main.greeting(broken, startup=True) == "Hi {nme}"  # bad template: said as written


def test_wait_for_internet(monkeypatch):
    attempts = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def connect(address, timeout):
        attempts.append(address)
        if len(attempts) < 3:
            raise OSError("network not ready")
        return Connection()

    monkeypatch.setattr(main.socket, "create_connection", connect)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    assert main.wait_for_internet(timeout=30) is True and len(attempts) == 3

    monkeypatch.setattr(main.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert main.wait_for_internet(timeout=0) is False


def test_greet_after_startup_speaks_even_offline(monkeypatch):
    said = []
    monkeypatch.setattr(main, "wait_for_internet", lambda timeout: False)
    session = type("S", (), {"say": lambda self, text: said.append(text)})()
    main.greet_after_startup(session, "Hello Sagar, welcome back.").join(2)
    assert said == ["Hello Sagar, welcome back."]


def test_main_startup_greets_first(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")
    greeted = []
    monkeypatch.setattr(main, "greet_after_startup", lambda session, text: greeted.append(text))
    monkeypatch.setattr(main, "setup_self_improvement", lambda *a, **k: None)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "exit")
    assert main.main(["--startup", "--mute", "--no-voice"]) == 0
    out = capsys.readouterr().out
    assert greeted == ["Hello Sagar, welcome back."]
    assert out.index("Hello Sagar, welcome back.") < out.index("Bye for now")
    assert "Good morning" not in out and "Good afternoon" not in out and "Good evening" not in out


# ----------------------------------------------------------------- autostart tool
@pytest.fixture()
def windows(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData"
    app_dir = tmp_path / "Suzy"
    app_dir.mkdir()
    (app_dir / "Start Suzy.bat").write_text("@echo off", encoding="ascii")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(winutils, "IS_WINDOWS", True)
    scripts = []

    def fake_powershell(script, timeout=15):
        scripts.append(script)
        autostart.shortcut_path().write_text("lnk")  # what WScript.Shell would create
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(winutils, "run_powershell", fake_powershell)
    monkeypatch.setattr(autostart, "APP_DIR", app_dir)
    return scripts, app_dir


def test_set_autostart_on_and_off(windows):
    scripts, app_dir = windows
    autostart.enable(app_dir)
    assert autostart.is_enabled()
    script = scripts[0]
    assert "$s.Arguments = '--startup'" in script and "$s.WindowStyle = 7" in script
    assert str(app_dir / "Start Suzy.bat") in script
    result = autostart.set_autostart(False)
    assert result.success and not autostart.is_enabled()
    autostart.disable()  # already off: no error


def test_set_autostart_tool(windows):
    scripts, _app_dir = windows
    result = autostart.set_autostart(True)
    assert result.success and "start with Windows" in result.message and autostart.is_enabled()
    assert len(scripts) == 1


def test_set_autostart_failure(windows, monkeypatch):
    scripts, app_dir = windows
    (app_dir / "Start Suzy.bat").unlink()
    result = autostart.set_autostart(True)
    assert not result.success and "re-run the installer" in result.message


def test_set_autostart_off_windows(registry, monkeypatch):
    monkeypatch.setattr(winutils, "IS_WINDOWS", False)
    result = registry.execute("set_autostart", {"enabled": True})
    assert not result.success and "only works on Windows" in result.message


# ----------------------------------------------------------------- guard
def test_guard_says_welcome_back_only_once(monkeypatch, tmp_path):
    seen = []
    codes = iter([3, 0])

    def call(command, **kwargs):
        seen.append(command[2:])
        return next(codes)

    monkeypatch.setattr(guard, "MARKER", tmp_path / "none.json")
    monkeypatch.setattr(guard.subprocess, "call", call)
    assert guard.run(["--startup"]) == 0
    assert seen == [["--startup"], []]
