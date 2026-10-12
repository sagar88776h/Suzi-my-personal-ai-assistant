"""Tests for the global hotkey (pynput mocked)."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from core.hotkey import GlobalHotkey, to_pynput


@pytest.mark.parametrize(
    "combo, expected",
    [
        ("ctrl+alt+s", "<ctrl>+<alt>+s"),
        ("Ctrl + Alt + S", "<ctrl>+<alt>+s"),
        ("control+shift+F9", "<ctrl>+<shift>+<f9>"),
        ("win+space", "<cmd>+<space>"),
    ],
)
def test_to_pynput(combo, expected):
    assert to_pynput(combo) == expected


@pytest.mark.parametrize("combo", ["", "ctrl+banana", "+"])
def test_invalid_combos(combo):
    with pytest.raises(ValueError):
        to_pynput(combo)


class FakeHotKeys:
    last: FakeHotKeys | None = None

    def __init__(self, mapping):
        self.mapping, self.started, self.stopped = mapping, False, False
        FakeHotKeys.last = self

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture()
def fake_pynput(monkeypatch):
    keyboard = types.SimpleNamespace(GlobalHotKeys=FakeHotKeys)
    package = types.ModuleType("pynput")
    package.keyboard = keyboard
    monkeypatch.setitem(sys.modules, "pynput", package)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    return keyboard


def test_hotkey_fires_callback_on_worker_thread(fake_pynput):
    fired = threading.Event()
    threads = []

    def callback():
        threads.append(threading.current_thread().name)
        fired.set()

    hotkey = GlobalHotkey("ctrl+alt+s", callback)
    assert hotkey.start() and FakeHotKeys.last.started
    FakeHotKeys.last.mapping["<ctrl>+<alt>+s"]()
    assert fired.wait(2) and threads == ["HotkeyAction"]
    hotkey.stop()
    assert FakeHotKeys.last.stopped


def test_debounce(fake_pynput):
    count = []
    hotkey = GlobalHotkey("ctrl+alt+s", lambda: count.append(1))
    hotkey.start()
    trigger = FakeHotKeys.last.mapping["<ctrl>+<alt>+s"]
    trigger()
    trigger()  # immediate repeat (key bounce) is ignored
    hotkey.DEBOUNCE_SECONDS = 0
    trigger()
    threading.Event().wait(0.3)
    assert len(count) == 2


def test_callback_errors_are_contained(fake_pynput):
    def boom():
        raise RuntimeError("bug")

    hotkey = GlobalHotkey("ctrl+alt+s", boom)
    hotkey.start()
    FakeHotKeys.last.mapping["<ctrl>+<alt>+s"]()  # must not raise


def test_bad_combo_reports_error(fake_pynput):
    hotkey = GlobalHotkey("ctrl+banana", lambda: None)
    assert hotkey.start() is False and "isn't valid" in hotkey.error


def test_listener_failure_reports_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    hotkey = GlobalHotkey("ctrl+alt+s", lambda: None)
    assert hotkey.start() is False and "couldn't register" in hotkey.error
    hotkey.stop()  # safe even though it never started
