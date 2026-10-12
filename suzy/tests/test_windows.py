"""Tests for window / tab / keyboard control on a fake Windows desktop."""

from __future__ import annotations

import pytest

from tools import context, win32, windows, winutils
from tools.win32 import ForegroundTracker, Window, is_suzy_window, parse_keys

SUZY = Window(1, "Suzy", 100, "windowsterminal.exe")
CHROME = Window(2, "YouTube - Google Chrome", 200, "chrome.exe")
CHROME2 = Window(3, "Gmail - Google Chrome", 200, "chrome.exe")
NOTEPAD = Window(4, "*notes.txt - Notepad", 300, "notepad.exe")
EDGE = Window(5, "News - Microsoft Edge", 400, "msedge.exe")
WORD = Window(6, "invoice.docx - Word", 500, "winword.exe")


class FakeDesktop:
    def __init__(self, windows, foreground, unsaved=()):
        self.open = {w.hwnd: w for w in windows}
        self.order = [w.hwnd for w in windows]
        self.fg = foreground.hwnd if foreground else None
        self.unsaved = set(unsaved)  # these windows show "Save changes?" instead of closing
        self.closed, self.activated, self.pressed, self.typed = [], [], [], []
        self.minimized, self.maximized, self.desktop_shown = [], [], 0

    def windows(self):
        return [self.open[h] for h in self.order if h in self.open]

    def foreground(self):
        return self.open.get(self.fg)

    def exists(self, hwnd):
        return hwnd in self.open

    def close(self, hwnd):
        self.closed.append(hwnd)
        if hwnd not in self.unsaved:
            self.open.pop(hwnd, None)

    def activate(self, hwnd):
        self.activated.append(hwnd)
        self.fg = hwnd
        return True

    def minimize(self, hwnd):
        self.minimized.append(hwnd)

    def maximize(self, hwnd):
        self.maximized.append(hwnd)

    def press(self, codes):
        self.pressed.append((self.fg, codes))

    def type_text(self, text):
        self.typed.append((self.fg, text))

    def show_desktop(self):
        self.desktop_shown += 1


@pytest.fixture()
def desk(monkeypatch):
    def make(windows_list=(SUZY, CHROME, CHROME2, NOTEPAD, EDGE, WORD), foreground=SUZY, last=NOTEPAD,
             unsaved=()):
        fake = FakeDesktop(list(windows_list), foreground, unsaved)
        tracker = ForegroundTracker(desktop=fake)
        tracker.last = last
        monkeypatch.setattr(win32, "DESKTOP", fake)
        monkeypatch.setattr(win32, "TRACKER", tracker)
        monkeypatch.setattr(windows, "_wait_closed",
                            lambda ws, seconds=3.0: [w for w in ws if fake.exists(w.hwnd)])
        context.set_user_message("")
        return fake
    return make


# ----------------------------------------------------------------- helpers
def test_parse_keys():
    assert parse_keys("ctrl+shift+t") == [0x11, 0x10, ord("T")]
    assert parse_keys("Alt + F4") == [0x12, 0x73]
    assert parse_keys("ctrl+page up") == [0x11, 0x21]
    with pytest.raises(ValueError, match="banana"):
        parse_keys("ctrl+banana")
    with pytest.raises(ValueError):
        parse_keys("")


def test_is_suzy_window():
    assert is_suzy_window(SUZY) and is_suzy_window(Window(9, "Suzy - main.py", 1, "cmd.exe"))
    assert not is_suzy_window(Window(9, "Suzy fan club - Google Chrome", 1, "chrome.exe"))
    assert not is_suzy_window(CHROME)


@pytest.mark.parametrize("name, expected", [
    ("chrome", "chrome.exe"), ("Google Chrome", "chrome.exe"), ("edge", "msedge.exe"), ("word", "winword.exe"),
    ("notepad app", "notepad.exe"), ("invoice", "winword.exe"),
])
def test_find_windows(name, expected):
    found = windows.find_windows(name, [CHROME, NOTEPAD, EDGE, WORD])
    assert found and found[0].process == expected


def test_find_windows_nothing():
    assert windows.find_windows("photoshop", [CHROME, NOTEPAD]) == []


def test_tracker_never_targets_suzy(desk):
    fake = desk(foreground=SUZY, last=NOTEPAD)
    assert win32.TRACKER.target() == NOTEPAD
    fake.fg = CHROME.hwnd
    assert win32.TRACKER.target() == CHROME
    win32.TRACKER.poll()
    assert win32.TRACKER.last == CHROME
    fake.fg = SUZY.hwnd
    win32.TRACKER.poll()
    assert win32.TRACKER.last == CHROME  # Suzy's console is never remembered


# ----------------------------------------------------------------- closing
def test_close_this_window_typed_in_suzy_closes_previous_window(desk):
    fake = desk(foreground=SUZY, last=NOTEPAD)
    result = windows.close_window()
    assert result.success and fake.closed == [NOTEPAD.hwnd] and SUZY.hwnd in fake.open


def test_close_this_window_by_voice_closes_active_window(desk):
    fake = desk(foreground=EDGE, last=NOTEPAD)
    windows.close_window("this window")
    assert fake.closed == [EDGE.hwnd]


def test_close_named_window(desk):
    fake = desk()
    assert windows.close_window("word").message == "Closed the Word window."
    assert fake.closed == [WORD.hwnd]


def test_close_window_unknown_or_no_target(desk):
    desk()
    assert "can't see any open window" in windows.close_window("photoshop").message
    desk(foreground=SUZY, last=None)
    assert "not sure which window" in windows.close_window().message


def test_close_app_closes_all_its_windows(desk):
    fake = desk()
    result = windows.close_app("chrome")
    assert result.success and result.message == "Closed Chrome."
    assert sorted(fake.closed) == [CHROME.hwnd, CHROME2.hwnd]


def test_close_app_with_unsaved_work(desk):
    fake = desk(unsaved={NOTEPAD.hwnd})
    result = windows.close_app("notepad")
    assert result.success and "asking whether to save" in result.message
    assert NOTEPAD.hwnd in fake.open


def test_close_app_not_open(desk):
    desk()
    assert not windows.close_app("spotify").success


def test_force_close_needs_explicit_words(desk, monkeypatch):
    desk()
    killed = []
    monkeypatch.setattr(windows, "_force_kill", lambda program: killed.append(program) or 1)
    context.set_user_message("close notepad")
    assert not windows.close_app("notepad", force=True).success and killed == []
    context.set_user_message("force close notepad")
    assert windows.close_app("notepad", force=True).message == "Force-closed Notepad."
    assert killed == ["notepad.exe"]


def test_force_close_refuses_windows_and_suzy(desk, monkeypatch):
    desk()
    monkeypatch.setattr(windows, "_force_kill", lambda program: pytest.fail("must not kill"))
    context.set_user_message("force kill explorer")
    assert "part of Windows" in windows.close_app("explorer", force=True).message
    context.set_user_message("force kill python")
    assert not windows.close_app("python", force=True).success


# ----------------------------------------------------------------- tabs & keys
def test_close_tab_in_active_browser(desk):
    fake = desk(foreground=EDGE)
    assert windows.close_tab().message == "Closed the tab in Edge."
    assert fake.pressed == [(EDGE.hwnd, [0x11, ord("W")])]


def test_close_tab_from_suzy_console_goes_to_a_browser(desk):
    fake = desk(foreground=SUZY, last=NOTEPAD)
    windows.close_tab()
    assert fake.activated == [CHROME.hwnd] and fake.pressed[0][0] == CHROME.hwnd


def test_close_tab_in_named_app(desk):
    fake = desk(foreground=CHROME)
    windows.close_tab("edge")
    assert fake.activated == [EDGE.hwnd] and fake.pressed[0][0] == EDGE.hwnd


def test_press_keys_and_type_text(desk):
    fake = desk(foreground=SUZY, last=NOTEPAD)
    assert windows.press_keys("ctrl+s").message == "Pressed CTRL+S."
    assert fake.pressed == [(NOTEPAD.hwnd, [0x11, ord("S")])]
    assert windows.type_text("नमस्ते Sagar").success
    assert fake.typed == [(NOTEPAD.hwnd, "नमस्ते Sagar")]
    assert not windows.press_keys("ctrl+banana").success
    assert not windows.type_text("").success


def test_switch_minimize_maximize_and_desktop(desk):
    fake = desk(foreground=SUZY, last=NOTEPAD)
    assert windows.switch_to_window("word").message == "Switched to Word."
    assert fake.activated == [WORD.hwnd]
    assert "want me to open it" in windows.switch_to_window("spotify").message
    windows.minimize_window()
    windows.maximize_window("chrome")
    assert fake.minimized == [WORD.hwnd] and fake.maximized == [CHROME.hwnd]  # Word is now in front
    assert windows.minimize_all().success and fake.desktop_shown == 1


def test_list_open_windows_hides_suzy(desk):
    desk()
    result = windows.list_open_windows()
    assert "Suzy" not in [t for t in result.data["windows"]] and len(result.data["windows"]) == 5
    desk(windows_list=[SUZY], foreground=SUZY, last=None)
    assert windows.list_open_windows().message == "No windows are open."


def test_real_desktop_off_windows(registry, monkeypatch):
    monkeypatch.setattr(winutils, "IS_WINDOWS", False)
    monkeypatch.setattr(win32, "DESKTOP", win32.Desktop())
    monkeypatch.setattr(win32, "TRACKER", ForegroundTracker())
    result = registry.execute("close_window", {})
    assert not result.success and "only works on Windows" in result.message
