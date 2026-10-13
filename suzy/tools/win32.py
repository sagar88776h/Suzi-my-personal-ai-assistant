"""Windows desktop control through the Win32 API (ctypes - nothing extra to install).

:class:`Desktop` lists, activates, closes, minimises and maximises windows and
sends keystrokes. :class:`ForegroundTracker` remembers the last window Sagar
used that isn't Suzy's own, so "close this window" typed into Suzy's console
closes the window he was working in - never Suzy herself.

Everything goes through the module-level ``DESKTOP`` object so tests can swap
in a fake desktop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from tools import winutils

log = logging.getLogger("suzy.tools.win32")

WM_CLOSE = 0x0010
SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE, SW_SHOW = 3, 6, 9, 5
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
GW_OWNER = 4
DWMWA_CLOAKED = 14
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0001, 0x0002, 0x0004
INPUT_KEYBOARD = 1

# Processes that host Suzy's own console window.
CONSOLE_PROCESSES = {"cmd.exe", "conhost.exe", "openconsole.exe", "windowsterminal.exe", "python.exe",
                     "pythonw.exe", "powershell.exe", "pwsh.exe"}

VK_CODES: dict[str, int] = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B, "windows": 0x5B, "cmd": 0x5B,
    "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "capslock": 0x14, "menu": 0x5D,
    "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stop": 0xB2, "playpause": 0xB3,
    ",": 0xBC, ".": 0xBE, "-": 0xBD, "=": 0xBB, "/": 0xBF, ";": 0xBA, "'": 0xDE,
    "[": 0xDB, "]": 0xDD, "\\": 0xDC, "`": 0xC0,
}
VK_CODES.update({f"f{n}": 0x6F + n for n in range(1, 25)})
EXTENDED_KEYS = {0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22, 0x25, 0x26, 0x27, 0x28, 0x5B, 0x5D}
MODIFIERS = {"ctrl", "control", "alt", "shift", "win", "windows", "cmd"}
KEY_ALIASES = {"pgup": "pageup", "pgdn": "pagedown", "page up": "pageup", "page down": "pagedown",
               "arrow up": "up", "arrow down": "down", "arrow left": "left", "arrow right": "right",
               "plus": "=", "minus": "-", "comma": ",", "period": ".", "dot": ".", "esc key": "esc"}


def parse_keys(combo: str) -> list[int]:
    """"ctrl+shift+t" -> virtual-key codes. Raises ValueError for unknown keys."""
    parts = [p.strip().lower() for p in combo.replace(" + ", "+").split("+") if p.strip()]
    if not parts:
        raise ValueError("no keys given")
    codes = []
    for part in parts:
        part = KEY_ALIASES.get(part, part)
        if part in VK_CODES:
            codes.append(VK_CODES[part])
        elif len(part) == 1 and part.isalnum():
            codes.append(ord(part.upper()))
        else:
            raise ValueError(f"I don't know the key '{part}'")
    return codes


@dataclass(frozen=True)
class Window:
    hwnd: int
    title: str
    pid: int
    process: str         # e.g. "chrome.exe"
    minimized: bool = False

    @property
    def app(self) -> str:
        return self.process.removesuffix(".exe")


def is_suzy_window(window: Window) -> bool:
    """Suzy's own console ("title Suzy" in Start Suzy.bat)."""
    return "suzy" in window.title.lower() and window.process.lower() in CONSOLE_PROCESSES


class Desktop:
    """Thin wrapper over user32 for the window and keyboard operations Suzy needs."""

    def __init__(self) -> None:
        self._user32: Any = None
        self._dwmapi: Any = None

    def _api(self) -> Any:
        winutils.require_windows("Controlling windows")
        if self._user32 is None:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            # Declare exact types: window handles are pointer-sized on 64-bit Windows.
            hwnd = wintypes.HWND
            signatures = {
                "GetWindowTextLengthW": ([hwnd], ctypes.c_int),
                "GetWindowTextW": ([hwnd, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
                "GetWindowThreadProcessId": ([hwnd, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
                "IsWindowVisible": ([hwnd], wintypes.BOOL),
                "IsWindow": ([hwnd], wintypes.BOOL),
                "IsIconic": ([hwnd], wintypes.BOOL),
                "GetWindow": ([hwnd, wintypes.UINT], hwnd),
                "GetWindowLongW": ([hwnd, ctypes.c_int], wintypes.LONG),
                "GetForegroundWindow": ([], hwnd),
                "SetForegroundWindow": ([hwnd], wintypes.BOOL),
                "ShowWindow": ([hwnd, ctypes.c_int], wintypes.BOOL),
                "PostMessageW": ([hwnd, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], wintypes.BOOL),
                "keybd_event": ([wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t], None),
                "EnumWindows": ([ctypes.c_void_p, wintypes.LPARAM], wintypes.BOOL),
            }
            for name, (argtypes, restype) in signatures.items():
                function = getattr(user32, name)
                function.argtypes = argtypes
                function.restype = restype
            self._user32 = user32
            try:
                self._dwmapi = ctypes.WinDLL("dwmapi")
                self._dwmapi.DwmGetWindowAttribute.argtypes = [hwnd, wintypes.DWORD, ctypes.c_void_p,
                                                               wintypes.DWORD]
            except (OSError, AttributeError):
                self._dwmapi = None
        return self._user32

    # ------------------------------------------------------------------ windows
    def _describe(self, hwnd: int) -> Window | None:
        import ctypes
        from ctypes import wintypes

        user32 = self._api()
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return None
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            import psutil

            process = psutil.Process(pid.value).name().lower()
        except Exception:  # noqa: BLE001 - process exited / access denied
            process = ""
        return Window(int(hwnd), buffer.value, int(pid.value), process, bool(user32.IsIconic(hwnd)))

    def _is_app_window(self, hwnd: int) -> bool:
        import ctypes

        user32 = self._api()
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, GW_OWNER):
            return False
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return False
        if self._dwmapi is not None:  # hidden (cloaked) Store-app windows
            cloaked = ctypes.c_int(0)
            self._dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            if cloaked.value:
                return False
        return True

    def windows(self) -> list[Window]:
        """Visible application windows, front-most first."""
        import ctypes
        from ctypes import wintypes

        user32 = self._api()
        found: list[Window] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd: Any, _lparam: Any) -> bool:
            try:
                if self._is_app_window(hwnd):
                    window = self._describe(hwnd)
                    if window and window.title != "Program Manager":
                        found.append(window)
            except Exception:  # noqa: BLE001 - one odd window must not stop the scan
                log.debug("Skipping a window", exc_info=True)
            return True

        user32.EnumWindows(callback, 0)
        return found

    def foreground(self) -> Window | None:
        hwnd = self._api().GetForegroundWindow()
        return self._describe(hwnd) if hwnd else None

    def exists(self, hwnd: int) -> bool:
        return bool(self._api().IsWindow(hwnd))

    def close(self, hwnd: int) -> None:
        """Ask the window to close, exactly like clicking its X (apps can still ask to save)."""
        self._api().PostMessageW(hwnd, WM_CLOSE, 0, 0)

    def activate(self, hwnd: int) -> bool:
        user32 = self._api()
        user32.ShowWindow(hwnd, SW_RESTORE if user32.IsIconic(hwnd) else SW_SHOW)
        # Windows only lets the foreground app hand over focus; a tap of Alt unlocks it.
        user32.keybd_event(VK_CODES["alt"], 0, 0, 0)
        result = bool(user32.SetForegroundWindow(hwnd))
        user32.keybd_event(VK_CODES["alt"], 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.15)  # let the window take focus before keys are sent
        return result

    def minimize(self, hwnd: int) -> None:
        self._api().ShowWindow(hwnd, SW_MINIMIZE)

    def maximize(self, hwnd: int) -> None:
        self._api().ShowWindow(hwnd, SW_MAXIMIZE)

    def restore(self, hwnd: int) -> None:
        self._api().ShowWindow(hwnd, SW_RESTORE)

    # ------------------------------------------------------------------ keyboard
    def press(self, codes: list[int]) -> None:
        """Press a key combination: modifiers down, last key, then everything up."""
        user32 = self._api()

        def send(code: int, up: bool) -> None:
            flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if code in EXTENDED_KEYS else 0)
            user32.keybd_event(code, 0, flags, 0)

        for code in codes:
            send(code, up=False)
        for code in reversed(codes):
            send(code, up=True)

    def type_text(self, text: str) -> None:
        """Type any text (Hindi too) as Unicode keystrokes."""
        import ctypes
        from ctypes import wintypes

        user32 = self._api()

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class _UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", _UNION)]

        units = text.replace("\n", "\r").encode("utf-16-le")
        for index in range(0, len(units), 2):
            unit = int.from_bytes(units[index:index + 2], "little")
            for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                event = INPUT(type=INPUT_KEYBOARD)
                event.union.ki = KEYBDINPUT(0, unit, flags, 0, 0)
                user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
            time.sleep(0.003)

    def show_desktop(self) -> None:
        self.press([VK_CODES["win"], ord("D")])


DESKTOP = Desktop()


# =========================================================================== focus tracking
class ForegroundTracker:
    """Remembers the last window Sagar used that isn't Suzy's own console."""

    def __init__(self, desktop: Any = None, interval: float = 0.4) -> None:
        self.desktop = desktop
        self.interval = interval
        self.last: Window | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _desktop(self) -> Any:
        return self.desktop or DESKTOP

    def poll(self) -> None:
        window = self._desktop().foreground()
        if window is not None and not is_suzy_window(window):
            self.last = window

    def start(self) -> None:
        if self._thread is not None or not winutils.IS_WINDOWS:
            return

        def run() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.poll()
                except Exception:  # noqa: BLE001 - tracking is best-effort
                    log.debug("Foreground poll failed", exc_info=True)

        self._thread = threading.Thread(target=run, name="ForegroundTracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def target(self) -> Window | None:
        """The window an action like "close this" means: the active one, unless that's Suzy."""
        desktop = self._desktop()
        current = desktop.foreground()
        if current is not None and not is_suzy_window(current):
            return current
        if self.last is not None and desktop.exists(self.last.hwnd):
            return self.last
        return None


TRACKER = ForegroundTracker()
