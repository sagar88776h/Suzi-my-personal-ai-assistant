"""System-wide hotkey (default Ctrl+Alt+S) that works from any window."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger("suzy.hotkey")

_NAMED_KEYS = {
    "ctrl", "alt", "shift", "cmd", "space", "enter", "esc", "tab", "home", "end",
    "page_up", "page_down", "insert", "delete", "up", "down", "left", "right",
} | {f"f{n}" for n in range(1, 25)}
_ALIASES = {"control": "ctrl", "win": "cmd", "windows": "cmd", "super": "cmd", "escape": "esc", "return": "enter"}


def to_pynput(combo: str) -> str:
    """Convert "ctrl+alt+s" to pynput's "<ctrl>+<alt>+s" format."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    converted = []
    for part in parts:
        part = _ALIASES.get(part, part)
        if part in _NAMED_KEYS:
            converted.append(f"<{part}>")
        elif len(part) == 1:
            converted.append(part)
        else:
            raise ValueError(f"unknown key '{part}' in hotkey '{combo}'")
    return "+".join(converted)


class GlobalHotkey:
    """Calls ``callback`` on a worker thread whenever the hotkey is pressed."""

    DEBOUNCE_SECONDS = 0.4

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self.callback = callback
        self.error: str | None = None
        self._listener: Any = None
        self._last_fired = 0.0

    def _fire(self) -> None:
        now = time.monotonic()
        if now - self._last_fired < self.DEBOUNCE_SECONDS:
            return
        self._last_fired = now

        def run() -> None:
            try:
                self.callback()
            except Exception:  # noqa: BLE001 - a hotkey must never kill the listener
                log.exception("Hotkey callback failed")

        threading.Thread(target=run, name="HotkeyAction", daemon=True).start()

    def start(self) -> bool:
        """Start listening. Returns False (with ``error`` set) if the hotkey can't be registered."""
        try:
            pynput_combo = to_pynput(self.combo)
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys({pynput_combo: self._fire})
            self._listener.daemon = True
            self._listener.start()
        except ValueError as exc:
            self.error = f"The hotkey '{self.combo}' in config.yaml isn't valid ({exc})."
        except Exception as exc:  # noqa: BLE001 - ImportError, OS hook failures, ...
            self.error = f"I couldn't register the {self.combo} hotkey ({exc})."
        if self.error:
            log.warning(self.error)
            return False
        log.info("Global hotkey %s registered", self.combo)
        return True

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                log.debug("Stopping the hotkey listener failed", exc_info=True)
            self._listener = None
