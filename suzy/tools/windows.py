"""Window, tab and keyboard tools: close / switch / minimise / maximise, close tabs, press keys, type.

"This window" means the window Sagar is using - or, when he's typing in Suzy's own
console, the window he used just before (see tools/win32.ForegroundTracker).
Closing works like clicking X, so apps still ask to save unsaved work; killing a
program only happens when Sagar explicitly asks to force-close it.
"""

from __future__ import annotations

import logging
import re
import time

from rapidfuzz import fuzz

from tools import context, win32
from tools.registry import ToolResult, tool

log = logging.getLogger("suzy.tools.windows")

# Spoken name -> program name, where they differ.
APP_ALIASES = {
    "edge": "msedge", "microsoft edge": "msedge", "word": "winword", "microsoft word": "winword",
    "excel": "excel", "powerpoint": "powerpnt", "file explorer": "explorer", "explorer": "explorer",
    "files": "explorer", "folder": "explorer", "vs code": "code", "vscode": "code", "visual studio code": "code",
    "teams": "ms-teams", "task manager": "taskmgr", "settings": "systemsettings", "whatsapp": "whatsapp",
    "calculator": "calculatorapp", "media player": "microsoft.media.player", "photos": "photos",
    "paint": "mspaint", "command prompt": "cmd", "terminal": "windowsterminal",
}
BROWSERS = ("chrome", "msedge", "firefox", "brave", "opera", "vivaldi")
# Never force-killed: Windows itself.
PROTECTED_PROCESSES = {"explorer.exe", "dwm.exe", "winlogon.exe", "csrss.exe", "lsass.exe", "services.exe",
                       "svchost.exe", "smss.exe", "wininit.exe", "system", "fontdrvhost.exe"}
FORCE_WORDS = re.compile(r"\b(force|forcefully|kill|zabardasti|jabardasti|turant)\b", re.IGNORECASE)


def _program(name: str) -> str:
    cleaned = re.sub(r"\b(app|application|window|browser|the|my)\b", "", name.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).removesuffix(".exe")
    return APP_ALIASES.get(cleaned, cleaned)


def find_windows(name: str, windows: list[win32.Window]) -> list[win32.Window]:
    """Windows matching a spoken app or window name ("chrome", "word", "invoice.pdf")."""
    program = _program(name)
    if not program:
        return []
    by_program = [w for w in windows if w.app == program or (len(program) >= 3 and program in w.app)]
    if by_program:
        return by_program
    scored = []
    for window in windows:
        score = max(fuzz.partial_ratio(program, window.title.lower()), fuzz.ratio(program, window.app))
        if score >= 80:
            scored.append((score, window))
    return [w for _score, w in sorted(scored, key=lambda item: -item[0])]


def _visible() -> list[win32.Window]:
    return [w for w in win32.DESKTOP.windows() if not win32.is_suzy_window(w)]


def _pick(name: str | None) -> tuple[win32.Window | None, str]:
    """The window an action is about, plus a message if none was found."""
    if not name or _program(name) in ("this", "current", "active", "it", "is", "yeh", "ye", ""):
        window = win32.TRACKER.target()
        return window, "" if window else "I'm not sure which window you mean - click on it and ask me again."
    matches = find_windows(name, _visible())
    if not matches:
        return None, f"I can't see any open window for {name}."
    return matches[0], ""


def _label(window: win32.Window) -> str:
    app = window.app.replace("msedge", "Edge").replace("winword", "Word").replace("powerpnt", "PowerPoint")
    return app[:1].upper() + app[1:]


# ============================================================================ closing
@tool(
    description=(
        "Close a window: the one Sagar is using ('close this window'), or a named one ('close the Chrome "
        "window', 'close invoice.pdf'). Works like clicking X - apps still ask to save unsaved work."
    ),
    parameters={"type": "object", "properties": {
        "name": {"type": "string", "description": "App or window title; leave empty for the current window."}}},
    category="windows",
)
def close_window(name: str | None = None) -> ToolResult:
    window, problem = _pick(name)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.close(window.hwnd)
    return ToolResult.ok(f"Closed the {_label(window)} window.", title=window.title)


def _wait_closed(windows: list[win32.Window], seconds: float = 3.0) -> list[win32.Window]:
    deadline = time.monotonic() + seconds
    remaining = windows
    while time.monotonic() < deadline:
        remaining = [w for w in remaining if win32.DESKTOP.exists(w.hwnd)]
        if not remaining:
            break
        time.sleep(0.2)
    return remaining


def _force_kill(process_name: str) -> int:
    import psutil

    killed = 0
    for process in psutil.process_iter(["name"]):
        if (process.info.get("name") or "").lower() != process_name:
            continue
        try:
            process.terminate()
            try:
                process.wait(3)
            except psutil.TimeoutExpired:
                process.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


@tool(
    description=(
        "Close an app completely (all its windows), e.g. 'close Chrome', 'Notepad band karo'. Set force=true "
        "ONLY if Sagar explicitly says force-close/kill (unsaved work is lost)."
    ),
    parameters={"type": "object", "properties": {
        "name": {"type": "string", "description": "The app, e.g. 'chrome', 'notepad', 'word'."},
        "force": {"type": "boolean", "description": "Kill it even if it's asking to save. Only on explicit request."}},
        "required": ["name"]},
    category="windows",
)
def close_app(name: str, force: bool = False) -> ToolResult:
    windows = find_windows(name, _visible())
    if force:
        if not FORCE_WORDS.search(context.user_message()) and not context.user_confirmed():
            return ToolResult.fail("Force-closing loses unsaved work - say 'force close' if you're sure.")
        program = windows[0].process if windows else _program(name) + ".exe"
        if program in PROTECTED_PROCESSES or program.startswith("python"):
            return ToolResult.fail(f"I won't force-close {program} - it's part of Windows or of me.")
        killed = _force_kill(program)
        if not killed:
            return ToolResult.fail(f"{name} doesn't seem to be running.")
        return ToolResult.ok(f"Force-closed {_label(windows[0]) if windows else name}.")

    if not windows:
        return ToolResult.fail(f"{name} doesn't seem to be open.")
    app_windows = [w for w in _visible() if w.process == windows[0].process] or windows
    for window in app_windows:
        win32.DESKTOP.close(window.hwnd)
    still_open = _wait_closed(app_windows)
    label = _label(app_windows[0])
    if still_open:
        return ToolResult.ok(f"{label} is asking whether to save your work - answer it, or tell me to "
                             "force close it.", waiting=len(still_open))
    return ToolResult.ok(f"Closed {label}.")


# ============================================================================ tabs & keys
def _focus_for_keys(name: str | None) -> tuple[win32.Window | None, str]:
    window, problem = _pick(name)
    if window is None:
        return None, problem
    current = win32.DESKTOP.foreground()
    if current is None or current.hwnd != window.hwnd:
        win32.DESKTOP.activate(window.hwnd)
    return window, ""


@tool(
    description=(
        "Close the current browser tab (or a tab in a named app, e.g. 'close the Chrome tab'). For 'close all "
        "tabs' use close_window."
    ),
    parameters={"type": "object", "properties": {
        "app": {"type": "string", "description": "Optional app, e.g. 'chrome' or 'edge'."}}},
    category="windows",
)
def close_tab(app: str | None = None) -> ToolResult:
    if not app:
        # The window in use if it's a browser; otherwise the front-most open browser.
        target = win32.TRACKER.target()
        if target is None or not any(b in target.app for b in BROWSERS):
            browsers = [w for w in _visible() if any(b in w.app for b in BROWSERS)]
            if browsers:
                app = browsers[0].app
    window, problem = _focus_for_keys(app)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.press(win32.parse_keys("ctrl+w"))
    return ToolResult.ok(f"Closed the tab in {_label(window)}.")


@tool(
    description=(
        "Press a keyboard shortcut in the window Sagar is using (or a named app), e.g. 'ctrl+shift+t' (reopen tab), "
        "'ctrl+t' (new tab), 'alt+tab', 'f5', 'ctrl+s', 'enter'."
    ),
    parameters={"type": "object", "properties": {
        "keys": {"type": "string", "description": "Keys joined with +, e.g. 'ctrl+s'."},
        "app": {"type": "string", "description": "Optional app to send them to."}}, "required": ["keys"]},
    category="input",
)
def press_keys(keys: str, app: str | None = None) -> ToolResult:
    try:
        codes = win32.parse_keys(keys)
    except ValueError as exc:
        return ToolResult.fail(f"{exc}.")
    window, problem = _focus_for_keys(app)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.press(codes)
    return ToolResult.ok(f"Pressed {keys.upper()}.")


@tool(
    description="Type text into the window Sagar is using (or a named app), like typing on the keyboard.",
    parameters={"type": "object", "properties": {
        "text": {"type": "string"}, "app": {"type": "string", "description": "Optional app to type into."}},
        "required": ["text"]},
    category="input",
)
def type_text(text: str, app: str | None = None) -> ToolResult:
    if not text:
        return ToolResult.fail("What should I type?")
    window, problem = _focus_for_keys(app)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.type_text(text)
    return ToolResult.ok(f"Typed it into {_label(window)}.")


# ============================================================================ arranging
@tool(
    description="Bring an open app or window to the front, e.g. 'switch to Chrome', 'Word pe jao'.",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    category="windows",
)
def switch_to_window(name: str) -> ToolResult:
    matches = find_windows(name, _visible())
    if not matches:
        return ToolResult.fail(f"I can't see {name} open - want me to open it?")
    win32.DESKTOP.activate(matches[0].hwnd)
    return ToolResult.ok(f"Switched to {_label(matches[0])}.")


@tool(
    description="Minimise a window: the current one, or a named app.",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    category="windows",
)
def minimize_window(name: str | None = None) -> ToolResult:
    window, problem = _pick(name)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.minimize(window.hwnd)
    return ToolResult.ok(f"Minimised {_label(window)}.")


@tool(
    description="Maximise (full screen) a window: the current one, or a named app.",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    category="windows",
)
def maximize_window(name: str | None = None) -> ToolResult:
    window, problem = _pick(name)
    if window is None:
        return ToolResult.fail(problem)
    win32.DESKTOP.maximize(window.hwnd)
    return ToolResult.ok(f"Maximised {_label(window)}.")


@tool(description="Minimise all windows and show the desktop.", category="windows")
def minimize_all() -> ToolResult:
    win32.DESKTOP.show_desktop()
    return ToolResult.ok("Here's your desktop.")


@tool(description="List the windows and apps that are open right now.", category="windows", speak_result=False)
def list_open_windows() -> ToolResult:
    windows = _visible()
    if not windows:
        return ToolResult.ok("No windows are open.")
    names = [f"{_label(w)}: {w.title[:60]}" for w in windows[:15]]
    return ToolResult.ok("Open windows - " + "; ".join(names), windows=[w.title for w in windows])
