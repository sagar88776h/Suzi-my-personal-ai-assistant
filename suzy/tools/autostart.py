"""Start Suzy automatically when Windows starts (a shortcut in the Startup folder).

The shortcut runs "Start Suzy.bat --startup" in a minimised window; with
``--startup`` Suzy greets Sagar with "Hello Sagar, welcome back."
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from core.paths import APP_DIR
from tools import winutils
from tools.registry import ToolResult, tool

log = logging.getLogger("suzy.tools.autostart")

SHORTCUT_NAME = "Suzy.lnk"
LAUNCHER_NAME = "Start Suzy.bat"


def startup_folder() -> Path:
    """The current user's Startup folder (programs here run at login)."""
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path:
    return startup_folder() / SHORTCUT_NAME


def is_enabled() -> bool:
    return shortcut_path().exists()


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def enable(app_dir: Path | None = None) -> None:
    """Create the Startup shortcut. Raises OSError / PlatformNotSupportedError on failure."""
    app_dir = app_dir or APP_DIR
    launcher = app_dir / LAUNCHER_NAME
    if not launcher.exists():
        raise OSError(f"{LAUNCHER_NAME} is missing - re-run the installer.")
    startup_folder().mkdir(parents=True, exist_ok=True)
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$s = $shell.CreateShortcut({_ps_quote(str(shortcut_path()))}); "
        f"$s.TargetPath = {_ps_quote(str(launcher))}; "
        "$s.Arguments = '--startup'; "
        f"$s.WorkingDirectory = {_ps_quote(str(app_dir))}; "
        "$s.WindowStyle = 7; "  # start minimised
        "$s.Description = 'Start Suzy when Windows starts'; "
        "$s.Save()"
    )
    completed = winutils.run_powershell(script, timeout=20)
    if completed.returncode != 0 or not shortcut_path().exists():
        raise OSError((completed.stderr or "the shortcut wasn't created").strip()[:200])


def disable() -> None:
    try:
        shortcut_path().unlink()
    except FileNotFoundError:
        pass


@tool(
    description=(
        "Turn on or off starting Suzy automatically when Windows starts (she then says "
        "'Hello Sagar, welcome back')."
    ),
    parameters={
        "type": "object",
        "properties": {"enabled": {"type": "boolean", "description": "true = start with Windows."}},
        "required": ["enabled"],
    },
    category="settings",
)
def set_autostart(enabled: bool) -> ToolResult:
    winutils.require_windows("Starting with Windows")
    if not enabled:
        disable()
        return ToolResult.ok("Okay, I won't start with Windows any more.")
    try:
        enable()
    except OSError as exc:
        log.warning("Couldn't enable autostart: %s", exc)
        return ToolResult.fail(f"I couldn't add myself to Windows startup ({exc}).")
    return ToolResult.ok("Done! I'll start with Windows and say hello when you log in.")
