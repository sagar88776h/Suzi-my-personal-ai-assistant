"""Small Windows helpers shared by the tool modules.

Everything that touches a Windows-only API goes through this module, which
makes the tools easy to unit-test on any OS by patching these functions.
"""

from __future__ import annotations

import os
import subprocess
import sys

IS_WINDOWS: bool = sys.platform == "win32"

# Prevents a console window from flashing when launching PowerShell.
CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class PlatformNotSupportedError(RuntimeError):
    """Raised when a Windows-only action is attempted on another OS."""

    def __init__(self, action: str = "That") -> None:
        super().__init__(f"{action} only works on Windows.")


def require_windows(action: str = "That") -> None:
    if not IS_WINDOWS:
        raise PlatformNotSupportedError(action)


def startfile(target: str) -> None:
    """Open a file, shortcut, executable or URI the way a double-click would."""
    require_windows("Launching programs")
    os.startfile(target)  # type: ignore[attr-defined]  # Windows-only API


def launch_shell_app(app_id: str) -> None:
    """Launch a Start-menu app (including Microsoft Store apps) by its AppID."""
    require_windows("Launching programs")
    subprocess.Popen(  # noqa: S603 - fixed executable, no shell
        ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
        creationflags=CREATE_NO_WINDOW,
    )


def run_powershell(script: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    """Run a PowerShell snippet and capture its UTF-8 output.

    Raises ``subprocess.TimeoutExpired`` if it takes longer than ``timeout``.
    """
    require_windows("PowerShell")
    prefix = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    return subprocess.run(  # noqa: S603 - fixed executable, no shell
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            prefix + script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )
