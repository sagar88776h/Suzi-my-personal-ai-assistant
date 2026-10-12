"""App tools: find installed applications and launch them by spoken name.

Apps are discovered from four sources (highest priority first):

1. Start Menu shortcuts (``.lnk``/``.url``/``.appref-ms``) for all users and
   the current user.
2. ``Get-StartApps`` - every app on the Start menu, including Microsoft Store
   apps such as Calculator, WhatsApp or Spotify.
3. Built-in Windows programs (Notepad, Task Manager, Settings, ...).
4. ``.exe`` files in Program Files / ``%LOCALAPPDATA%\\Programs`` (two levels deep).

Spoken names rarely match exactly ("open chrome" vs "Google Chrome"), so the
lookup falls back to fuzzy matching and suggests the closest name when unsure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from tools import winutils
from tools.registry import ToolResult, tool

log = logging.getLogger("suzy.tools.apps")

SHORTCUT_EXTENSIONS = {".lnk", ".url", ".appref-ms"}

# Start-menu entries that are never what the user means by "open X".
NOISE_WORDS = (
    "uninstall", "unins", "readme", "read me", "release notes", "license",
    "documentation", "help", "website", "web site", "manual", "changelog",
    "setup", "updater", "update", "crash", "repair", "eula", "support",
)

# Built-in Windows programs and URIs that ShellExecute can open by name.
BUILTIN_APPS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "command prompt": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "windows terminal": "wt.exe",
    "terminal": "wt.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "this pc": "explorer.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",
    "snipping tool": "ms-screenclip:",
    "microsoft store": "ms-windows-store:",
    "store": "ms-windows-store:",
    "camera": "microsoft.windows.camera:",
    "clock": "ms-clock:",
    "alarms": "ms-clock:",
    "photos": "ms-photos:",
    "microsoft edge": "microsoft-edge:",
    "edge": "microsoft-edge:",
    "device manager": "devmgmt.msc",
    "disk management": "diskmgmt.msc",
    "registry editor": "regedit.exe",
    "character map": "charmap.exe",
    "on screen keyboard": "osk.exe",
    "magnifier": "magnify.exe",
    "resource monitor": "resmon.exe",
}

# Lower number = preferred when two entries have the same name.
KIND_PRIORITY = {"shortcut": 0, "startapp": 1, "builtin": 2, "exe": 3}


@dataclass(frozen=True)
class AppEntry:
    name: str    # display name, e.g. "Google Chrome"
    target: str  # path, URI or Start-menu AppID used to launch it
    kind: str    # "shortcut" | "startapp" | "builtin" | "exe"

    @property
    def key(self) -> str:
        return normalize_name(self.name)


@dataclass
class AppMatch:
    entry: AppEntry | None
    score: float
    suggestion: AppEntry | None = None


def normalize_name(text: str) -> str:
    """Lower-case, strip filler words and punctuation: "the Chrome app" -> "chrome"."""
    text = text.lower().strip()
    text = re.sub(r"\.(exe|lnk|url|appref-ms)$", "", text)
    text = re.sub(r"[_\-\.]+", " ", text)
    text = re.sub(r"[^\w\s+#]", "", text)
    text = re.sub(r"^(the|my)\s+", "", text)
    text = re.sub(r"\s+(app|application|program|software)$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in NOISE_WORDS)


def default_start_menu_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env_var in ("ProgramData", "APPDATA"):
        base = os.environ.get(env_var)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def default_program_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_var)
        if base:
            dirs.append(Path(base))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Programs")
    return dirs


def _walk_limited(root: Path, max_depth: int) -> Iterable[Path]:
    """Yield files under ``root`` no deeper than ``max_depth`` folders."""
    root_depth = len(root.parts)
    for current, subdirs, files in os.walk(root, onerror=lambda _err: None):
        depth = len(Path(current).parts) - root_depth
        if depth >= max_depth:
            subdirs[:] = []
        for filename in files:
            yield Path(current) / filename


class AppIndex:
    """Cached list of launchable apps with fuzzy lookup. Thread-safe."""

    REFRESH_AFTER_SECONDS = 600

    def __init__(
        self,
        start_menu_dirs: list[Path] | None = None,
        program_dirs: list[Path] | None = None,
        include_start_apps: bool | None = None,
        include_builtins: bool = True,
        fuzzy_threshold: float = 80,
        suggest_threshold: float = 60,
    ) -> None:
        self.start_menu_dirs = start_menu_dirs if start_menu_dirs is not None else default_start_menu_dirs()
        self.program_dirs = program_dirs if program_dirs is not None else default_program_dirs()
        self.include_start_apps = winutils.IS_WINDOWS if include_start_apps is None else include_start_apps
        self.include_builtins = include_builtins
        self.fuzzy_threshold = fuzzy_threshold
        self.suggest_threshold = suggest_threshold
        self._entries: list[AppEntry] = []
        self._built_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ building
    def _scan_shortcuts(self) -> list[AppEntry]:
        entries = []
        for folder in self.start_menu_dirs:
            if not folder.is_dir():
                continue
            for path in _walk_limited(folder, max_depth=6):
                if path.suffix.lower() in SHORTCUT_EXTENSIONS and not _is_noise(path.stem):
                    entries.append(AppEntry(path.stem, str(path), "shortcut"))
        return entries

    def _scan_start_apps(self) -> list[AppEntry]:
        try:
            completed = winutils.run_powershell(
                "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress", timeout=20
            )
        except (subprocess.SubprocessError, OSError, winutils.PlatformNotSupportedError) as exc:
            log.warning("Get-StartApps failed: %s", exc)
            return []
        output = (completed.stdout or "").strip()
        if completed.returncode != 0 or not output:
            log.warning("Get-StartApps returned code %s: %s", completed.returncode, completed.stderr)
            return []
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            log.warning("Could not parse Get-StartApps output")
            return []
        if isinstance(data, dict):
            data = [data]
        entries = []
        for item in data:
            name, app_id = item.get("Name"), item.get("AppID")
            if name and app_id and not _is_noise(name):
                entries.append(AppEntry(name, app_id, "startapp"))
        return entries

    def _scan_program_exes(self) -> list[AppEntry]:
        entries = []
        for folder in self.program_dirs:
            if not folder.is_dir():
                continue
            for path in _walk_limited(folder, max_depth=2):
                if path.suffix.lower() == ".exe" and not _is_noise(path.stem):
                    entries.append(AppEntry(path.stem, str(path), "exe"))
        return entries

    def build(self) -> list[AppEntry]:
        """Rescan every source. Returns the new entry list."""
        started = time.perf_counter()
        collected: list[AppEntry] = []
        collected += self._scan_shortcuts()
        if self.include_start_apps:
            collected += self._scan_start_apps()
        if self.include_builtins:
            collected += [AppEntry(name.title(), target, "builtin") for name, target in BUILTIN_APPS.items()]
        collected += self._scan_program_exes()

        # De-duplicate by normalized name, keeping the most reliable source.
        best: dict[str, AppEntry] = {}
        for entry in collected:
            key = entry.key
            if not key:
                continue
            current = best.get(key)
            if current is None or KIND_PRIORITY[entry.kind] < KIND_PRIORITY[current.kind]:
                best[key] = entry

        with self._lock:
            self._entries = list(best.values())
            self._built_at = time.monotonic()
        log.info("App index built: %d apps in %.2fs", len(best), time.perf_counter() - started)
        return list(best.values())

    def entries(self, force_refresh: bool = False) -> list[AppEntry]:
        with self._lock:
            stale = time.monotonic() - self._built_at > self.REFRESH_AFTER_SECONDS
            cached = list(self._entries)
        if force_refresh or not cached or stale:
            return self.build()
        return cached

    @property
    def age_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._built_at if self._built_at else float("inf")

    # ------------------------------------------------------------------ lookup
    def find(self, query: str, entries: list[AppEntry] | None = None) -> AppMatch:
        """Find the best app for a spoken name like "chrome" or "vs code"."""
        wanted = normalize_name(query)
        if not wanted:
            return AppMatch(None, 0.0)
        entries = entries if entries is not None else self.entries()
        if not entries:
            return AppMatch(None, 0.0)

        exact = [e for e in entries if e.key == wanted]
        if exact:
            return AppMatch(min(exact, key=lambda e: KIND_PRIORITY[e.kind]), 100.0)

        keys = [e.key for e in entries]
        candidates = process.extract(wanted, keys, scorer=fuzz.WRatio, limit=8)
        if not candidates:
            return AppMatch(None, 0.0)

        def rank(candidate: tuple[str, float, int]) -> tuple[float, int, int]:
            _key, score, index = candidate
            entry = entries[index]
            # Whole-word containment ("chrome" in "google chrome") beats partial
            # character overlap ("chrome" in "chromedriver").
            whole_word = 1 if re.search(rf"\b{re.escape(wanted)}\b", entry.key) else 0
            return (-(score + 5 * whole_word), KIND_PRIORITY[entry.kind], len(entry.key))

        best_key, best_score, best_index = sorted(candidates, key=rank)[0]
        best_entry = entries[best_index]
        if best_score >= self.fuzzy_threshold:
            return AppMatch(best_entry, best_score)
        if best_score >= self.suggest_threshold:
            return AppMatch(None, best_score, suggestion=best_entry)
        return AppMatch(None, best_score)


APP_INDEX = AppIndex()


def configure(fuzzy_threshold: float = 80, suggest_threshold: float = 60,
              extra_search_dirs: list[str] | None = None) -> None:
    """Apply settings from config.yaml (called once at startup)."""
    APP_INDEX.fuzzy_threshold = fuzzy_threshold
    APP_INDEX.suggest_threshold = suggest_threshold
    for folder in extra_search_dirs or []:
        path = Path(os.path.expandvars(folder)).expanduser()
        if path not in APP_INDEX.program_dirs:
            APP_INDEX.program_dirs.append(path)


def prewarm_index() -> None:
    """Build the app index on a background thread so the first "open X" is instant."""

    def _build() -> None:
        try:
            APP_INDEX.build()
        except Exception:  # noqa: BLE001 - background warm-up must never crash
            log.exception("Background app index build failed")

    threading.Thread(target=_build, name="AppIndexWarmup", daemon=True).start()


def launch(entry: AppEntry) -> None:
    """Start an app entry using the right Windows mechanism for its kind."""
    if entry.kind == "startapp":
        winutils.launch_shell_app(entry.target)
    else:
        winutils.startfile(entry.target)


# ============================================================================ tools
@tool(
    description=(
        "Open (launch) an installed application on the PC by name, e.g. 'chrome', "
        "'notepad', 'spotify', 'vs code', 'calculator', 'settings'. Fuzzy names are fine."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The app name as the user said it."},
        },
        "required": ["name"],
    },
    category="apps",
)
def open_app(name: str) -> ToolResult:
    name = name.strip()
    if not normalize_name(name):
        return ToolResult.fail("Which app should I open?")

    match = APP_INDEX.find(name)
    # Not found in a cache older than a minute? The app may be newly installed.
    if match.entry is None and APP_INDEX.age_seconds > 60:
        match = APP_INDEX.find(name, APP_INDEX.entries(force_refresh=True))

    if match.entry is None:
        if match.suggestion is not None:
            return ToolResult.fail(
                f"I couldn't find an app called {name}. Did you mean {match.suggestion.name}?",
                suggestion=match.suggestion.name,
            )
        return ToolResult.fail(f"I couldn't find an app called {name} on this PC.")

    try:
        launch(match.entry)
    except winutils.PlatformNotSupportedError:
        raise
    except OSError as exc:
        log.warning("Launching %s failed: %s", match.entry, exc)
        return ToolResult.fail(f"I found {match.entry.name}, but Windows wouldn't start it ({exc.strerror or exc}).")
    return ToolResult.ok(f"Opening {match.entry.name}.", app=match.entry.name, source=match.entry.kind)
