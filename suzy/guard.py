"""Suzy's guardian: starts Suzy and rescues her if a self-update breaks her.

Suzy can rewrite any of her own code, except this file. The guard:

1. starts ``main.py``;
2. restarts it when Suzy exits with code 3 (she just installed an update);
3. if Suzy crashes after installing an update, before she confirmed that
   the update starts up fine, it restores the backup taken just before that
   update and starts her again.

It deliberately uses only the Python standard library and imports nothing
from Suzy, so a broken Suzy can never break her own rescue.

Usage:  python guard.py [any main.py arguments]
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESTART_EXIT_CODE = 3
MARKER = ROOT / "data" / "update_marker.json"          # written when an update is installed
ROLLBACK_NOTICE = ROOT / "data" / "rollback_notice.json"  # read by Suzy on her next start
MAX_QUICK_RESTARTS = 5

# Folders and files that are not "code": never backed up, restored or deleted.
EXCLUDED_DIRS = {
    ".venv", "venv", "data", "logs", "models", "backups", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".git", "build", "dist",
}
EXCLUDED_FILES = {".env"}
PROTECTED_FILES = {"guard.py", ".env"}


def iter_code_files(root: Path) -> dict[str, Path]:
    """All of Suzy's code/config files, keyed by forward-slash relative path."""
    files: dict[str, Path] = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for name in names:
            if name in EXCLUDED_FILES or name.endswith((".pyc", ".pyo")):
                continue
            path = Path(current) / name
            files[path.relative_to(root).as_posix()] = path
    return files


def restore_snapshot(root: Path, snapshot_dir: Path) -> list[str]:
    """Make Suzy's code exactly match a backup. Returns the paths that changed."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(manifest["files"])
    changed: list[str] = []
    # Remove files the update added.
    for rel, path in iter_code_files(root).items():
        if rel not in wanted and rel not in PROTECTED_FILES:
            path.unlink()
            changed.append(rel)
    # Put every backed-up file back.
    for rel in sorted(wanted):
        if rel in PROTECTED_FILES:
            continue
        source = snapshot_dir / "files" / rel
        target = root / rel
        if target.exists() and target.read_bytes() == source.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        changed.append(rel)
    return changed


def _rollback() -> None:
    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    snapshot = Path(marker["snapshot"])
    request = marker.get("request", "my last update")
    print(f"\n[Guard] Suzy crashed right after installing the update: {request}")
    print("[Guard] Restoring the version from before that update...")
    changed = restore_snapshot(ROOT, snapshot)
    MARKER.unlink()
    ROLLBACK_NOTICE.parent.mkdir(parents=True, exist_ok=True)
    ROLLBACK_NOTICE.write_text(json.dumps({
        "request": request,
        "snapshot": str(snapshot),
        "restored_files": changed,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[Guard] Restored {len(changed)} file(s). Starting Suzy again.\n")


def run(argv: list[str]) -> int:
    env = dict(os.environ, SUZY_GUARDED="1")
    quick_restarts = 0
    while True:
        started = time.monotonic()
        # Ctrl+C reaches Suzy directly (same console); the guard ignores it while she runs.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            code = subprocess.call([sys.executable, str(ROOT / "main.py"), *argv], cwd=ROOT, env=env)
        finally:
            signal.signal(signal.SIGINT, previous)

        quick_restarts = quick_restarts + 1 if time.monotonic() - started < 15 else 0
        if quick_restarts > MAX_QUICK_RESTARTS:
            print("[Guard] Suzy keeps stopping right after starting - giving up for now.")
            return code

        if code == RESTART_EXIT_CODE:
            print("\n[Guard] Restarting Suzy...\n")
            argv = [arg for arg in argv if arg != "--startup"]  # say "welcome back" only once
            continue
        if code != 0 and MARKER.exists():
            try:
                _rollback()
            except Exception as exc:  # noqa: BLE001 - report, never crash the guard
                print(f"[Guard] Automatic restore FAILED: {exc}")
                print("[Guard] Your backups are in the 'backups' folder; re-running the installer also repairs Suzy.")
                return code
            continue
        return code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
