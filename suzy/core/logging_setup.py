"""Rotating log files for Suzy.

Files written to ``logs/``:

* ``suzy.log``    - everything at the configured level (DEBUG by default)
* ``errors.log``  - ERROR and above only, for quick troubleshooting
* ``actions.log`` - one line per PC action Suzy executes (audit trail)
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.paths import LOG_DIR

_FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s"
_ACTION_FORMAT = "%(asctime)s | %(message)s"
_configured = False


def setup_logging(
    level: str = "DEBUG",
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
    log_dir: Path = LOG_DIR,
    console_level: str = "WARNING",
) -> None:
    """Configure the ``suzy`` logger hierarchy. Safe to call more than once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("suzy")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    formatter = logging.Formatter(_FORMAT)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        main_handler = RotatingFileHandler(
            log_dir / "suzy.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        main_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        main_handler.setFormatter(formatter)
        root.addHandler(main_handler)

        error_handler = RotatingFileHandler(
            log_dir / "errors.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)

        actions = logging.getLogger("suzy.actions")
        actions.setLevel(logging.INFO)
        actions.propagate = False  # keep the audit trail in its own file
        action_handler = RotatingFileHandler(
            log_dir / "actions.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        action_handler.setFormatter(logging.Formatter(_ACTION_FORMAT))
        actions.addHandler(action_handler)
    except OSError as exc:
        # A read-only folder must not stop Suzy from running; log to console only.
        print(f"[Suzy] Warning: could not create log files in {log_dir}: {exc}")

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.WARNING))
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console)

    _configured = True
