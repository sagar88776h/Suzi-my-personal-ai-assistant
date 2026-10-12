"""Safety classification for Suzy's tools.

Every tool is tagged with a :class:`SafetyLevel` when it is registered:

* ``SAFE``      - runs immediately (open an app, tell the time, ...).
* ``CONFIRM``   - Suzy must ask "are you sure?" first (shutdown, delete, ...).
* ``BLOCKED``   - refused unless the user explicitly enables it in config.
"""

from __future__ import annotations

from enum import Enum


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked_unless_enabled"
