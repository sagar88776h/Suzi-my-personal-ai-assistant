"""Safety classification for Suzy's tools.

Every tool is tagged with a :class:`SafetyLevel` when it is registered:

* ``SAFE``      - runs immediately (open an app, tell the time, ...).
* ``CONFIRM``   - Suzy must ask "are you sure?" first (shutdown, delete, ...).
* ``BLOCKED``   - refused unless the user explicitly enables it in config.
"""

from __future__ import annotations

from enum import Enum


class ExecutionLevel(str, Enum):
    LEVEL_1_READ = "read"        # Retrieve, search, analyze, and summarize authorized info
    LEVEL_2_PREPARE = "prepare"  # Create drafts, prepare changes, propose actions
    LEVEL_3_EXECUTE = "execute"  # Carry out explicitly authorized consequential actions


class SafetyLevel(str, Enum):
    SAFE = "safe"                          # Level 1 Read: runs immediately
    PREPARE = "prepare"                    # Level 2 Prepare: drafts/proposals
    CONFIRM = "confirm"                    # Level 3 Execute: requires explicit confirmation
    BLOCKED = "blocked_unless_enabled"     # Prohibited unless explicitly enabled in config
