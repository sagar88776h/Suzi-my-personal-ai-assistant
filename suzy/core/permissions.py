"""Permission and action confirmation system for Suzy.

Implements three execution levels:
  - Level 1 (Read): safe read-only queries (emails, web search, system stats).
  - Level 2 (Prepare): staging changes, draft emails, proposed adjustments.
  - Level 3 (Execute): consequential operations requiring explicit user confirmation.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tools.safety import ExecutionLevel, SafetyLevel

log = logging.getLogger("suzy.permissions")


@dataclass
class PendingAction:
    id: str
    tool_name: str
    arguments: dict[str, Any]
    summary: str
    level: ExecutionLevel = ExecutionLevel.LEVEL_3_EXECUTE
    created_at: datetime = field(default_factory=datetime.now)

    def describe(self) -> str:
        return f"{self.summary} ({self.tool_name} with {self.arguments})"


class PermissionManager:
    """Manages staging and approval for Level 3 / CONFIRM operations."""

    def __init__(self) -> None:
        self._pending: PendingAction | None = None
        self._lock = threading.Lock()

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def pending_action(self) -> PendingAction | None:
        with self._lock:
            return self._pending

    def stage_action(self, tool_name: str, arguments: dict[str, Any], summary: str) -> PendingAction:
        with self._lock:
            action_id = uuid.uuid4().hex[:8]
            self._pending = PendingAction(
                id=action_id,
                tool_name=tool_name,
                arguments=arguments,
                summary=summary,
            )
            log.info("Staged Level 3 action #%s: %s", action_id, summary)
            return self._pending

    def pop_pending(self) -> PendingAction | None:
        with self._lock:
            action = self._pending
            self._pending = None
            return action

    def cancel_pending(self) -> PendingAction | None:
        with self._lock:
            action = self._pending
            if action:
                log.info("Cancelled pending action #%s: %s", action.id, action.summary)
            self._pending = None
            return action


# Global singleton instance
_GLOBAL_PERMISSION_MGR: PermissionManager | None = None


def get_permission_manager() -> PermissionManager:
    global _GLOBAL_PERMISSION_MGR
    if _GLOBAL_PERMISSION_MGR is None:
        _GLOBAL_PERMISSION_MGR = PermissionManager()
    return _GLOBAL_PERMISSION_MGR
