"""Unit tests for 3-level permission system."""

from __future__ import annotations

from core.permissions import ExecutionLevel, PermissionManager
from tools.safety import SafetyLevel


def test_permission_manager_stage_and_pop():
    mgr = PermissionManager()
    assert not mgr.has_pending

    action = mgr.stage_action("send_email", {"to": "boss@co.com"}, "Send email to boss")
    assert mgr.has_pending
    assert action.id
    assert action.level == ExecutionLevel.LEVEL_3_EXECUTE

    popped = mgr.pop_pending()
    assert popped.id == action.id
    assert not mgr.has_pending


def test_permission_manager_cancel():
    mgr = PermissionManager()
    mgr.stage_action("delete_file", {"path": "doc.txt"}, "Delete document")
    assert mgr.has_pending

    cancelled = mgr.cancel_pending()
    assert cancelled is not None
    assert not mgr.has_pending
