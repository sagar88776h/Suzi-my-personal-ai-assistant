"""Tests for the self-improvement tools, the yes-detection and tool auto-discovery."""

from __future__ import annotations

import pytest

from tools import context, self_dev
from tools.registry import tool_module_names


class FakeManager:
    def __init__(self, job_status=None):
        self.started, self.applied, self.undone = [], 0, 0
        self.job = type("Job", (), {"status": job_status})() if job_status else None

    def start(self, request):
        self.started.append(request)
        return "On it!"

    def status(self):
        return "Working on it."

    def apply(self):
        self.applied += 1
        return "Installed!"

    def undo_last(self):
        self.undone += 1
        return "Undone."

    def cancel(self):
        return "Stopped."

    def discard(self):
        return "Discarded."

    def history(self, limit):
        return [{"time": "2026-10-12 10:00:00", "action": "applied", "request": "Add cricket"}]


class FakeWatcher:
    def __init__(self, errors):
        self.errors = errors

    def recent_errors(self):
        return self.errors


@pytest.fixture()
def manager(monkeypatch):
    fake = FakeManager()
    monkeypatch.setattr(self_dev, "MANAGER", fake)
    monkeypatch.setattr(self_dev, "WATCHER", FakeWatcher(["Traceback (most recent call last):\nIndexError: x"]))
    return fake


@pytest.mark.parametrize("text, expected", [
    ("yes", True), ("Yes, install it", True), ("haan kar do", True), ("ok go ahead", True),
    ("theek hai", True), ("no", False), ("not yet", False), ("nahi, wait", False),
    ("what did you change?", False), ("", False), ("yes... actually no", False),
])
def test_user_confirmed(text, expected):
    context.set_user_message(text)
    assert context.user_confirmed() is expected


def test_improve_myself_and_status(manager):
    assert self_dev.improve_myself("Add cricket scores").message == "On it!"
    assert manager.started == ["Add cricket scores"]
    assert self_dev.my_update_status().message == "Working on it."


def test_apply_requires_a_clear_yes(manager):
    context.set_user_message("what does the update do?")
    result = self_dev.apply_my_update()
    assert not result.success and "clear yes" in result.message and manager.applied == 0
    context.set_user_message("haan, install it")
    assert self_dev.apply_my_update().success and manager.applied == 1


def test_undo_requires_a_clear_yes(manager):
    context.set_user_message("undo")
    assert not self_dev.undo_my_last_update().success
    context.set_user_message("yes undo it")
    assert self_dev.undo_my_last_update().success and manager.undone == 1


def test_fix_my_errors(manager, monkeypatch):
    result = self_dev.fix_my_errors()
    assert result.success and "IndexError" in manager.started[0] and result.data["errors"] == 1
    monkeypatch.setattr(self_dev, "WATCHER", FakeWatcher([]))
    assert "don't have any unexpected errors" in self_dev.fix_my_errors().message


def test_discard_stops_running_job(monkeypatch):
    monkeypatch.setattr(self_dev, "MANAGER", FakeManager("working"))
    assert self_dev.discard_my_update().message == "Stopped."
    monkeypatch.setattr(self_dev, "MANAGER", FakeManager("ready"))
    assert self_dev.discard_my_update().message == "Discarded."


def test_list_my_updates(manager):
    assert "Add cricket" in self_dev.list_my_updates().message


def test_tools_unavailable_without_manager(monkeypatch):
    monkeypatch.setattr(self_dev, "MANAGER", None)
    for func in (lambda: self_dev.improve_myself("x"), self_dev.fix_my_errors, self_dev.my_update_status,
                 self_dev.apply_my_update, self_dev.discard_my_update, self_dev.undo_my_last_update,
                 self_dev.list_my_updates):
        assert not func().success


def test_tool_modules_are_discovered_automatically(registry):
    names = tool_module_names()
    assert {"tools.apps", "tools.system", "tools.web", "tools.productivity", "tools.self_dev"} <= set(names)
    assert not {"tools.registry", "tools.safety", "tools.winutils", "tools.context"} & set(names)
    assert registry.get("improve_myself") is not None and registry.get("apply_my_update").safety.value == "confirm"
