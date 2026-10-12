"""Tests for self-improvement: backups, workspace, coding agent, manager, guard, error watcher."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import guard
from core.config import Config
from core.llm import LLMError, LLMErrorKind, LLMResponse, LLMUnavailableError, ToolCall
from core.selfcode import (
    AgentSettings,
    CodeVault,
    DevAgent,
    ErrorWatcher,
    SelfDevManager,
    Workspace,
    WorkspaceError,
    error_fingerprint,
    estimate_tokens,
    fix_request,
    read_tracebacks,
)


# ============================================================================ fixtures
@pytest.fixture()
def project(tmp_path) -> Path:
    """A tiny fake Suzy installation."""
    root = tmp_path / "suzy"
    files = {
        "main.py": "print('hi')\n",
        "guard.py": "# guard\n",
        "config.yaml": "user:\n  name: Sagar\n",
        "requirements.txt": "groq>=1.0  # brain\npsutil>=5\n",
        "tools/__init__.py": "",
        "tools/web.py": "def search():\n    return 'old'\n",
        ".env": "GROQ_API_KEY=secret\n",
        "data/state.json": "{}",
        "logs/suzy.log": "log",
        ".venv/lib.py": "venv",
        "models/model.bin": "weights",
        "tools/__pycache__/web.cpython-313.pyc": "bytecode",
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def ok_tests(_root, _target=None):
    return True, "5 passed"


class ScriptedLLM:
    """Returns scripted responses; each item is an LLMResponse or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools, system_prompt, options=None):
        self.calls.append({"messages": [dict(m) for m in messages], "options": options,
                           "system": system_prompt})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, **arguments):
    return LLMResponse("", [ToolCall(f"id-{name}-{time.monotonic_ns()}", name, arguments)], "fake")


CRICKET = "from tools.registry import ToolResult, tool\n\n@tool(description='Cricket')\ndef cricket():\n    return ToolResult.ok('India won.')\n"


# ============================================================================ guard / backups
def test_iter_code_files_skips_private_and_generated(project):
    assert sorted(guard.iter_code_files(project)) == [
        "config.yaml", "guard.py", "main.py", "requirements.txt", "tools/__init__.py", "tools/web.py"]


def test_snapshot_and_restore_roundtrip(project):
    vault = CodeVault(project)
    snap = vault.snapshot("before: test")
    (project / "tools/web.py").write_text("broken", encoding="utf-8")
    (project / "tools/new.py").write_text("new", encoding="utf-8")
    (project / "main.py").unlink()
    changed = vault.restore(snap)
    assert set(changed) == {"tools/web.py", "tools/new.py", "main.py"}
    assert (project / "tools/web.py").read_text(encoding="utf-8").startswith("def search")
    assert not (project / "tools/new.py").exists() and (project / "main.py").exists()
    assert (project / ".env").read_text(encoding="utf-8") == "GROQ_API_KEY=secret\n"  # never touched
    assert vault.list()[0].label == "before: test"


def test_vault_prunes_old_backups(project):
    vault = CodeVault(project, keep=2)
    for n in range(4):
        vault.snapshot(f"before: {n}")
    assert [s.label for s in vault.list()] == ["before: 2", "before: 3"]


def test_guard_restarts_on_exit_code_3(project, monkeypatch):
    codes = iter([3, 0])
    monkeypatch.setattr(guard.subprocess, "call", lambda *a, **k: next(codes))
    monkeypatch.setattr(guard, "MARKER", project / "data" / "update_marker.json")
    assert guard.run([]) == 0


def test_guard_rolls_back_a_crashing_update(project, monkeypatch):
    vault = CodeVault(project)
    snap = vault.snapshot("before: add cricket")
    (project / "main.py").write_text("raise SystemExit(1)  # broken update", encoding="utf-8")
    marker = project / "data" / "update_marker.json"
    marker.write_text(json.dumps({"snapshot": str(snap.path), "request": "add cricket"}), encoding="utf-8")
    notice = project / "data" / "rollback_notice.json"
    monkeypatch.setattr(guard, "ROOT", project)
    monkeypatch.setattr(guard, "MARKER", marker)
    monkeypatch.setattr(guard, "ROLLBACK_NOTICE", notice)
    codes = iter([1, 0])  # crash after the update, then fine after the rollback
    monkeypatch.setattr(guard.subprocess, "call", lambda *a, **k: next(codes))
    assert guard.run([]) == 0
    assert (project / "main.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert not marker.exists()
    assert json.loads(notice.read_text(encoding="utf-8"))["request"] == "add cricket"


def test_guard_normal_crash_without_update_is_not_rolled_back(project, monkeypatch):
    monkeypatch.setattr(guard, "MARKER", project / "data" / "update_marker.json")
    monkeypatch.setattr(guard.subprocess, "call", lambda *a, **k: 1)
    assert guard.run([]) == 1


def test_guard_gives_up_on_restart_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "MARKER", tmp_path / "none.json")
    monkeypatch.setattr(guard.subprocess, "call", lambda *a, **k: 3)
    assert guard.run([]) == 3


# ============================================================================ workspace
def test_workspace_copy_changes_and_diff(project):
    ws = Workspace(project, project / "data" / "dev_workspace")
    ws.create()
    assert not (ws.path / ".env").exists() and not (ws.path / "models").exists()
    (ws.path / "tools/web.py").write_text("def search():\n    return 'new'\n", encoding="utf-8")
    (ws.path / "tools/cricket.py").write_text(CRICKET, encoding="utf-8")
    (ws.path / "main.py").unlink()
    assert ws.changes() == {"main.py": "deleted", "tools/cricket.py": "added", "tools/web.py": "modified"}
    diff = ws.diff()
    assert "-    return 'old'" in diff and "+    return 'new'" in diff and "b/tools/cricket.py" in diff


@pytest.mark.parametrize("path", ["../outside.py", "/etc/passwd", ".env", "./.env", "data/state.json",
                                  "backups/x.py", ".venv/lib.py", "models/model.bin"])
def test_workspace_blocks_unsafe_paths(project, path):
    ws = Workspace(project, project / "data" / "dev_workspace")
    ws.create()
    with pytest.raises(WorkspaceError):
        ws.resolve(path)


def test_guard_is_readable_but_not_writable(project):
    ws = Workspace(project, project / "data" / "dev_workspace")
    ws.create()
    assert ws.resolve("guard.py").name == "guard.py"
    with pytest.raises(WorkspaceError, match="protected"):
        ws.resolve("guard.py", for_writing=True)


def test_live_change_detection(project):
    ws = Workspace(project, project / "data" / "dev_workspace")
    ws.create()
    assert ws.live_changed_since_copy() == []
    (project / "tools/web.py").write_text("changed by installer", encoding="utf-8")
    assert ws.live_changed_since_copy() == ["tools/web.py"]


# ============================================================================ coding agent
def make_agent(project, llm, runner=ok_tests, **settings):
    ws = Workspace(project, project / "data" / "dev_workspace")
    ws.create()
    sleeps = []
    agent = DevAgent(llm, ws, settings=AgentSettings(**settings), test_runner=runner, sleep=sleeps.append)
    return agent, ws, sleeps


def test_agent_builds_a_feature(project):
    llm = ScriptedLLM(
        call("list_files", folder="tools"),
        call("read_file", path="tools/web.py"),
        call("search_code", pattern="def search"),
        call("write_file", path="tools/cricket.py", content=CRICKET),
        call("run_tests", test_file="tests"),
        call("finish", summary="I can now tell cricket scores. Just ask 'cricket score'."),
    )
    progress = []
    agent, ws, _ = make_agent(project, llm)
    agent.on_progress = progress.append
    outcome = agent.run("Add a cricket score tool")
    assert outcome.success and "cricket" in outcome.summary and outcome.steps == 6
    assert ws.changes() == {"tools/cricket.py": "added"}
    tool_results = [m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert any("tools/web.py" in r and "bytes" in r for r in tool_results)        # list_files
    assert any("1| def search():" in r for r in tool_results)                    # read_file
    assert any("tools/web.py:1: def search():" in r for r in tool_results)       # search_code
    assert "writing tools/cricket.py" in progress and "running the full test suite" in progress
    assert llm.calls[0]["options"] == {"max_tokens": 3000, "reasoning_effort": "medium"}
    assert "groq, psutil" in llm.calls[0]["system"]


def test_agent_cannot_finish_with_failing_tests(project):
    results = iter([(False, "1 failed: test_cricket"), (True, "6 passed")])
    llm = ScriptedLLM(
        call("write_file", path="tools/cricket.py", content=CRICKET),
        call("finish", summary="done"),
        call("replace_in_file", path="tools/cricket.py", old_text="India won.", new_text="India won!"),
        call("finish", summary="Cricket scores added."),
    )
    agent, ws, _ = make_agent(project, llm, runner=lambda root, target: next(results))
    outcome = agent.run("Add cricket")
    assert outcome.success and outcome.summary == "Cricket scores added."
    tool_messages = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert "test suite FAILED" in tool_messages[-1]["content"]
    assert "India won!" in (ws.path / "tools/cricket.py").read_text(encoding="utf-8")


def test_agent_finish_without_changes_is_rejected(project):
    llm = ScriptedLLM(call("finish", summary="nothing"), call("give_up", reason="Needs a paid API."))
    agent, _, _ = make_agent(project, llm)
    outcome = agent.run("Add stock trading")
    assert not outcome.success and outcome.error == "Needs a paid API."
    assert "haven't changed any files" in llm.calls[1]["messages"][-1]["content"]


def test_agent_tool_errors_are_reported_not_raised(project):
    llm = ScriptedLLM(
        call("read_file", path=".env"),
        call("write_file", path="guard.py", content="evil"),
        call("replace_in_file", path="tools/web.py", old_text="missing", new_text="x"),
        call("read_file", path="nope.py"),
        call("search_code", pattern="("),
        call("give_up", reason="stop"),
    )
    agent, ws, _ = make_agent(project, llm)
    agent.run("try bad things")
    results = [m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert "private keys" in results[0] and "protected" in results[1]
    assert "appears 0 times" in results[2] and "does not exist" in results[3] and "invalid regular" in results[4]
    assert (ws.path / "guard.py").read_text(encoding="utf-8") == "# guard\n"


def test_agent_waits_out_rate_limits(project):
    limited = LLMUnavailableError([LLMError("groq", LLMErrorKind.RATE_LIMIT, "429")])
    llm = ScriptedLLM(limited, limited, call("give_up", reason="ok"))
    agent, _, sleeps = make_agent(project, llm, rate_limit_wait=7)
    agent.run("x")
    assert sleeps == [7, 7]


def test_agent_shrinks_too_large_requests(project):
    too_big = LLMUnavailableError([LLMError("groq", LLMErrorKind.BAD_REQUEST, "Request too large for model")])
    llm = ScriptedLLM(too_big, call("give_up", reason="ok"))
    agent, _, _ = make_agent(project, llm)
    agent.run("x")
    assert llm.calls[1]["options"]["max_tokens"] < 3000


def test_agent_gives_up_when_brain_is_down(project):
    down = LLMUnavailableError([LLMError("groq", LLMErrorKind.AUTH, "bad key")])
    agent, _, _ = make_agent(project, ScriptedLLM(down))
    outcome = agent.run("x")
    assert not outcome.success and "stopped responding" in outcome.error


def test_agent_nudges_then_stops_when_it_only_talks(project):
    chat = LLMResponse("I think we should add it.", [], "fake")
    agent, _, _ = make_agent(project, ScriptedLLM(chat, chat, chat))
    outcome = agent.run("x")
    assert not outcome.success and "couldn't work out" in outcome.error


def test_agent_step_limit(project):
    llm = ScriptedLLM(*[call("list_files")] * 3)
    agent, _, _ = make_agent(project, llm, max_steps=3)
    assert "ran out of steps" in agent.run("x").error


def test_compaction_keeps_requests_under_budget(project):
    agent, _, _ = make_agent(project, ScriptedLLM())
    messages = [{"role": "user", "content": "task"}]
    for n in range(30):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{n}", "type": "function",
             "function": {"name": "write_file", "arguments": json.dumps({"path": "a.py", "content": "x" * 3000})}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{n}", "name": "write_file", "content": "y" * 3000})
    agent._compact(messages, 4500)
    total = estimate_tokens(messages) + estimate_tokens(agent.system_prompt) + 900
    assert total < 5500 and messages[0]["content"] == "task"
    # No tool message is left without the assistant message that called it.
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[index - 1]["role"] in ("assistant", "tool")


# ============================================================================ manager
def make_manager(project, llm, notes=None, **config):
    notes = notes if notes is not None else []
    manager = SelfDevManager(Config({"self_improvement": config}), llm, root=project,
                             notify=notes.append, test_runner=ok_tests)
    return manager, notes


def wait_for(manager, status, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.job and manager.job.status == status:
            return
        time.sleep(0.02)
    raise AssertionError(f"job status is {manager.job.status if manager.job else None}, wanted {status}")


def feature_llm():
    return ScriptedLLM(call("write_file", path="tools/cricket.py", content=CRICKET),
                       call("finish", summary="I can tell cricket scores now."))


def test_full_cycle_build_apply_confirm(project):
    manager, notes = make_manager(project, feature_llm())
    assert "background" in manager.start("Add cricket scores")
    assert "still working" in manager.start("another thing").lower() or manager.job.status != "working"
    wait_for(manager, "ready")
    assert "Your update is ready" in notes[-1] and "tools/cricket.py (added)" in notes[-1]
    assert "READY to install" in manager.context_line()
    assert "already have an update ready" in manager.start("another")
    assert not (project / "tools/cricket.py").exists()  # nothing live yet

    assert "restarting" in manager.apply().lower()
    assert (project / "tools/cricket.py").read_text(encoding="utf-8") == CRICKET
    assert manager.restart_pending and (project / "data/update_marker.json").exists()
    assert manager.history()[-1]["action"] == "applied"

    restarted, _ = make_manager(project, ScriptedLLM())
    assert restarted.confirm_healthy_start() == ["My update 'Add cricket scores' is installed and working."]
    assert not (project / "data/update_marker.json").exists()


def test_undo_last_update(project):
    manager, _ = make_manager(project, feature_llm())
    manager.start("Add cricket scores")
    wait_for(manager, "ready")
    manager.apply()
    message = manager.undo_last()
    assert "before 'Add cricket scores'" in message and manager.restart_pending
    assert not (project / "tools/cricket.py").exists()
    assert manager.history()[-1]["action"] == "undone"


def test_undo_with_nothing_to_undo(project):
    manager, _ = make_manager(project, ScriptedLLM())
    assert "nothing to undo" in manager.undo_last()


def test_failed_job_changes_nothing(project):
    manager, notes = make_manager(project, ScriptedLLM(call("give_up", reason="It needs a paid API.")))
    manager.start("Add paid stuff")
    wait_for(manager, "failed")
    assert "It needs a paid API." in notes[-1] and "Nothing was changed" in notes[-1]
    assert "didn't work out" in manager.status()
    assert "no finished update" in manager.apply()


def test_discard_and_cancel(project):
    manager, _ = make_manager(project, feature_llm())
    manager.start("Add cricket")
    wait_for(manager, "ready")
    assert "Discarded" in manager.discard()
    assert manager.discard() == "There's no update waiting to be installed."
    assert "not working" in manager.cancel()


def test_apply_refuses_if_live_code_changed(project):
    manager, _ = make_manager(project, feature_llm())
    manager.start("Add cricket")
    wait_for(manager, "ready")
    (project / "tools/web.py").write_text("updated by the installer", encoding="utf-8")
    assert "was changed while I was working" in manager.apply()
    assert not (project / "tools/cricket.py").exists()


def test_disabled(project):
    manager, _ = make_manager(project, ScriptedLLM(), enabled=False)
    assert "switched off" in manager.start("anything")


def test_crashing_agent_is_contained(project):
    manager, notes = make_manager(project, ScriptedLLM(RuntimeError("SDK bug")))
    manager.start("x")
    wait_for(manager, "failed")
    assert "SDK bug" in notes[-1]


def test_rollback_notice_is_reported(project):
    (project / "data/rollback_notice.json").write_text(json.dumps({"request": "add cricket"}), encoding="utf-8")
    manager, _ = make_manager(project, ScriptedLLM())
    assert "made me crash" in manager.confirm_healthy_start()[0]


# ============================================================================ error watcher
LOG = """2026-10-12 10:00:00,000 | ERROR   | MainThread | suzy.main | Brain check failed for gemini: auth
2026-10-12 10:00:01,000 | ERROR   | MainThread | suzy.tools | Tool open_app crashed
Traceback (most recent call last):
  File "C:\\\\Users\\\\sagar\\\\Suzy\\\\tools\\\\apps.py", line 42, in open_app
    entry = index[0]
IndexError: list index out of range
2026-10-12 10:00:02,000 | ERROR   | MainThread | suzy.tools | Tool open_app crashed
Traceback (most recent call last):
  File "C:\\\\Users\\\\sagar\\\\Suzy\\\\tools\\\\apps.py", line 42, in open_app
    entry = index[0]
IndexError: list index out of range
"""


def test_read_tracebacks_ignores_explained_errors(tmp_path):
    path = tmp_path / "errors.log"
    path.write_text(LOG, encoding="utf-8")
    tracebacks, offset = read_tracebacks(path)
    assert len(tracebacks) == 2 and offset == path.stat().st_size
    assert all("IndexError" in t for t in tracebacks)
    assert error_fingerprint(tracebacks[0]) == error_fingerprint(tracebacks[1])


def test_error_watcher_reports_each_error_once(project):
    (project / "logs/errors.log").write_text(LOG, encoding="utf-8")
    watcher = ErrorWatcher(project)
    first = watcher.new_errors()
    assert len(first) == 1 and "IndexError" in first[0]
    assert watcher.new_errors() == []  # already seen
    assert not watcher.already_tried(first[0])
    watcher.mark_tried(first[0])
    assert watcher.already_tried(first[0])
    assert len(watcher.recent_errors()) == 1
    assert "IndexError" in fix_request(first) and "root cause" in fix_request(first)


def test_error_watcher_handles_rotation_and_missing_log(project):
    watcher = ErrorWatcher(project)
    assert watcher.new_errors() == []
    (project / "data/selfdev_state.json").write_text(json.dumps({"errors_offset": 10**9}), encoding="utf-8")
    (project / "logs/errors.log").write_text(LOG, encoding="utf-8")
    assert len(watcher.new_errors()) == 1  # offset beyond the (rotated) file -> read from the start
