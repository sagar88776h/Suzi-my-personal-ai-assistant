"""Self-improvement tools: Suzy adds abilities to herself and fixes her own bugs.

The heavy lifting lives in core/selfcode.py. These tools are what the
assistant's brain calls. Installing or undoing an update requires that
Sagar's latest message clearly says yes (see tools/context.py), so text from
a web page or file can never make Suzy rewrite herself.
"""

from __future__ import annotations

from typing import Any

from tools import context
from tools.registry import ToolResult, tool
from tools.safety import SafetyLevel

# Set by main.py at startup (None when self-improvement isn't available, e.g. in tests).
MANAGER: Any = None
WATCHER: Any = None


def configure(manager: Any, watcher: Any) -> None:
    global MANAGER, WATCHER
    MANAGER, WATCHER = manager, watcher


def _unavailable() -> ToolResult:
    return ToolResult.fail("Self-improvement isn't available right now.")


@tool(
    description=(
        "Change your OWN code: add a new ability/feature, change how you behave, or fix a problem in "
        "yourself. Use this whenever Sagar asks you to learn something new, add a feature, or fix/improve "
        "yourself. Write a clear, detailed request. It runs in the background for a few minutes; nothing "
        "is installed until Sagar says yes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "request": {"type": "string",
                        "description": "Detailed description of the change, e.g. 'Add a tool that tells the "
                                       "latest cricket scores using a free API, and how to use it.'"},
        },
        "required": ["request"],
    },
    category="self",
)
def improve_myself(request: str) -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    return ToolResult.ok(MANAGER.start(request))


@tool(
    description="Fix your own recent errors: reads your error log and repairs the code that caused the errors.",
    category="self",
)
def fix_my_errors() -> ToolResult:
    if MANAGER is None or WATCHER is None:
        return _unavailable()
    from core.selfcode import fix_request

    errors = WATCHER.recent_errors()
    if not errors:
        return ToolResult.ok("Good news - I don't have any unexpected errors in my log to fix.")
    return ToolResult.ok(MANAGER.start(fix_request(errors)), errors=len(errors))


@tool(description="Say how your current self-improvement work is going.", category="self")
def my_update_status() -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    return ToolResult.ok(MANAGER.status())


@tool(
    description=(
        "Install the finished update to your own code and restart. ONLY call this right after Sagar "
        "clearly said yes to installing it."
    ),
    safety=SafetyLevel.CONFIRM,
    category="self",
)
def apply_my_update() -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    if not context.user_confirmed():
        return ToolResult.fail("I need a clear yes from Sagar before installing. Ask him first.")
    return ToolResult.ok(MANAGER.apply())


@tool(description="Throw away the finished-but-not-installed update, or stop the one in progress.",
      category="self")
def discard_my_update() -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    job = MANAGER.job
    if job is not None and job.status == "working":
        return ToolResult.ok(MANAGER.cancel())
    return ToolResult.ok(MANAGER.discard())


@tool(
    description=(
        "Undo your most recent self-update (go back to your previous code) and restart. ONLY call this "
        "after Sagar clearly said yes."
    ),
    safety=SafetyLevel.CONFIRM,
    category="self",
)
def undo_my_last_update() -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    if not context.user_confirmed():
        return ToolResult.fail("I need a clear yes from Sagar before undoing. Ask him first.")
    return ToolResult.ok(MANAGER.undo_last())


@tool(description="List the recent changes you made to your own code.", category="self")
def list_my_updates() -> ToolResult:
    if MANAGER is None:
        return _unavailable()
    records = MANAGER.history(5)
    if not records:
        return ToolResult.ok("I haven't changed my own code yet.")
    lines = [f"{r['time']}: {r['action']} - {r['request'][:80]}" for r in records]
    return ToolResult.ok(" ".join(lines), changes=records)
