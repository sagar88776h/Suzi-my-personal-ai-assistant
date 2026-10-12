"""Self-improvement: Suzy changes her own code.

How a change happens:

1. Sagar asks for a new ability or a fix. :class:`SelfDevManager` copies all of
   Suzy's code into a private *workspace* (``data/dev_workspace``).
2. :class:`DevAgent` - the LLM with file and test tools - edits the workspace and
   runs the test suite until it passes (or gives up).
3. Suzy describes the change and asks Sagar. Only after a clear "yes":
   :meth:`SelfDevManager.apply` backs up the live code (``backups/``), copies the
   changed files in, and restarts Suzy through ``guard.py``.
4. If the updated Suzy crashes before she confirms a healthy start, the guard
   restores the backup automatically. Sagar can also say "undo your last update".

Nothing is ever changed in the running Suzy until step 3.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import guard
from core.config import Config
from core.llm import LLMErrorKind, LLMUnavailableError
from core.paths import APP_DIR

log = logging.getLogger("suzy.selfcode")

RESTART_EXIT_CODE = guard.RESTART_EXIT_CODE
Notify = Callable[[str], None]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(text: str, length: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:length] or "change"


# =========================================================================== backups
@dataclass
class Snapshot:
    path: Path
    label: str
    created: str


class CodeVault:
    """Full copies of Suzy's code, taken before every change."""

    def __init__(self, root: Path = APP_DIR, keep: int = 10) -> None:
        self.root = root
        self.dir = root / "backups"
        self.keep = keep

    def snapshot(self, label: str) -> Snapshot:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.dir / f"{stamp}-{uuid.uuid4().hex[:4]}-{_slug(label)}"
        files = guard.iter_code_files(self.root)
        for rel, path in files.items():
            destination = target / "files" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        (target / "manifest.json").write_text(json.dumps(
            {"label": label, "created": created, "created_ns": time.time_ns(), "files": sorted(files)},
            indent=2), encoding="utf-8")
        self.prune()
        log.info("Backup %s created (%d files)", target.name, len(files))
        return Snapshot(target, label, created)

    def list(self) -> list[Snapshot]:
        """All backups, oldest first."""
        found: list[tuple[int, Snapshot]] = []
        if self.dir.is_dir():
            for folder in self.dir.iterdir():
                manifest = folder / "manifest.json"
                if manifest.is_file():
                    try:
                        data = json.loads(manifest.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    order = int(data.get("created_ns") or folder.stat().st_mtime_ns)
                    found.append((order, Snapshot(folder, data.get("label", ""), data.get("created", ""))))
        return [snapshot for _order, snapshot in sorted(found, key=lambda item: item[0])]

    def restore(self, snapshot: Snapshot) -> list[str]:
        return guard.restore_snapshot(self.root, snapshot.path)

    def prune(self) -> None:
        snapshots = self.list()
        for old in snapshots[: max(0, len(snapshots) - self.keep)]:
            shutil.rmtree(old.path, ignore_errors=True)


# =========================================================================== workspace
class WorkspaceError(ValueError):
    """A file operation the agent tried that isn't allowed."""


class Workspace:
    """A private copy of Suzy's code where changes are made and tested."""

    def __init__(self, live_root: Path, path: Path) -> None:
        self.live_root = live_root
        self.path = path
        self.baseline: dict[str, str] = {}  # live file digests when the copy was made

    def create(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        for rel, source in guard.iter_code_files(self.live_root).items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.baseline[rel] = file_digest(source)

    def resolve(self, rel: str, for_writing: bool = False) -> Path:
        rel = rel.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = "" if rel == "." else rel
        target = (self.path / rel).resolve()
        if target != self.path.resolve() and self.path.resolve() not in target.parents:
            raise WorkspaceError(f"'{rel}' is outside the project.")
        parts = Path(rel).parts
        if any(part in guard.EXCLUDED_DIRS for part in parts):
            raise WorkspaceError(f"'{rel}' is not part of the code (it's in an excluded folder).")
        if Path(rel).name in guard.EXCLUDED_FILES:
            raise WorkspaceError(f"'{rel}' holds private keys and can't be read or changed.")
        if for_writing and rel in guard.PROTECTED_FILES:
            raise WorkspaceError(f"'{rel}' is protected and can't be changed.")
        return target

    def files(self) -> dict[str, Path]:
        return guard.iter_code_files(self.path)

    def changes(self) -> dict[str, str]:
        """{relative path: "added" | "modified" | "deleted"} compared with the live code."""
        current = {rel: file_digest(path) for rel, path in self.files().items()}
        result: dict[str, str] = {}
        for rel, digest in current.items():
            if rel not in self.baseline:
                result[rel] = "added"
            elif digest != self.baseline[rel]:
                result[rel] = "modified"
        for rel in self.baseline:
            if rel not in current:
                result[rel] = "deleted"
        return dict(sorted(result.items()))

    def diff(self) -> str:
        chunks = []
        for rel, change in self.changes().items():
            old = (self.live_root / rel).read_text(encoding="utf-8", errors="replace").splitlines(True) \
                if change != "added" and (self.live_root / rel).exists() else []
            new = (self.path / rel).read_text(encoding="utf-8", errors="replace").splitlines(True) \
                if change != "deleted" else []
            chunks.extend(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}"))
        return "".join(chunks)

    def live_changed_since_copy(self) -> list[str]:
        """Live files modified after the workspace was created (e.g. by the installer)."""
        live = {rel: file_digest(path) for rel, path in guard.iter_code_files(self.live_root).items()}
        return sorted(rel for rel in set(live) | set(self.baseline) if live.get(rel) != self.baseline.get(rel))


def run_test_suite(root: Path, target: str | None = None, timeout: int = 900) -> tuple[bool, str]:
    """Run pytest in ``root`` with Suzy's own Python. Returns (passed, last part of the output)."""
    command = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]
    if target:
        command.append(target)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("SUZY_GUARDED", None)
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, f"Tests took longer than {timeout} seconds and were stopped."
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return completed.returncode == 0, "\n".join(output[-60:])


# =========================================================================== coding agent
def _schema(name: str, description: str, properties: dict[str, Any] | None = None,
            required: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}


DEV_TOOLS = [
    _schema("list_files", "List the project's code files with their sizes.",
            {"folder": {"type": "string", "description": "Folder to list, e.g. 'tools'. Empty = all."}}),
    _schema("read_file", "Read part of a file with line numbers (max 150 lines per call).",
            {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            ["path"]),
    _schema("search_code", "Find lines matching a regular expression across the project.",
            {"pattern": {"type": "string"}}, ["pattern"]),
    _schema("write_file", "Create a file, or completely replace one, with the given content.",
            {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _schema("replace_in_file", "Replace one exact, unique piece of text in a file.",
            {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            ["path", "old_text", "new_text"]),
    _schema("delete_file", "Delete a file.", {"path": {"type": "string"}}, ["path"]),
    _schema("run_tests", "Run the pytest test suite (optionally one test file) and see the result.",
            {"test_file": {"type": "string", "description": "e.g. tests/test_cricket.py; empty = all"}}),
    _schema("finish", "The change is complete. Runs all tests first; succeeds only if they pass.",
            {"summary": {"type": "string",
                         "description": "2-4 plain sentences for Sagar: what changed and how to use it."}},
            ["summary"]),
    _schema("give_up", "Stop because the task can't be done (explain why in plain words).",
            {"reason": {"type": "string"}}, ["reason"]),
]

AGENT_PROMPT = """You are the developer of Suzy, a Python 3.11+ voice assistant for Windows 10/11. \
You are changing Suzy's OWN source code, in a private copy of the project. Paths are relative to \
the project root.

PROJECT LAYOUT
- main.py: entry point (terminal chat, startup checks). guard.py: rescue launcher (protected, never edit).
- core/: assistant.py (LLM tool loop), llm.py (Groq/Gemini), prompts.py (Suzy's personality), \
voice.py, audio.py (mic/speakers), stt.py (Whisper), tts.py (voice), hotkey.py, config.py \
(DEFAULT_CONFIG), selfcode.py (this self-improvement system).
- tools/: Suzy's abilities. EVERY module in tools/ is loaded automatically - a new file = new abilities.
- tests/: pytest tests (they must never touch the real PC, network, mic or speakers - mock them).
- config.yaml: user settings.

ADDING AN ABILITY (a new file tools/<topic>.py):
    from tools.registry import ToolResult, tool

    @tool(description="What it does, for the AI to decide when to use it.",
          parameters={{"type": "object", "properties": {{"city": {{"type": "string"}}}}, "required": ["city"]}},
          category="<topic>")
    def get_something(city: str) -> ToolResult:
        ...
        return ToolResult.ok("One short spoken sentence.", detail=value)   # or ToolResult.fail("Why, in plain words.")
Installed packages (use nothing else): {packages}. For HTTP use urllib.request (requests is NOT \
installed) with a timeout of 10 s. Only free APIs that need no key. Windows-only calls go through \
tools/winutils.py helpers. Also add tests/test_<topic>.py that mocks network/OS calls.

RULES
- Look at the relevant files first (list_files, search_code, read_file). Make the smallest change \
that fully does the job; keep the existing style and comments.
- Edit with replace_in_file; use write_file for new files. Keep each file under ~250 lines.
- Run the tests for what you changed. When everything works, call finish with a plain summary for \
{user}. If the task needs something impossible (a paid API, admin rights, missing hardware), call \
give_up and explain.
"""


@dataclass
class AgentOutcome:
    success: bool
    summary: str = ""
    error: str = ""
    steps: int = 0


@dataclass
class AgentSettings:
    max_steps: int = 40
    context_tokens: int = 4500       # keep each request small for free-tier token limits
    max_output_tokens: int = 3000
    reasoning_effort: str = "medium"
    rate_limit_wait: float = 20.0    # seconds to wait when the free tier is exhausted
    max_rate_limit_waits: int = 15
    test_timeout: int = 900


def estimate_tokens(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str)) // 3 + 1


class DevAgent:
    """An LLM with file-editing and test tools, working inside a :class:`Workspace`."""

    def __init__(self, llm: Any, workspace: Workspace, user_name: str = "Sagar",
                 settings: AgentSettings | None = None, on_progress: Notify | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 test_runner: Callable[[Path, str | None], tuple[bool, str]] | None = None) -> None:
        self.llm = llm
        self.workspace = workspace
        self.settings = settings or AgentSettings()
        self.on_progress = on_progress or (lambda _text: None)
        self._sleep = sleep
        self._run_tests = test_runner or (lambda root, target: run_test_suite(root, target, self.settings.test_timeout))
        self.system_prompt = AGENT_PROMPT.format(packages=self._packages(), user=user_name)
        self.cancelled = threading.Event()
        self._finished: AgentOutcome | None = None

    def _packages(self) -> str:
        requirements = self.workspace.path / "requirements.txt"
        names = []
        if requirements.exists():
            for line in requirements.read_text(encoding="utf-8").splitlines():
                name = re.split(r"[<>=;\s#]", line.strip(), maxsplit=1)[0]
                if name:
                    names.append(name)
        return ", ".join(names) or "the standard library only"

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, args: dict[str, Any]) -> str:
        ws = self.workspace
        if name == "list_files":
            folder = str(args.get("folder") or "").strip().strip("/")
            lines = [f"{rel}  ({path.stat().st_size} bytes)" for rel, path in sorted(ws.files().items())
                     if not folder or rel.startswith(folder + "/")]
            return "\n".join(lines) or "No files there."
        if name == "read_file":
            path = ws.resolve(str(args["path"]))
            if not path.is_file():
                return f"Error: {args['path']} does not exist."
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, int(args.get("start_line") or 1))
            end = min(len(lines), int(args.get("end_line") or start + 149), start + 149)
            body = "\n".join(f"{n:4d}| {lines[n - 1]}" for n in range(start, end + 1))
            more = f"\n... ({len(lines) - end} more lines)" if end < len(lines) else ""
            self.on_progress(f"reading {args['path']}")
            return f"{args['path']} lines {start}-{end} of {len(lines)}:\n{body}{more}"
        if name == "search_code":
            try:
                pattern = re.compile(str(args["pattern"]))
            except re.error as exc:
                return f"Error: invalid regular expression ({exc})."
            hits = []
            for rel, path in sorted(ws.files().items()):
                if path.suffix not in (".py", ".yaml", ".txt", ".md", ".ini", ".qss"):
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{rel}:{number}: {line.strip()[:160]}")
                        if len(hits) >= 40:
                            return "\n".join(hits) + "\n... (more matches - narrow the pattern)"
            return "\n".join(hits) or "No matches."
        if name == "write_file":
            path = ws.resolve(str(args["path"]), for_writing=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args["content"]), encoding="utf-8")
            self.on_progress(f"writing {args['path']}")
            return f"Wrote {args['path']} ({len(str(args['content']).splitlines())} lines)."
        if name == "replace_in_file":
            path = ws.resolve(str(args["path"]), for_writing=True)
            if not path.is_file():
                return f"Error: {args['path']} does not exist."
            text = path.read_text(encoding="utf-8")
            old = str(args["old_text"])
            count = text.count(old)
            if count != 1:
                return (f"Error: the old_text appears {count} times in {args['path']}; it must appear exactly "
                        "once. Read the file again and copy the exact text (including indentation).")
            path.write_text(text.replace(old, str(args["new_text"])), encoding="utf-8")
            self.on_progress(f"editing {args['path']}")
            return f"Edited {args['path']}."
        if name == "delete_file":
            path = ws.resolve(str(args["path"]), for_writing=True)
            if not path.is_file():
                return f"Error: {args['path']} does not exist."
            path.unlink()
            self.on_progress(f"deleting {args['path']}")
            return f"Deleted {args['path']}."
        if name == "run_tests":
            target = str(args.get("test_file") or "").strip() or None
            if target:
                ws.resolve(target)
            self.on_progress(f"running tests{' for ' + target if target else ''}")
            passed, output = self._run_tests(ws.path, target)
            return ("PASSED\n" if passed else "FAILED\n") + output
        if name == "finish":
            if not self.workspace.changes():
                return "Error: you haven't changed any files yet."
            self.on_progress("running the full test suite")
            passed, output = self._run_tests(ws.path, None)
            if not passed:
                return "Can't finish: the test suite FAILED. Fix it, then call finish again.\n" + output
            self._finished = AgentOutcome(True, summary=str(args.get("summary", "")).strip())
            return "Finished."
        if name == "give_up":
            self._finished = AgentOutcome(False, error=str(args.get("reason", "")).strip())
            return "Stopped."
        return f"Error: unknown tool {name}."

    def run_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            return self._tool(name, args)
        except (WorkspaceError, KeyError, ValueError, OSError) as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------ context
    def _compact(self, messages: list[dict[str, Any]], budget: int) -> None:
        """Shrink older messages so each request fits the free tier's token limit."""
        def total() -> int:
            return estimate_tokens(messages) + estimate_tokens(DEV_TOOLS) + estimate_tokens(self.system_prompt)

        # 1. Shorten old tool results (keep the last three intact).
        tool_indexes = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        for index in tool_indexes[:-3]:
            if total() <= budget:
                return
            content = messages[index]["content"]
            if len(content) > 300:
                messages[index]["content"] = content[:300] + "\n...(older output trimmed)"
        # 2. Drop the contents of old write_file calls.
        for message in messages[1:-4]:
            if total() <= budget:
                return
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                if len(arguments) > 300:
                    try:
                        parsed = json.loads(arguments)
                    except ValueError:
                        parsed = {}
                    parsed = {k: (v if len(str(v)) < 200 else "(omitted - already applied)") for k, v in parsed.items()}
                    call["function"]["arguments"] = json.dumps(parsed)
        # 3. Drop whole old steps (assistant message + its tool results), oldest first.
        while total() > budget and len(messages) > 4:
            end = 2
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            del messages[1:end]

    def _ask_llm(self, messages: list[dict[str, Any]]) -> Any:
        budget = self.settings.context_tokens
        max_output = self.settings.max_output_tokens
        waits = 0
        while True:
            if self.cancelled.is_set():
                raise LLMUnavailableError([])
            self._compact(messages, budget)
            try:
                return self.llm.chat(messages, DEV_TOOLS, self.system_prompt,
                                     options={"max_tokens": max_output,
                                              "reasoning_effort": self.settings.reasoning_effort})
            except LLMUnavailableError as exc:
                details = " ".join(e.detail for e in exc.errors).lower()
                if "too large" in details and budget > 2000:
                    budget = int(budget * 0.7)
                    max_output = max(1500, int(max_output * 0.8))
                    log.info("Request too large; shrinking to %d context / %d output tokens", budget, max_output)
                    continue
                if LLMErrorKind.RATE_LIMIT in exc.kinds() and waits < self.settings.max_rate_limit_waits:
                    waits += 1
                    self.on_progress("waiting a little for my free AI usage limit to reset")
                    self._sleep(self.settings.rate_limit_wait)
                    continue
                raise

    def run(self, task: str) -> AgentOutcome:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        nudges = 0
        for step in range(1, self.settings.max_steps + 1):
            if self.cancelled.is_set():
                return AgentOutcome(False, error="Cancelled.", steps=step)
            try:
                response = self._ask_llm(messages)
            except LLMUnavailableError as exc:
                return AgentOutcome(False, error=f"My AI brain stopped responding ({exc.user_message()})", steps=step)

            if not response.tool_calls:
                messages.append({"role": "assistant", "content": response.text or "(no reply)"})
                nudges += 1
                if nudges > 2:
                    return AgentOutcome(False, error="I couldn't work out how to make that change.", steps=step)
                messages.append({"role": "user", "content": "Use the tools to make the change, then call "
                                 "finish. If it can't be done, call give_up."})
                continue

            assistant_message: dict[str, Any] = {
                "role": "assistant", "content": response.text or None,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                               for c in response.tool_calls],
            }
            assistant_message.update(response.message_extras)
            messages.append(assistant_message)
            for call in response.tool_calls:
                result = self.run_tool(call.name, call.arguments)
                log.debug("Dev tool %s(%s) -> %s", call.name, str(call.arguments)[:200], result[:200])
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
                if self._finished is not None:
                    self._finished.steps = step
                    return self._finished
        return AgentOutcome(False, error=f"I ran out of steps ({self.settings.max_steps}) before finishing.",
                            steps=self.settings.max_steps)


# =========================================================================== manager
@dataclass
class DevJob:
    id: str
    request: str
    status: str = "working"   # working | ready | failed | applied | discarded
    started: float = field(default_factory=time.time)
    summary: str = ""
    error: str = ""
    changes: dict[str, str] = field(default_factory=dict)
    progress: list[str] = field(default_factory=list)


class SelfDevManager:
    """Runs one self-improvement job at a time and installs it on request."""

    def __init__(self, config: Config, llm: Any, root: Path = APP_DIR, notify: Notify | None = None,
                 progress: Notify | None = None, agent_settings: AgentSettings | None = None,
                 test_runner: Callable[[Path, str | None], tuple[bool, str]] | None = None) -> None:
        self.config = config
        self.llm = llm
        self.root = root
        self.data_dir = root / "data"
        self.notify = notify or (lambda _text: None)
        self.progress = progress or (lambda _text: None)
        self.agent_settings = agent_settings or AgentSettings(
            max_steps=int(config.get("self_improvement.max_steps", 40)),
            context_tokens=int(config.get("self_improvement.context_tokens", 4500)),
            max_output_tokens=int(config.get("self_improvement.max_output_tokens", 3000)),
        )
        self.test_runner = test_runner
        self.vault = CodeVault(root, keep=int(config.get("self_improvement.keep_backups", 10)))
        self.workspace = Workspace(root, self.data_dir / "dev_workspace")
        self.job: DevJob | None = None
        self.restart_pending = False
        self._agent: DevAgent | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("self_improvement.enabled", True))

    # ------------------------------------------------------------------ jobs
    def start(self, request: str) -> str:
        request = request.strip()
        if not self.enabled:
            return "Self-improvement is switched off in config.yaml (self_improvement.enabled)."
        if not request:
            return "Tell me what you'd like me to add or fix."
        with self._lock:
            if self.job and self.job.status == "working":
                return f"I'm still working on: {self.job.request}. Let me finish that first."
            if self.job and self.job.status == "ready":
                return (f"I already have an update ready ({self.job.request}). Tell me to install it "
                        "or discard it first.")
            self.job = DevJob(uuid.uuid4().hex[:8], request)
            job = self.job
        self._thread = threading.Thread(target=self._run_job, args=(job,), name="SelfImprove", daemon=True)
        self._thread.start()
        return ("On it! I'm working on that in the background - it can take a few minutes. "
                "I'll tell you when it's ready to try.")

    def _job_progress(self, job: DevJob, text: str) -> None:
        job.progress.append(text)
        self.progress(text)

    def _run_job(self, job: DevJob) -> None:
        try:
            self.workspace.create()
            self._agent = DevAgent(self.llm, self.workspace, str(self.config.get("user.name", "Sagar")),
                                   self.agent_settings, on_progress=lambda t: self._job_progress(job, t),
                                   test_runner=self.test_runner)
            outcome = self._agent.run(job.request)
        except Exception as exc:  # noqa: BLE001 - a failed job must never crash Suzy
            log.exception("Self-improvement job crashed")
            outcome = AgentOutcome(False, error=f"something went wrong while I was working ({exc})")
        with self._lock:
            if job.status != "working":  # discarded meanwhile
                return
            if outcome.success:
                job.status = "ready"
                job.summary = outcome.summary
                job.changes = self.workspace.changes()
                try:
                    (self.data_dir / "pending_update.diff").write_text(self.workspace.diff(), encoding="utf-8")
                except OSError:
                    log.warning("Couldn't save the update's diff")
            else:
                job.status = "failed"
                job.error = outcome.error
        log.info("Self-improvement job %s finished: %s", job.id, job.status)
        if job.status == "ready":
            files = ", ".join(f"{rel} ({kind})" for rel, kind in job.changes.items())
            self.notify(f"Your update is ready: {job.summary} Files: {files}. All tests pass. "
                        "Say 'yes, install it' and I'll restart with it, or 'discard it'.")
        else:
            self.notify(f"I couldn't finish '{job.request}': {job.error} Nothing was changed.")

    def cancel(self) -> str:
        with self._lock:
            if not self.job or self.job.status != "working":
                return "I'm not working on any change right now."
            self.job.status = "discarded"
            if self._agent:
                self._agent.cancelled.set()
        return "Okay, I stopped working on that. Nothing was changed."

    def discard(self) -> str:
        with self._lock:
            if not self.job or self.job.status != "ready":
                return "There's no update waiting to be installed."
            self.job.status = "discarded"
        shutil.rmtree(self.workspace.path, ignore_errors=True)
        return "Discarded. My code stays exactly as it was."

    # ------------------------------------------------------------------ install / undo
    def apply(self) -> str:
        with self._lock:
            job = self.job
            if not job or job.status != "ready":
                return "There's no finished update to install."
            conflicts = self.workspace.live_changed_since_copy()
            if conflicts:
                job.status = "failed"
                return ("My code was changed while I was working (" + ", ".join(conflicts[:5]) +
                        "), so I won't install this update. Ask me again and I'll redo it.")
            snapshot = self.vault.snapshot(f"before: {job.request}")
            for rel, change in job.changes.items():
                live = self.root / rel
                if change == "deleted":
                    if live.exists():
                        live.unlink()
                else:
                    live.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.workspace.path / rel, live)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            marker = self.data_dir / "update_marker.json"
            marker.write_text(json.dumps({"snapshot": str(snapshot.path), "request": job.request,
                                          "applied": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
            self._log_change("applied", job.request, job.summary, job.changes, snapshot)
            job.status = "applied"
            self.restart_pending = True
        shutil.rmtree(self.workspace.path, ignore_errors=True)
        log.info("Installed self-update '%s' (%d files)", job.request, len(job.changes))
        return "Installed! I'm restarting now to load it - give me a few seconds."

    def undo_last(self) -> str:
        snapshots = [s for s in self.vault.list() if s.label.startswith("before:")]
        if not snapshots:
            return "I haven't installed any updates to myself, so there's nothing to undo."
        latest = snapshots[-1]
        safety = self.vault.snapshot(f"undo of: {latest.label[len('before: '):]}")
        changed = self.vault.restore(latest)
        self._log_change("undone", latest.label[len("before: "):], "", {rel: "restored" for rel in changed}, safety)
        self.restart_pending = True
        return f"Done - I've gone back to how I was before '{latest.label[len('before: '):]}'. Restarting now."

    def _log_change(self, action: str, request: str, summary: str, changes: dict[str, str],
                    snapshot: Snapshot) -> None:
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action, "request": request,
                  "summary": summary, "files": changes, "backup": snapshot.path.name}
        with (self.data_dir / "self_changes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def history(self, limit: int = 5) -> list[dict[str, Any]]:
        path = self.data_dir / "self_changes.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        return records[-limit:]

    # ------------------------------------------------------------------ status
    def status(self) -> str:
        job = self.job
        if job is None:
            return "I'm not working on any changes to myself right now."
        if job.status == "working":
            minutes = (time.time() - job.started) / 60
            last = job.progress[-1] if job.progress else "getting started"
            return f"Still working on '{job.request}' ({minutes:.0f} min so far) - right now I'm {last}."
        if job.status == "ready":
            return f"The update '{job.request}' is ready: {job.summary} It's waiting for your yes to install."
        if job.status == "failed":
            return f"My last attempt ('{job.request}') didn't work out: {job.error}"
        return f"My last update ('{job.request}') was {job.status}."

    def context_line(self) -> str:
        """Live status for the assistant's system prompt."""
        job = self.job
        if job and job.status == "working":
            return f"You are currently changing your own code in the background: '{job.request}'."
        if job and job.status == "ready":
            return (f"A self-update is READY to install: '{job.request}' - {job.summary} "
                    f"Call apply_my_update only if {self.config.get('user.name', 'Sagar')} clearly says yes.")
        return ""

    # ------------------------------------------------------------------ startup
    def confirm_healthy_start(self) -> list[str]:
        """Called once Suzy started successfully. Returns messages for the user."""
        messages = []
        marker = self.data_dir / "update_marker.json"
        if marker.exists():
            try:
                request = json.loads(marker.read_text(encoding="utf-8")).get("request", "")
            except (OSError, ValueError):
                request = ""
            marker.unlink(missing_ok=True)
            messages.append(f"My update '{request}' is installed and working.")
        notice = self.data_dir / "rollback_notice.json"
        if notice.exists():
            try:
                request = json.loads(notice.read_text(encoding="utf-8")).get("request", "")
            except (OSError, ValueError):
                request = ""
            notice.unlink(missing_ok=True)
            messages.append(f"My update '{request}' made me crash, so I went back to the previous version. "
                            "Nothing is lost.")
        return messages


# =========================================================================== error detection
_RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| ", re.MULTILINE)


def read_tracebacks(log_path: Path, start_offset: int = 0, max_bytes: int = 400_000) -> tuple[list[str], int]:
    """Unexpected errors (records with a Python traceback) in errors.log after ``start_offset``.

    Returns (tracebacks, new offset). Problems Suzy already explains (no internet,
    bad key, ...) are logged without a traceback, so they are ignored here.
    """
    if not log_path.exists():
        return [], 0
    size = log_path.stat().st_size
    if start_offset > size:  # the log was rotated
        start_offset = 0
    start = max(start_offset, size - max_bytes)
    with log_path.open("rb") as handle:
        handle.seek(start)
        text = handle.read().decode("utf-8", errors="replace")
    starts = [m.start() for m in _RECORD_START.finditer(text)] + [len(text)]
    records = [text[a:b].strip() for a, b in zip(starts, starts[1:], strict=False)]
    return [r for r in records if "Traceback (most recent call last)" in r], size


def error_fingerprint(traceback_text: str) -> str:
    """Identify an error by its final line and the deepest file/function in the traceback."""
    lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    last = re.sub(r"\d+", "N", lines[-1] if lines else "")[:200]
    frames = re.findall(r'File "([^"]+)", line \d+, in (\w+)', traceback_text)
    where = f"{Path(frames[-1][0].replace(chr(92), '/')).name}:{frames[-1][1]}" if frames else ""
    return hashlib.sha1(f"{where}|{last}".encode()).hexdigest()[:12]


class ErrorWatcher:
    """Finds new unexpected errors since Suzy last started, so she can offer to fix herself."""

    def __init__(self, root: Path = APP_DIR) -> None:
        self.log_path = root / "logs" / "errors.log"
        self.state_path = root / "data" / "selfdev_state.json"

    def _state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @staticmethod
    def _unique(tracebacks: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for text in tracebacks:
            unique[error_fingerprint(text)] = text
        return list(unique.values())

    def new_errors(self) -> list[str]:
        """Distinct tracebacks logged since the previous call."""
        state = self._state()
        tracebacks, offset = read_tracebacks(self.log_path, int(state.get("errors_offset", 0)))
        state["errors_offset"] = offset
        self._save(state)
        return self._unique(tracebacks)

    def recent_errors(self, limit: int = 3) -> list[str]:
        tracebacks, _ = read_tracebacks(self.log_path, 0)
        return self._unique(tracebacks)[-limit:]

    def already_tried(self, traceback_text: str) -> bool:
        return error_fingerprint(traceback_text) in self._state().get("auto_fix_attempted", [])

    def mark_tried(self, traceback_text: str) -> None:
        state = self._state()
        attempted = state.setdefault("auto_fix_attempted", [])
        attempted.append(error_fingerprint(traceback_text))
        state["auto_fix_attempted"] = attempted[-50:]
        self._save(state)


def fix_request(tracebacks: list[str]) -> str:
    """Task description asking the agent to fix logged errors."""
    shown = "\n\n".join(text[-1500:] for text in tracebacks[-3:])
    return ("Fix the bug(s) behind these errors from my error log. Find the root cause in the code, fix it, "
            "and add a test that would have caught it. Errors:\n\n" + shown)
