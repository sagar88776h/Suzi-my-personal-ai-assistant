"""Suzy - desktop voice AI assistant for Windows.

Phase 1 entry point: a text chat in the terminal.

Usage:
    python main.py                     interactive chat
    python main.py --once "open notepad"   run a single command and exit
    python main.py --quiet             hide the tool-call trace
    python main.py --check             only test the API keys / models and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from core.assistant import Assistant, AssistantReply
from core.config import Config
from core.llm import LLMRouter, ProviderStatus
from core.logging_setup import setup_logging
from tools import apps
from tools.registry import ToolRegistry, load_all_tools

log = logging.getLogger("suzy.main")

EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}
HELP_TEXT = """Commands:
  /help    show this help
  /tools   list everything Suzy can do on your PC
  /reset   forget the conversation so far
  exit     quit Suzy
Anything else is sent to Suzy, e.g. "open notepad", "set volume to 30", "what time is it"."""


def _configure_console() -> None:
    """Make the Windows console print Unicode (Hinglish, accents, ...) safely."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def build_assistant(config: Config) -> tuple[Assistant, LLMRouter, ToolRegistry]:
    registry = load_all_tools()
    apps.configure(
        fuzzy_threshold=float(config.get("tools.apps.fuzzy_threshold", 80)),
        suggest_threshold=float(config.get("tools.apps.suggest_threshold", 60)),
        extra_search_dirs=config.get("tools.apps.extra_search_dirs", []) or [],
    )
    apps.prewarm_index()
    router = LLMRouter.from_config(config)
    return Assistant(config, router, registry), router, registry


def startup_report(config: Config, router: LLMRouter) -> list[str]:
    """Plain-language list of problems found at startup (empty = all good)."""
    problems = list(config.load_errors)
    configured = router.configured_providers
    all_names = [p.name for p in router.providers]
    if not configured:
        problems.append("No API key found. Add GROQ_API_KEY or GEMINI_API_KEY to the .env file, "
                        "otherwise I can't think!")
    else:
        for name in all_names:
            if name not in configured:
                problems.append(f"No {name.title()} API key, so I have no {name.title()} backup brain.")
    return problems


def run_brain_check(router: LLMRouter) -> list[ProviderStatus]:
    """Test every configured brain and print a plain-language status line for each."""
    if not router.configured_providers:
        return []  # startup_report already explains that no key is set
    print("Suzy: Checking my brain connections...")
    statuses = router.check()
    for status in statuses:
        print(f"      {status.provider.title():<7}: {status.describe()}")
        if not status.ok and status.error is not None:
            log.error("Brain check failed for %s: %s", status.provider, status.error)
    if statuses and not any(s.ok for s in statuses):
        print("Suzy: Neither brain is working, so I can't answer yet. Technical details:")
        for status in statuses:
            if status.error is not None:
                print(f"      {status.provider}: {' '.join(status.error.detail.split())[:220]}")
    print()
    return statuses


def greeting(config: Config, now: datetime | None = None) -> str:
    hour = (now or datetime.now()).hour
    part = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening"
    return f"Good {part}, {config.get('user.name', 'Sagar')}. Suzy here - what can I do for you?"


def print_reply(reply: AssistantReply, show_actions: bool) -> None:
    if show_actions:
        for action in reply.actions:
            status = "ok" if action.result.success else "failed"
            print(f"   -> {action.tool}({action.arguments}) [{status}]")
        if reply.error and reply.detail:
            print(f"   (details: {reply.detail})")
    print(f"Suzy: {reply.text}\n")


def run_terminal_chat(assistant: Assistant, registry: ToolRegistry, show_actions: bool) -> None:
    print(f"Suzy: {greeting(assistant.config)}")
    print("      (type /help for commands, 'exit' to quit)\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSuzy: Bye for now!")
            return
        if not line:
            continue
        lowered = line.lower()
        if lowered in EXIT_WORDS:
            print("Suzy: Bye for now!")
            return
        if lowered == "/help":
            print(HELP_TEXT + "\n")
            continue
        if lowered == "/tools":
            for tool in sorted(registry.all(), key=lambda t: (t.category, t.name)):
                print(f"  [{tool.category}] {tool.name}: {tool.description}")
            print()
            continue
        if lowered == "/reset":
            assistant.reset()
            print("Suzy: Fresh start - I've cleared our conversation.\n")
            continue
        try:
            reply = assistant.process(line)
        except KeyboardInterrupt:
            print("\nSuzy: Okay, stopped.\n")
            continue
        print_reply(reply, show_actions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suzy - your desktop AI assistant")
    parser.add_argument("--once", metavar="TEXT", help="run a single command and exit")
    parser.add_argument("--quiet", action="store_true", help="hide the tool-call trace")
    parser.add_argument("--debug", action="store_true", help="show debug logs in the console")
    parser.add_argument("--check", action="store_true", help="test the API keys and models, then exit")
    parser.add_argument("--no-check", action="store_true", help="skip the brain check at startup")
    args = parser.parse_args(argv)

    _configure_console()
    config = Config.load()
    setup_logging(
        level=str(config.get("logging.level", "DEBUG")),
        max_bytes=int(config.get("logging.max_bytes", 2_000_000)),
        backup_count=int(config.get("logging.backup_count", 5)),
        console_level="DEBUG" if args.debug else "CRITICAL",  # friendly messages cover the rest
    )
    log.info("Suzy starting")

    try:
        assistant, router, registry = build_assistant(config)
    except Exception:  # noqa: BLE001 - report instead of crashing
        log.exception("Startup failed")
        print("Suzy: Something went wrong while starting up. Details are in logs/errors.log.")
        return 1

    for problem in startup_report(config, router):
        print(f"Suzy: Heads up - {problem}")

    if args.check:
        statuses = run_brain_check(router)
        return 0 if any(s.ok for s in statuses) else 1
    if not args.no_check and not args.once:
        run_brain_check(router)

    if args.once:
        print_reply(assistant.process(args.once), show_actions=not args.quiet)
        return 0

    run_terminal_chat(assistant, registry, show_actions=not args.quiet)
    log.info("Suzy stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
