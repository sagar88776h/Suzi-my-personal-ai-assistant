"""Suzy - desktop voice AI assistant for Windows.

Phase 2 entry point: terminal chat where Suzy speaks her replies and listens
with push-to-talk.

Usage:
    python main.py                        chat (type, or press Enter / Ctrl+Alt+S to talk)
    python main.py --once "open notepad"  run a single command and exit
    python main.py --mute                 don't speak replies out loud
    python main.py --no-voice             no microphone / hotkey (typing only)
    python main.py --quiet                hide the tool-call trace
    python main.py --check                only test the API keys / models and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.assistant import Assistant, AssistantReply
from core.audio import Recorder, RecordingSettings, check_devices
from core.config import Config
from core.hotkey import GlobalHotkey
from core.llm import LLMRouter, ProviderStatus
from core.logging_setup import setup_logging
from core.stt import SpeechToText, STTSettings
from core.tts import TextToSpeech, TTSSettings
from core.voice import VoiceSession
from tools import apps
from tools.registry import ToolRegistry, load_all_tools

log = logging.getLogger("suzy.main")

EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}
HELP_TEXT = """How to talk to Suzy:
  type a message        and press Enter
  press Enter           on an empty line to speak (Suzy stops listening when you pause)
  Ctrl+Alt+S            from ANY window: talk / stop listening / interrupt Suzy
  Ctrl+C                while Suzy is talking: make her stop

Commands:
  /help    show this help
  /tools   list everything Suzy can do on your PC
  /mute    turn Suzy's voice off / on
  /reset   forget the conversation so far
  exit     quit Suzy"""


def _configure_console() -> None:
    """Make the Windows console print Unicode (Hinglish, Hindi, accents) safely."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


# =========================================================================== building
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


@dataclass
class VoiceSetup:
    session: VoiceSession
    hotkey: GlobalHotkey | None
    notes: list[str]      # informational lines for the user
    problems: list[str]   # things that don't work


def build_voice(config: Config, assistant: Assistant, ui: TerminalUI,
                mute: bool = False, no_voice: bool = False) -> VoiceSetup:
    """Create TTS, STT, recorder and hotkey. Never raises: problems are reported instead."""
    notes: list[str] = []
    problems: list[str] = []

    tts_settings = TTSSettings.from_config(config)
    if mute:
        tts_settings.enabled = False
    tts = TextToSpeech(tts_settings)

    listen = bool(config.get("voice.enabled", True)) and not no_voice
    devices = check_devices() if (listen or tts_settings.enabled) else None
    if devices is not None:
        problems += devices.problems
        if devices.microphone is None:
            listen = False

    stt = recorder = None
    if listen:
        stt = SpeechToText(STTSettings.from_config(config))
        recorder = Recorder(RecordingSettings(
            max_seconds=float(config.get("voice.max_record_seconds", 15)),
            silence_seconds=float(config.get("voice.silence_seconds", 1.2)),
            no_speech_timeout=float(config.get("voice.no_speech_timeout", 6)),
        ))
        if not stt.is_downloaded:
            notes.append(f"First run: downloading my speech model ('{stt.settings.model}', about 480 MB) "
                         "in the background. You can type meanwhile.")
        stt.load_async()

    session = VoiceSession(assistant, tts, stt, recorder, on_event=ui.on_event,
                           cues=bool(config.get("voice.sound_cues", True)))

    hotkey = None
    if listen:
        combo = str(config.get("voice.hotkey", "ctrl+alt+s"))
        hotkey = GlobalHotkey(combo, session.hotkey_pressed)
        if hotkey.start():
            ui.hotkey_label = "+".join(part.strip().capitalize() for part in combo.split("+"))
        else:
            problems.append(hotkey.error or "The push-to-talk hotkey couldn't be registered.")
            hotkey = None

    if tts_settings.enabled and devices is not None and devices.speaker:
        notes.append(f"Voice: {tts_settings.voice}, speaking on {devices.speaker}.")
    if listen and devices is not None:
        notes.append(f"Listening with: {devices.microphone}.")
    return VoiceSetup(session, hotkey, notes, problems)


# =========================================================================== reporting
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


# =========================================================================== terminal UI
class TerminalUI:
    """Prints voice-session events without garbling the "You: " prompt."""

    PROMPT = "You: "

    def __init__(self, show_actions: bool = True) -> None:
        self.show_actions = show_actions
        self.hotkey_label: str | None = None
        self._lock = threading.Lock()
        self._at_prompt = False  # main thread is waiting in input()
        self._reprompt = False   # an event interrupted the prompt; redraw it when idle

    def _print(self, text: str) -> None:
        with self._lock:
            if self._at_prompt:
                print()  # leave the half-typed prompt line
                self._at_prompt = False
                self._reprompt = True
            print(text, flush=True)

    def prompt(self) -> str:
        with self._lock:
            self._at_prompt = True
            self._reprompt = False
        try:
            return input(self.PROMPT)
        finally:
            with self._lock:
                self._at_prompt = False

    def on_event(self, kind: str, payload: Any) -> None:
        if kind == "listening":
            self._print("Suzy: Listening... (speak now - I'll stop when you pause)")
        elif kind == "transcribing":
            self._print("      (understanding...)")
        elif kind == "heard":
            self._print(f"You (voice): {payload}")
        elif kind == "reply":
            with self._lock:
                if self._at_prompt:
                    print()
                    self._at_prompt = False
                    self._reprompt = True
                print_reply(payload, self.show_actions)
        elif kind == "info":
            self._print(f"      ({payload})")
        elif kind == "error":
            self._print(f"Suzy: {payload}")
        elif kind == "idle":
            with self._lock:
                # A hotkey turn finished while the main thread waits in input(): show the prompt again.
                if self._reprompt:
                    print(self.PROMPT, end="", flush=True)
                    self._reprompt = False
                    self._at_prompt = True


def run_terminal_chat(voice: VoiceSetup, registry: ToolRegistry, ui: TerminalUI) -> None:
    session = voice.session
    assistant = session.assistant
    hint = "type, or press Enter on an empty line to talk" if session.can_listen else "type your message"
    if voice.hotkey is not None:
        hint += f" ({ui.hotkey_label} works from any window)"
    print(f"      ({hint}; /help for commands, 'exit' to quit)\n")

    while True:
        try:
            line = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSuzy: Bye for now!")
            return
        lowered = line.lower()
        try:
            if not line:
                if session.can_listen:
                    session.listen_and_respond(wait=True)
                continue
            if lowered in EXIT_WORDS:
                print("Suzy: Bye for now!")
                session.say("Bye for now!")
                return
            if lowered == "/help":
                print(HELP_TEXT + "\n")
                continue
            if lowered == "/tools":
                for tool in sorted(registry.all(), key=lambda t: (t.category, t.name)):
                    print(f"  [{tool.category}] {tool.name}: {tool.description}")
                print()
                continue
            if lowered == "/mute":
                settings = session.tts.settings
                settings.enabled = not settings.enabled
                if not settings.enabled:
                    session.stop_speaking()
                print(f"Suzy: Voice {'on' if settings.enabled else 'off'}.\n")
                continue
            if lowered == "/reset":
                assistant.reset()
                print("Suzy: Fresh start - I've cleared our conversation.\n")
                continue
            session.handle_text(line)
        except KeyboardInterrupt:
            session.stop_speaking()
            if session.recorder is not None:
                session.recorder.stop()
            print("\nSuzy: Okay, stopped.\n")


# =========================================================================== main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suzy - your desktop AI assistant")
    parser.add_argument("--once", metavar="TEXT", help="run a single command and exit")
    parser.add_argument("--quiet", action="store_true", help="hide the tool-call trace")
    parser.add_argument("--debug", action="store_true", help="show debug logs in the console")
    parser.add_argument("--check", action="store_true", help="test the API keys and models, then exit")
    parser.add_argument("--no-check", action="store_true", help="skip the brain check at startup")
    parser.add_argument("--mute", action="store_true", help="don't speak replies out loud")
    parser.add_argument("--no-voice", action="store_true", help="disable the microphone and hotkey")
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

    ui = TerminalUI(show_actions=not args.quiet)
    one_shot = args.once is not None
    voice = build_voice(config, assistant, ui, mute=args.mute, no_voice=args.no_voice or one_shot)

    if one_shot:
        reply = assistant.process(args.once)
        print_reply(reply, show_actions=not args.quiet)
        voice.session.say(reply.text)
        return 0

    if not args.no_check:
        run_brain_check(router)
    for problem in voice.problems:
        print(f"Suzy: Heads up - {problem}")
    for note in voice.notes:
        print(f"      {note}")

    hello = greeting(config)
    print(f"Suzy: {hello}")
    voice.session.say_async(hello)
    try:
        run_terminal_chat(voice, registry, ui)
    finally:
        if voice.hotkey is not None:
            voice.hotkey.stop()
        voice.session.stop_speaking()
    log.info("Suzy stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
