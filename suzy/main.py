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
    python main.py --mic-test             check the microphone (lists mics, shows your voice level)
    python main.py --voices               listen to Suzy's voice options and pick one
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.assistant import Assistant, AssistantReply
from core.audio import (
    AudioDeviceError,
    Player,
    Recorder,
    RecordingSettings,
    check_devices,
    list_input_devices,
    measure_microphone,
    resolve_input_device,
)
from core.config import Config
from core.hotkey import GlobalHotkey
from core.listener import (
    ListenerRecorder,
    ListenerSettings,
    SegmenterSettings,
    WakeListener,
    WakeWordDetector,
)
from core.llm import LLMRouter, ProviderStatus
from core.logging_setup import setup_logging
from core.paths import APP_DIR
from core.selfcode import RESTART_EXIT_CODE, ErrorWatcher, SelfDevManager, fix_request
from core.stt import (
    HybridSpeechToText,
    SpeechToText,
    STTError,
    STTSettings,
    create_speech_to_text,
)
from core.tts import EdgeSynthesizer, TextToSpeech, TTSSettings
from core.voice import VoiceSession
from tools import apps, self_dev, voice_settings, win32
from tools.registry import ToolRegistry, load_all_tools

log = logging.getLogger("suzy.main")

EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}
HELP_TEXT = """How to talk to Suzy:
  say "Suzy" / "Hi Suzy" / "Hello Suzy"   then your request - or all at once:
                        "Hey Suzy, open Notepad". After she answers you have 8 seconds
                        to ask something else without saying her name.
  "Suzy" or "Suzy stop" while she's talking interrupts her
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
    win32.TRACKER.start()  # remembers the window Sagar uses, so "close this window" never closes Suzy
    router = LLMRouter.from_config(config)
    assistant = Assistant(config, router, registry)

    from core.memory import get_memory_store
    from core.moods import get_mood_instructions

    assistant.context_providers.append(lambda: get_memory_store().get_context_summary())
    assistant.context_providers.append(lambda: get_mood_instructions())

    return assistant, router, registry


@dataclass
class VoiceSetup:
    session: VoiceSession
    hotkey: GlobalHotkey | None
    notes: list[str]      # informational lines for the user
    problems: list[str]   # things that don't work
    listener: Any = None  # WakeListener when the wake word is on


def build_voice(config: Config, assistant: Assistant, ui: TerminalUI,
                mute: bool = False, no_voice: bool = False) -> VoiceSetup:
    """Create TTS, STT, recorder and hotkey. Never raises: problems are reported instead."""
    notes: list[str] = []
    problems: list[str] = []

    tts_settings = TTSSettings.from_config(config)
    if mute:
        tts_settings.enabled = False
    tts = TextToSpeech(tts_settings)
    tts.on_notice = lambda text: ui.on_event("info", text)
    voice_settings.configure(tts, config)

    listen = bool(config.get("voice.enabled", True)) and not no_voice
    devices = check_devices() if (listen or tts_settings.enabled) else None
    if devices is not None:
        problems += devices.problems
        if devices.microphone is None:
            listen = False

    stt = recorder = None
    mic_device = None
    if listen:
        try:
            mic_device = resolve_input_device(config.get("voice.input_device"))
        except AudioDeviceError as exc:
            problems.append(f"{exc} Using the Windows default microphone instead.")
    if listen:
        stt = create_speech_to_text(STTSettings.from_config(config), Config.api_key("groq"))
        recorder = Recorder(RecordingSettings(
            max_seconds=float(config.get("voice.max_record_seconds", 15)),
            silence_seconds=float(config.get("voice.silence_seconds", 1.2)),
            no_speech_timeout=float(config.get("voice.no_speech_timeout", 6)),
            energy_threshold=float(config.get("voice.energy_threshold", 0.006)),
        ), device=mic_device)
        if isinstance(stt, HybridSpeechToText):
            notes.append("Speech recognition: Groq cloud Whisper (fast), with the local model as backup.")
        if not stt.is_downloaded:
            notes.append(f"First run: downloading my backup speech model ('{stt.settings.model}', about "
                         "480 MB) in the background.")
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

    listener = None
    if listen and bool(config.get("voice.wake_word.enabled", True)):
        listener = build_wake_listener(config, session, mic_device)
        start_wake_listener_async(config, listener, session, ui)
        notes.append("Wake word: just say 'Suzy', 'Hi Suzy' or 'Hello Suzy' - or in one go: "
                     "'Hey Suzy, open Notepad'.")

    if tts_settings.enabled and devices is not None and devices.speaker:
        notes.append(f"Voice: {tts_settings.voice}, speaking on {devices.speaker}.")
    if listen and devices is not None:
        mic_name = devices.microphone
        if mic_device is not None:
            mic_name = next((n for i, n, _d in list_input_devices() if i == mic_device), mic_name)
        notes.append(f"Listening with: {mic_name}.")
    return VoiceSetup(session, hotkey, notes, problems, listener)


def build_wake_listener(config: Config, session: VoiceSession, mic_device: int | None) -> WakeListener:
    """The always-on "Suzy" listener: Silero VAD + a tiny Whisper model that checks for her name."""
    wake_stt = SpeechToText(STTSettings(
        model=str(config.get("voice.wake_word.model", "tiny")),
        language="en",               # keeps her name spelled "Suzy" even in Hinglish
        allowed_languages=(),
        beam_size=1,
        initial_prompt="Hey Suzy. Hi Suzy. Hello Suzy.",
        models_dir=str(config.get("stt.models_dir", "models")),
    ))
    settings = ListenerSettings(
        command_timeout=float(config.get("voice.wake_word.command_timeout", 6)),
        followup_seconds=float(config.get("voice.wake_word.followup_seconds", 8)),
        segment=SegmenterSettings(
            threshold=float(config.get("voice.wake_word.vad_threshold", 0.5)),
            end_silence=float(config.get("voice.wake_word.end_silence", 0.7)),
            max_seconds=float(config.get("voice.max_record_seconds", 15)),
        ),
    )
    return WakeListener(WakeWordDetector(wake_stt), on_event=session.on_listener_event,
                        is_speaking=lambda: session.tts.is_speaking,
                        spoken_text=lambda: session.current_speech,
                        settings=settings, device=mic_device)


def start_wake_listener_async(config: Config, listener: WakeListener, session: VoiceSession,
                              ui: TerminalUI) -> threading.Thread:
    """Load the wake-word model (downloads ~75 MB the first time), then start listening."""

    def run() -> None:
        try:
            listener.detector.stt.load()
        except STTError as exc:
            ui.on_event("error", f"The wake word isn't available ({exc}). Press Enter or "
                                 "Ctrl+Alt+S to talk instead.")
            return
        listener.start()
        session.listener = listener
        session.recorder = ListenerRecorder(listener, float(config.get("voice.no_speech_timeout", 6)))
        ui.on_event("info", "Wake word ready - say 'Suzy' any time.")

    thread = threading.Thread(target=run, name="WakeWordStartup", daemon=True)
    thread.start()
    return thread


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


def greeting(config: Config, now: datetime | None = None, startup: bool = False) -> str:
    name = config.get("user.name", "Sagar")
    if startup:  # started together with Windows
        template = str(config.get("assistant.startup_greeting", "Hello {name}, welcome back."))
        try:
            return template.format(name=name)
        except (KeyError, IndexError, ValueError):
            return template
    hour = (now or datetime.now()).hour
    part = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening"
    return f"Good {part}, {name}. Suzy here - what can I do for you?"


def wait_for_internet(timeout: float = 20.0, host: str = "speech.platform.bing.com") -> bool:
    """Right after Windows starts, Wi-Fi may still be connecting: wait (briefly) for it so the
    welcome is spoken in Suzy's natural voice instead of the offline one."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, 443), timeout=2):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)


def greet_after_startup(session: VoiceSession, text: str, timeout: float = 20.0) -> threading.Thread:
    def run() -> None:
        if not wait_for_internet(timeout):
            log.info("No internet %.0fs after startup; greeting with the offline voice", timeout)
        session.say(text)

    thread = threading.Thread(target=run, name="StartupGreeting", daemon=True)
    thread.start()
    return thread


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
        self.after_idle: Any = None  # called after every finished turn (used to restart after self-updates)

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
        elif kind == "announce":
            self._print(f"Suzy: {payload}")
        elif kind == "progress":
            self._print(f"      (working on myself: {payload})")
        elif kind == "idle":
            if self.after_idle is not None:
                self.after_idle()
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


# =========================================================================== mic test
def _meter(level: float, width: int = 40) -> str:
    filled = min(width, int(level / 0.1 * width))  # 0.1 RMS = full bar (loud speech)
    return "#" * filled + "-" * (width - filled)


def run_mic_test(config: Config, seconds: float = 5.0) -> int:
    """Interactive microphone check: lists mics, shows a live level meter, gives advice."""
    print("Suzy microphone test\n")
    try:
        mics = list_input_devices()
    except AudioDeviceError as exc:
        print(f"  {exc}")
        return 1
    if not mics:
        print("  No microphones found at all. Plug one in, or check Windows Sound settings.")
        return 1
    print("  Microphones on this PC:")
    for index, name, is_default in mics:
        print(f"    [{index}] {name}{'   <- Windows default' if is_default else ''}")

    spec = config.get("voice.input_device")
    try:
        device = resolve_input_device(spec)
    except AudioDeviceError as exc:
        print(f"\n  {exc}")
        device = None
    chosen = next((n for i, n, d in mics if (i == device if device is not None else d)), "default")
    print(f"\n  Testing: {chosen}")
    print(f"  Speak normally for {seconds:.0f} seconds, e.g. \"Hey Suzy, open Notepad\"...\n")

    def show(level: float) -> None:
        print(f"\r  level [{_meter(level)}] {level:.3f} ", end="", flush=True)

    try:
        measurement = measure_microphone(seconds, device, show)
    except AudioDeviceError as exc:
        print(f"\n\n  {exc}")
        return 1
    peak, average = measurement.peak, measurement.average
    threshold = float(config.get("voice.energy_threshold", 0.006))
    print(f"\n\n  Loudest: {peak:.3f}   average: {average:.3f}   Suzy needs speech above: {threshold:.3f}")
    if peak < 0.0005:
        print("  RESULT: pure silence. Windows is probably blocking the microphone, or it's muted.")
        print("    1. Settings > Privacy & security > Microphone: turn on 'Microphone access' AND")
        print("       'Let desktop apps access your microphone'.")
        print("    2. Settings > System > Sound > Input: pick your mic and check it isn't muted.")
        print("    3. Or choose another mic from the list above: set voice.input_device in config.yaml")
        print("       to its number or part of its name, then run this test again.")
        return 1
    if peak < threshold * 1.5:
        suggested = max(0.001, round(peak / 2, 4))
        print("  RESULT: the mic works but your voice is very quiet for me.")
        print("    Raise the mic volume: Settings > System > Sound > (your mic) > Input volume to 80-100,")
        print(f"    or set voice.energy_threshold: {suggested} in config.yaml.")
        return 1
    print("  RESULT: your microphone sounds good - Suzy should hear you fine.")
    return _transcription_check(config, measurement.audio)


def _transcription_check(config: Config, audio: Any) -> int:
    """Run the mic-test recording through Whisper so the whole listening chain is verified."""
    stt = SpeechToText(STTSettings.from_config(config))
    if not stt.is_downloaded:
        print("\n  (Speech model not downloaded yet - start Suzy once to download it, then re-run this test\n"
              "   to also check speech recognition.)")
        return 0
    print("\n  Checking speech recognition on your recording...")
    try:
        transcript = stt.transcribe(audio)
    except STTError as exc:
        print(f"  Speech recognition FAILED: {exc}")
        return 1
    if transcript.text:
        print(f"  I heard: \"{transcript.text}\"  ({transcript.language}, {transcript.seconds:.1f}s)")
        print("  Everything works. If that's what you said, Suzy is ready to listen!")
    else:
        print("  The mic level is fine but I couldn't make out words - speak clearly during the test.")
    return 0


# =========================================================================== voice preview
VOICE_SAMPLE = ("Hi {name}! Main Suzy hoon. I can open your apps, play music, and keep you on time. "
                "Bataiye, aaj main aapki kya madad karoon?")
PREFERRED_LOCALES = ("en-IN", "hi-IN")


def fetch_voice_choices() -> list[dict[str, str]]:
    """Female Indian voices first, then the most natural multilingual ones."""
    import asyncio

    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    female = [v for v in voices if v.get("Gender") == "Female"]
    indian = [v for v in female if v.get("Locale") in PREFERRED_LOCALES]
    multilingual = [v for v in female if "Multilingual" in v.get("ShortName", "") and v not in indian]
    return [{"name": v["ShortName"], "locale": v.get("Locale", "")} for v in indian + multilingual[:6]]


def run_voice_preview(config: Config) -> int:
    """Play a sample of each voice, then save the one Sagar picks."""
    print("Suzy voice preview - getting the list of voices...\n")
    try:
        choices = fetch_voice_choices()
    except Exception as exc:  # noqa: BLE001 - offline, service change, ...
        print(f"  I couldn't get the voice list ({exc}). Check your internet connection and try again.")
        return 1
    if not choices:
        print("  No voices found.")
        return 1
    sample = VOICE_SAMPLE.format(name=config.get("user.name", "Sagar"))
    current = config.get("tts.voice")
    player = Player()
    print("  Listen to each one (press Ctrl+C to skip to the next):\n")
    for number, choice in enumerate(choices, 1):
        marker = "   <- current" if choice["name"] == current else ""
        print(f"  [{number}] {choice['name']}  ({choice['locale']}){marker}")
        try:
            samples, rate = EdgeSynthesizer(choice["name"]).synthesize(sample)
            player.reset()
            player.play(samples, rate)
        except KeyboardInterrupt:
            player.stop()
        except Exception as exc:  # noqa: BLE001 - one broken voice shouldn't stop the preview
            print(f"      (couldn't play this one: {exc})")
    print()
    try:
        answer = input("  Type the number of the voice you liked best (Enter = keep the current one): ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if not answer:
        print("  Keeping the current voice.")
        return 0
    if not answer.isdigit() or not 1 <= int(answer) <= len(choices):
        print("  That's not one of the numbers - nothing changed.")
        return 1
    chosen = choices[int(answer) - 1]["name"]
    config.persist("tts.voice", chosen)
    print(f"  Saved! Suzy will use {chosen} from her next start.")
    return 0


# =========================================================================== self-improvement
def restart_for_update(session: VoiceSession) -> None:
    """Exit so guard.py restarts Suzy with her freshly installed code."""
    session.stop_speaking()
    if os.environ.get("SUZY_GUARDED") == "1":
        print("\nSuzy: Restarting to load my update...\n", flush=True)
    else:
        print("\nSuzy: My update is installed. Please start me again (double-click 'Suzy') to load it.",
              flush=True)
    logging.shutdown()
    os._exit(RESTART_EXIT_CODE)


def setup_self_improvement(config: Config, router: LLMRouter, assistant: Assistant,
                           session: VoiceSession, ui: TerminalUI, root: Path = APP_DIR) -> SelfDevManager:
    """Connect the self-improvement system and report on the last update / new errors."""

    def announce(text: str) -> None:
        ui.on_event("announce", text)
        session.say_async(text)

    manager = SelfDevManager(config, router, root=root, notify=announce,
                             progress=lambda text: ui.on_event("progress", text))
    watcher = ErrorWatcher(root)
    self_dev.configure(manager, watcher)
    assistant.context_providers.append(manager.context_line)
    ui.after_idle = lambda: restart_for_update(session) if manager.restart_pending else None

    for message in manager.confirm_healthy_start():
        print(f"Suzy: {message}")
    errors = watcher.new_errors()
    if errors and manager.enabled:
        latest = errors[-1].splitlines()[-1].strip()
        if config.get("self_improvement.auto_fix_errors", True) and not watcher.already_tried(errors[-1]):
            watcher.mark_tried(errors[-1])
            manager.start(fix_request(errors))
            print(f"Suzy: I found an error in myself since last time ({latest[:120]}). "
                  "I'm fixing it in the background and will ask you before installing the fix.")
        else:
            print(f"Suzy: Heads up - I hit {len(errors)} unexpected error(s) last time "
                  f"({latest[:120]}). Say 'fix your errors' and I'll repair myself.")
    return manager


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
    parser.add_argument("--mic-test", action="store_true", help="test the microphone and exit")
    parser.add_argument("--voices", action="store_true", help="listen to the voice options and pick one")
    parser.add_argument("--startup", action="store_true", help="started with Windows: say 'welcome back'")
    parser.add_argument("--ui", action="store_true", help="launch the Siri-level web & voice interface in your browser")
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

    if args.mic_test:
        return run_mic_test(config)
    if args.voices:
        return run_voice_preview(config)

    try:
        assistant, router, registry = build_assistant(config)
    except Exception:  # noqa: BLE001 - report instead of crashing
        log.exception("Startup failed")
        print("Suzy: Something went wrong while starting up. Details are in logs/errors.log.")
        return 1

    if args.ui:
        import webbrowser
        from web_server import SuzyWebServer
        port = int(config.get("server.port", 8000))
        server = SuzyWebServer(assistant, router, registry, config, port=port)
        print(f"Suzy: Opening Siri interface at http://127.0.0.1:{port}")
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
        server.run()
        return 0

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

    hello = greeting(config, startup=args.startup)
    if args.startup:  # greet first: Sagar just turned the PC on
        print(f"Suzy: {hello}")
        greet_after_startup(voice.session, hello)

    if not args.no_check:
        run_brain_check(router)
    for problem in voice.problems:
        print(f"Suzy: Heads up - {problem}")
    for note in voice.notes:
        print(f"      {note}")

    if not args.startup:
        print(f"Suzy: {hello}")
        voice.session.say_async(hello)
    setup_self_improvement(config, router, assistant, voice.session, ui)
    try:
        run_terminal_chat(voice, registry, ui)
    finally:
        if voice.hotkey is not None:
            voice.hotkey.stop()
        if voice.listener is not None:
            voice.listener.stop()
        voice.session.stop_speaking()
    log.info("Suzy stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
