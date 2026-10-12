"""Voice conversation loop: microphone -> speech-to-text -> assistant -> speech.

:class:`VoiceSession` is shared by every front end. It reports what is going
on through ``on_event(kind, payload)`` so the terminal (and later the floating
orb UI) can show it:

    "listening"     payload None          - microphone is open
    "transcribing"  payload None          - turning speech into text
    "heard"         payload str           - what the user said
    "thinking"      payload str           - the request sent to the brain
    "reply"         payload AssistantReply
    "speaking"      payload str           - text being spoken
    "info"          payload str           - minor notice (e.g. "didn't hear anything")
    "error"         payload str           - problem explained in plain words
    "idle"          payload None          - ready for the next request
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from core.assistant import Assistant, AssistantReply
from core.audio import (
    AudioDeviceError,
    Player,
    Recorder,
    describe_quiet_recording,
    play_cue,
)
from core.stt import SpeechToText, STTError
from core.tts import TextToSpeech

log = logging.getLogger("suzy.voice")

EventHandler = Callable[[str, Any], None]
MODEL_WAIT_SECONDS = 600  # first run may be downloading the speech model


class VoiceSession:
    def __init__(
        self,
        assistant: Assistant,
        tts: TextToSpeech,
        stt: SpeechToText | None = None,
        recorder: Recorder | None = None,
        on_event: EventHandler | None = None,
        cues: bool = True,
        cue_player: Player | None = None,
    ) -> None:
        self.assistant = assistant
        self.tts = tts
        self.stt = stt
        self.recorder = recorder
        self.on_event = on_event or (lambda _kind, _payload: None)
        self.cues = cues
        self.cue_player = cue_player or Player()
        self._turn = threading.Lock()  # one request at a time

    @property
    def can_listen(self) -> bool:
        return self.stt is not None and self.recorder is not None

    @property
    def busy(self) -> bool:
        return self._turn.locked()

    def _emit(self, kind: str, payload: Any = None) -> None:
        try:
            self.on_event(kind, payload)
        except Exception:  # noqa: BLE001 - a UI bug must not break the conversation
            log.exception("Event handler failed for %s", kind)

    # ------------------------------------------------------------------ speaking
    def say(self, text: str) -> bool:
        """Speak ``text`` (blocking). Returns True if spoken completely."""
        if not text or not self.tts.settings.enabled:
            return False
        self._emit("speaking", text)
        return self.tts.speak(text)

    def say_async(self, text: str) -> threading.Thread:
        thread = threading.Thread(target=self.say, args=(text,), name="SpeakAsync", daemon=True)
        thread.start()
        return thread

    def stop_speaking(self) -> None:
        self.tts.stop()

    # ------------------------------------------------------------------ turns
    def _respond(self, text: str, speak: bool) -> AssistantReply:
        self._emit("thinking", text)
        reply = self.assistant.process(text)
        self._emit("reply", reply)
        if speak:
            self.say(reply.text)
        return reply

    def handle_text(self, text: str, speak: bool = True) -> AssistantReply:
        """Typed request: answer it (and speak the answer). Never raises."""
        self.stop_speaking()  # typing interrupts whatever Suzy is saying
        with self._turn:
            try:
                return self._respond(text, speak)
            finally:
                self._emit("idle")

    def listen_and_respond(self, wait: bool = False) -> AssistantReply | None:
        """One push-to-talk turn. Returns the reply, or None if nothing usable was heard.

        With ``wait=False`` it returns immediately if another turn is running.
        Never raises (except KeyboardInterrupt).
        """
        if not self.can_listen:
            self._emit("error", "Voice input is turned off.")
            return None
        if not self._turn.acquire(timeout=5 if wait else 0):
            return None
        try:
            return self._listen_turn()
        finally:
            self._turn.release()
            self._emit("idle")

    def _listen_turn(self) -> AssistantReply | None:
        assert self.stt is not None and self.recorder is not None
        if not self.stt.is_ready:
            if self.stt.load_error:
                return self._fail(self.stt.load_error)
            self._emit("info", "Give me a moment - I'm still loading my speech model...")
            if not self.stt.wait_until_loaded(MODEL_WAIT_SECONDS):
                return self._fail(self.stt.load_error or "My speech model is taking too long to load.")

        try:
            if self.cues:
                play_cue("start", self.cue_player)
            self._emit("listening")
            audio = self.recorder.record()
            if self.cues:
                play_cue("stop", self.cue_player)
        except AudioDeviceError as exc:
            return self._fail(str(exc))

        if audio is None:
            peak = getattr(self.recorder, "last_peak", None)
            threshold = getattr(self.recorder, "last_threshold", None)
            if peak is None or threshold is None:
                self._emit("info", "I didn't hear anything.")
            else:
                log.info("Nothing heard: peak level %.4f, threshold %.4f", peak, threshold)
                self._emit("info", describe_quiet_recording(peak, threshold))
            return None

        self._emit("transcribing")
        try:
            transcript = self.stt.transcribe(audio)
        except STTError as exc:
            return self._fail(str(exc))
        log.info("Heard (%s, %.1fs): %r", transcript.language, transcript.seconds, transcript.text)
        if not transcript.text:
            message = "Sorry, I didn't catch that."
            self._emit("info", message)
            self.say(message)
            return None

        self._emit("heard", transcript.text)
        return self._respond(transcript.text, speak=True)

    def _fail(self, message: str) -> None:
        self._emit("error", message)
        self.say(message)
        return None

    # ------------------------------------------------------------------ hotkey
    def hotkey_pressed(self) -> None:
        """Push-to-talk button.

        * Suzy speaking -> she stops and listens instead.
        * Already listening -> stop listening now (don't wait for a pause).
        * Idle -> start listening.
        """
        if self.recorder is not None and self.recorder.is_recording:
            self.recorder.stop()
            return
        if self.tts.is_speaking:
            self.stop_speaking()
        self.listen_and_respond(wait=True)
