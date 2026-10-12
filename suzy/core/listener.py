"""Always-on listening: voice activity detection and the "Suzy" wake word.

Pipeline (all offline, on a background thread):

    microphone -> VAD (Silero) -> SpeechSegmenter -> one utterance at a time
        -> WakeWordDetector (tiny Whisper, ~0.3 s) - did they say "Suzy"?

The wake word is matched in text, so any of these work: "Suzy", "Hi Suzy",
"Hello Suzy", "Hey Suzy", even with the request in the same breath:
"Hey Suzy, open Notepad". After Suzy answers, :meth:`WakeListener.start_followup`
accepts one more request without the wake word for a few seconds.

Saying "Suzy" while she is talking interrupts her. Her own voice coming back
through the speakers is ignored unless it really contains her name.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.audio import SAMPLE_RATE, AudioDeviceError, _sd, rms_level
from core.stt import SpeechToText, STTError

log = logging.getLogger("suzy.listener")

FRAME = 512                       # samples per VAD frame (Silero's size)
FRAME_SECONDS = FRAME / SAMPLE_RATE  # 32 ms
EventHandler = Callable[[str, Any], None]


# =========================================================================== VAD
class SileroVAD:
    """Silero voice-activity model (bundled with faster-whisper), run frame by frame."""

    name = "silero"

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Probability (0-1) that this 32 ms frame contains speech."""
        x = np.concatenate([self._context, frame.astype(np.float32)]).reshape(1, -1)
        out, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-64:].astype(np.float32)
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Fallback when Silero can't load: loudness relative to a slowly adapting noise floor."""

    name = "energy"

    def __init__(self, min_level: float = 0.01, multiplier: float = 3.0) -> None:
        self.min_level = min_level
        self.multiplier = multiplier
        self.noise_floor = min_level / multiplier

    def reset(self) -> None:
        self.noise_floor = self.min_level / self.multiplier

    def __call__(self, frame: np.ndarray) -> float:
        level = rms_level(frame)
        threshold = max(self.min_level, self.noise_floor * self.multiplier)
        if level < threshold:  # only quiet frames teach the noise floor
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * level
        return 1.0 if level >= threshold else 0.0


def create_vad() -> Any:
    try:
        return SileroVAD()
    except Exception as exc:  # noqa: BLE001 - missing onnxruntime / model file
        log.warning("Silero VAD unavailable (%s); using the loudness-based detector", exc)
        return EnergyVAD()


# =========================================================================== segmenting
@dataclass
class SegmenterSettings:
    threshold: float = 0.5        # VAD probability that counts as speech
    start_frames: int = 3         # ~0.1 s of speech starts an utterance
    end_silence: float = 0.7      # pause that ends it
    pre_roll: float = 0.3         # audio kept from just before speech started
    max_seconds: float = 12.0
    min_speech: float = 0.25      # shorter utterances are treated as noise


class SpeechSegmenter:
    """Turns a stream of frames + speech probabilities into whole utterances."""

    def __init__(self, settings: SegmenterSettings | None = None) -> None:
        self.settings = settings or SegmenterSettings()
        self.max_seconds = self.settings.max_seconds
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, int(self.settings.pre_roll / FRAME_SECONDS)))
        self.last_started = 0.0  # when the most recently returned utterance began
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._pre.clear()
        self._run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self.active = False
        self.started_at = 0.0

    def feed(self, frame: np.ndarray, probability: float, now: float | None = None) -> np.ndarray | None:
        """Add one frame. Returns a finished utterance, or None."""
        now = time.monotonic() if now is None else now
        is_speech = probability >= self.settings.threshold
        if not self.active:
            self._pre.append(frame)
            self._run = self._run + 1 if is_speech else 0
            if self._run >= self.settings.start_frames:
                self.active = True
                self._frames = list(self._pre)
                self._speech_frames = self._run
                self._silence_frames = 0
                # When speech actually began (not the pre-roll before it).
                self.started_at = now - self._run * FRAME_SECONDS
            return None

        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
        ended = self._silence_frames * FRAME_SECONDS >= self.settings.end_silence
        too_long = len(self._frames) * FRAME_SECONDS >= self.max_seconds
        return self.flush() if (ended or too_long) else None

    def flush(self) -> np.ndarray | None:
        """End the current utterance now (e.g. the user pressed stop)."""
        if not self.active:
            return None
        frames, speech = self._frames, self._speech_frames
        started = self.started_at
        self.reset()
        self.last_started = started
        if speech * FRAME_SECONDS < self.settings.min_speech:
            return None
        return np.concatenate(frames).astype(np.float32)


# =========================================================================== wake word
WAKE_NAMES = ("suzy", "suzie", "susie", "susy", "suzi", "susi", "suzzy", "soozy", "suzee", "susee", "soosie")
WAKE_NAMES_HINDI = ("सूज़ी", "सुज़ी", "सूजी", "सुजी", "सूसी", "सुसी")
_GREETING = r"(?:\b(?:hey|hi+|hello|helo|hallo|ok|okay|oye|arre|yo)\b[\s,!.]*)?"
WAKE_RE = re.compile(
    rf"{_GREETING}(?:\b(?:{'|'.join(WAKE_NAMES)})\b|(?:{'|'.join(WAKE_NAMES_HINDI)}))",
    re.IGNORECASE,
)
FILLER_WORDS = {"please", "ji", "yaar", "there", "um", "uh", "hmm"}


def find_wake(text: str) -> re.Match[str] | None:
    return WAKE_RE.search(text or "")


def strip_wake(text: str) -> str:
    """Text after the wake phrase: "Hey Suzy, open Notepad." -> "open Notepad"."""
    match = find_wake(text)
    rest = text[match.end():] if match else text
    return rest.strip(" \t\n,.!?-:;।")


def has_request(text: str) -> bool:
    words = [w.lower() for w in re.findall(r"\w+", text)]
    return any(w not in FILLER_WORDS for w in words)


@dataclass
class WakeResult:
    found: bool
    text: str = ""       # everything the detector heard
    request: str = ""    # words after the wake phrase ("" if they only said the name)


class WakeWordDetector:
    """Listens for Suzy's name using a tiny, fast Whisper model (forced to English
    and primed with "Hey Suzy" so her name is spelled consistently)."""

    def __init__(self, stt: SpeechToText) -> None:
        self.stt = stt

    def detect(self, audio: np.ndarray) -> WakeResult:
        try:
            text = self.stt.transcribe(audio).text
        except STTError as exc:
            log.warning("Wake word check failed: %s", exc)
            return WakeResult(False)
        if not find_wake(text):
            if text:
                log.debug("Not for me: %r", text)
            return WakeResult(False, text)
        request = strip_wake(text)
        log.info("Wake word heard: %r", text)
        return WakeResult(True, text, request if has_request(request) else "")


# =========================================================================== listener
class Mode(Enum):
    WAKE = "wake"           # waiting for "Suzy"
    COMMAND = "command"     # she heard her name, waiting for the request
    FOLLOWUP = "followup"   # just answered: the next utterance needs no wake word
    CAPTURE = "capture"     # push-to-talk (Enter / hotkey) is recording


@dataclass
class VoiceCommand:
    audio: np.ndarray
    via: str              # "wake" | "followup"
    wake_text: str = ""   # what the wake detector heard (fallback if re-transcription loses it)


@dataclass
class ListenerSettings:
    command_timeout: float = 6.0     # after "Suzy", how long to wait for the request
    followup_seconds: float = 8.0    # 0 disables follow-up mode
    speaking_window: float = 3.0     # while Suzy talks, check for her name every few seconds
    segment: SegmenterSettings | None = None


class WakeListener:
    """Owns the microphone on a background thread and turns speech into events:

    ("wake", heard_text)        - name only; now waiting for the request
    ("command", VoiceCommand)   - a request to answer
    ("interrupt", None)         - user said "Suzy" while Suzy was talking
    ("timeout", None)           - said the name but nothing after it
    ("error", message)          - the microphone failed
    """

    def __init__(self, detector: WakeWordDetector, on_event: EventHandler, vad: Any = None,
                 is_speaking: Callable[[], bool] = lambda: False, spoken_text: Callable[[], str] = lambda: "",
                 settings: ListenerSettings | None = None, device: int | None = None) -> None:
        self.detector = detector
        self.on_event = on_event
        self.vad = vad if vad is not None else create_vad()
        self.is_speaking = is_speaking
        self.spoken_text = spoken_text
        self.settings = settings or ListenerSettings()
        self.device = device
        self.segmenter = SpeechSegmenter(self.settings.segment)
        self.mode = Mode.WAKE
        self._deadline = 0.0
        self._followup_from = 0.0
        self._lock = threading.RLock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_done = threading.Event()
        self._capture_audio: np.ndarray | None = None
        self.capture_peak = 0.0
        self.error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="WakeListener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._finish_capture(None)

    def _emit(self, kind: str, payload: Any = None) -> None:
        try:
            self.on_event(kind, payload)
        except Exception:  # noqa: BLE001 - a handler bug must not stop listening
            log.exception("Listener event handler failed for %s", kind)

    # ------------------------------------------------------------------ controls
    def start_followup(self) -> None:
        """After a reply: accept one more request without the wake word."""
        if self.settings.followup_seconds <= 0:
            return
        with self._lock:
            if self.mode is Mode.CAPTURE:
                return
            now = time.monotonic()
            self.mode = Mode.FOLLOWUP
            self._followup_from = now
            self._deadline = now + self.settings.followup_seconds

    def expect_command(self) -> None:
        """Wait for a request without the wake word (she was called by name)."""
        with self._lock:
            self.mode = Mode.COMMAND
            self._deadline = time.monotonic() + self.settings.command_timeout

    @property
    def is_capturing(self) -> bool:
        return self.mode is Mode.CAPTURE

    def capture(self, no_speech_timeout: float = 6.0) -> np.ndarray | None:
        """Push-to-talk through the shared microphone: wait for one utterance."""
        if not self.running:
            raise AudioDeviceError(self.error or "The microphone isn't listening right now.")
        with self._lock:
            self._capture_done.clear()
            self._capture_audio = None
            self.capture_peak = 0.0
            self.mode = Mode.CAPTURE
            self._deadline = time.monotonic() + no_speech_timeout
        self._capture_done.wait(no_speech_timeout + self.segmenter.settings.max_seconds + 2)
        with self._lock:
            if self.mode is Mode.CAPTURE:
                self.mode = Mode.WAKE
            return self._capture_audio

    def stop_capture(self) -> None:
        """Push-to-talk pressed again: end the utterance now."""
        with self._lock:
            if self.mode is Mode.CAPTURE:
                self._finish_capture(self.segmenter.flush())

    def _finish_capture(self, audio: np.ndarray | None) -> None:
        with self._lock:
            self._capture_audio = audio
            if self.mode is Mode.CAPTURE:
                self.mode = Mode.WAKE
            self._capture_done.set()

    # ------------------------------------------------------------------ main loop
    def _run(self) -> None:
        frames: queue.Queue[np.ndarray] = queue.Queue()

        def on_audio(indata: np.ndarray, _frames: int, _time: Any, status: Any) -> None:
            if status:
                log.debug("Microphone status: %s", status)
            frames.put(indata[:, 0].copy())

        try:
            sd = _sd()
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
                                    device=self.device, callback=on_audio)
            with stream:
                log.info("Always-on listening started (VAD: %s)", getattr(self.vad, "name", "?"))
                while self._running.is_set():
                    try:
                        frame = frames.get(timeout=0.2)
                    except queue.Empty:
                        self._check_deadlines()
                        continue
                    self._process_frame(frame)
        except Exception:  # noqa: BLE001 - mic unplugged, driver error, ...
            log.exception("Always-on listening stopped")
            self.error = "My microphone stopped working, so I can't hear 'Suzy' any more."
            self._emit("error", self.error)
        finally:
            self._running.clear()
            self._finish_capture(None)

    def _process_frame(self, frame: np.ndarray) -> None:
        if self.mode is Mode.CAPTURE:
            self.capture_peak = max(self.capture_peak, rms_level(frame))
        try:
            probability = self.vad(frame)
        except Exception:  # noqa: BLE001 - never die on one bad frame
            log.exception("VAD failed on a frame")
            probability = 0.0
        # While Suzy talks, check for her name in short windows so an interruption is quick.
        speaking = self.is_speaking() and self.mode is Mode.WAKE
        self.segmenter.max_seconds = (self.settings.speaking_window if speaking
                                      else self.segmenter.settings.max_seconds)
        segment = self.segmenter.feed(frame, probability)
        if segment is not None:
            self._handle_segment(segment, self.segmenter.last_started)
        self._check_deadlines()

    def _check_deadlines(self) -> None:
        if self.segmenter.active:
            return  # someone is talking: wait for them to finish
        with self._lock:
            if self.mode in (Mode.WAKE,) or time.monotonic() < self._deadline:
                return
            mode = self.mode
            self.mode = Mode.WAKE
        if mode is Mode.COMMAND:
            self._emit("timeout")
        elif mode is Mode.CAPTURE:
            self._finish_capture(None)

    def _is_echo(self, heard: str) -> bool:
        """Suzy's own voice through the speakers - only matters if she said her own name."""
        spoken = self.spoken_text() or ""
        if not find_wake(spoken):
            return False
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(heard.lower(), spoken.lower()) >= 80

    def _handle_segment(self, audio: np.ndarray, started: float) -> None:
        with self._lock:
            mode = self.mode
            if mode is Mode.CAPTURE:
                self._finish_capture(audio)
                return
            if mode is Mode.COMMAND or (mode is Mode.FOLLOWUP and started >= self._followup_from):
                self.mode = Mode.WAKE
                via = "wake" if mode is Mode.COMMAND else "followup"
                self._emit("command", VoiceCommand(audio, via))
                return

        result = self.detector.detect(audio)
        if not result.found:
            return
        speaking = self.is_speaking()
        if speaking and self._is_echo(result.text):
            log.debug("Ignoring my own voice: %r", result.text)
            return
        if speaking:
            self._emit("interrupt")
        if result.request:
            with self._lock:
                self.mode = Mode.WAKE
            self._emit("command", VoiceCommand(audio, "wake", result.text))
        else:
            self.expect_command()
            self._emit("wake", result.text)


class ListenerRecorder:
    """Lets push-to-talk (Enter / Ctrl+Alt+S) record through the always-on listener's microphone."""

    def __init__(self, listener: WakeListener, no_speech_timeout: float = 6.0) -> None:
        self.listener = listener
        self.no_speech_timeout = no_speech_timeout
        self.last_threshold = 0.006

    @property
    def is_recording(self) -> bool:
        return self.listener.is_capturing

    @property
    def last_peak(self) -> float:
        return self.listener.capture_peak

    def stop(self) -> None:
        self.listener.stop_capture()

    def record(self, level_callback: Any = None) -> np.ndarray | None:
        return self.listener.capture(self.no_speech_timeout)
