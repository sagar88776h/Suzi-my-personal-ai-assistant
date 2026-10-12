"""Low-level audio: microphone capture, speaker playback and short UI sounds.

``sounddevice`` (PortAudio) is imported lazily so a PC with a broken or
missing audio driver still starts Suzy - voice features just report a
friendly error instead.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("suzy.audio")

SAMPLE_RATE = 16_000        # what Whisper expects
BLOCK_SECONDS = 0.03        # 30 ms microphone blocks
LevelCallback = Callable[[float], None]


class AudioDeviceError(RuntimeError):
    """A microphone or speaker problem, with a message safe to show the user."""


def _sd() -> Any:
    """Import sounddevice on demand, turning a missing PortAudio into a friendly error."""
    try:
        import sounddevice
    except (OSError, ImportError) as exc:
        raise AudioDeviceError("The audio system isn't available on this PC (PortAudio failed to load).") from exc
    return sounddevice


@dataclass
class DeviceReport:
    microphone: str | None
    speaker: str | None
    problems: list[str]


def check_devices() -> DeviceReport:
    """Find the default microphone and speaker for the startup self-check."""
    problems: list[str] = []
    microphone = speaker = None
    try:
        sd = _sd()
    except AudioDeviceError as exc:
        return DeviceReport(None, None, [str(exc)])
    try:
        microphone = sd.query_devices(kind="input")["name"]
    except Exception:  # noqa: BLE001 - PortAudio raises various errors
        problems.append("I can't find a microphone. Plug one in, or check Windows Sound settings.")
    try:
        speaker = sd.query_devices(kind="output")["name"]
    except Exception:  # noqa: BLE001
        problems.append("I can't find speakers or headphones, so I won't be able to talk.")
    return DeviceReport(microphone, speaker, problems)


def rms_level(samples: np.ndarray) -> float:
    """Loudness of a block of float samples (0.0 = silence, ~0.3 = loud speech)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))


# =========================================================================== recording
@dataclass
class RecordingSettings:
    max_seconds: float = 15.0          # hard limit for one utterance
    silence_seconds: float = 1.2       # pause that ends the utterance
    no_speech_timeout: float = 6.0     # give up if nothing is said at all
    min_speech_seconds: float = 0.3    # shorter "speech" is treated as noise
    energy_threshold: float = 0.015    # minimum loudness counted as speech
    calibration_seconds: float = 0.3   # measure background noise first


class Recorder:
    """Records one spoken utterance from the default microphone.

    Recording stops when the speaker pauses, when ``stop()`` is called (e.g.
    the push-to-talk hotkey is pressed again), or after ``max_seconds``.
    Speech is detected by loudness relative to the measured background noise.
    """

    def __init__(self, settings: RecordingSettings | None = None, device: int | str | None = None) -> None:
        self.settings = settings or RecordingSettings()
        self.device = device
        self._stop = threading.Event()
        self._active = threading.Event()

    @property
    def is_recording(self) -> bool:
        return self._active.is_set()

    def stop(self) -> None:
        self._stop.set()

    def record(self, level_callback: LevelCallback | None = None) -> np.ndarray | None:
        """Block until an utterance is captured. Returns 16 kHz mono float32, or None if nothing was said."""
        sd = _sd()
        settings = self.settings
        blocks: queue.Queue[np.ndarray] = queue.Queue()

        def on_audio(indata: np.ndarray, _frames: int, _time: Any, status: Any) -> None:
            if status:
                log.debug("Microphone status: %s", status)
            blocks.put(indata[:, 0].copy())

        self._stop.clear()
        block_size = int(SAMPLE_RATE * BLOCK_SECONDS)
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                    blocksize=block_size, device=self.device, callback=on_audio)
        except Exception as exc:  # noqa: BLE001 - PortAudio errors vary by driver
            raise AudioDeviceError(
                "I couldn't open the microphone. Is it plugged in and allowed in Windows privacy settings?"
            ) from exc

        captured: list[np.ndarray] = []
        noise_samples: list[float] = []
        speech_seconds = 0.0
        silence_run = 0.0
        heard_speech = False
        started = time.monotonic()
        self._active.set()
        try:
            with stream:
                while not self._stop.is_set():
                    elapsed = time.monotonic() - started
                    if elapsed > settings.max_seconds:
                        log.debug("Recording hit max length")
                        break
                    if not heard_speech and elapsed > settings.no_speech_timeout:
                        log.debug("No speech detected before timeout")
                        break
                    try:
                        block = blocks.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    captured.append(block)
                    level = rms_level(block)
                    if level_callback is not None:
                        level_callback(level)

                    duration = len(block) / SAMPLE_RATE
                    if len(captured) * BLOCK_SECONDS <= settings.calibration_seconds:
                        noise_samples.append(level)
                        continue
                    noise_floor = float(np.median(noise_samples)) if noise_samples else 0.0
                    threshold = max(settings.energy_threshold, noise_floor * 3.0)
                    if level >= threshold:
                        speech_seconds += duration
                        silence_run = 0.0
                        if speech_seconds >= settings.min_speech_seconds:
                            heard_speech = True
                    else:
                        silence_run += duration
                        if heard_speech and silence_run >= settings.silence_seconds:
                            break
        except AudioDeviceError:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. mic unplugged mid-recording
            raise AudioDeviceError("The microphone stopped working while I was listening.") from exc
        finally:
            # Blocks still queued arrived after the decision to stop (trailing
            # silence, or audio after the user pressed stop), so they're dropped.
            self._active.clear()

        if not captured or (not heard_speech and not self._stop.is_set()):
            return None
        audio = np.concatenate(captured).astype(np.float32)
        if rms_level(audio) < settings.energy_threshold / 3:
            return None  # stopped early but nothing audible was said
        return audio


# =========================================================================== playback
class Player:
    """Plays mono float32 audio on the default speaker; can be stopped instantly."""

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._stop = threading.Event()
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        self._stop.clear()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def play(self, samples: np.ndarray, sample_rate: int, level_callback: LevelCallback | None = None) -> bool:
        """Play ``samples`` and block until done. Returns False if stopped early."""
        if self._stop.is_set():
            return False
        if samples.size == 0:
            return True
        sd = _sd()
        data = np.ascontiguousarray(samples, dtype=np.float32)
        position = 0
        finished = threading.Event()

        def on_output(outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
            nonlocal position
            if status:
                log.debug("Speaker status: %s", status)
            chunk = data[position:position + frames]
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0.0
            position += len(chunk)
            if level_callback is not None:
                level_callback(rms_level(chunk))
            if position >= len(data) or self._stop.is_set():
                raise sd.CallbackStop

        try:
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                     device=self.device, callback=on_output, finished_callback=finished.set)
        except Exception as exc:  # noqa: BLE001
            raise AudioDeviceError("I couldn't open the speakers. Are they plugged in?") from exc

        self._playing.set()
        try:
            with stream:
                timeout = len(data) / sample_rate + 2.0
                deadline = time.monotonic() + timeout
                while not finished.wait(0.05):
                    if self._stop.is_set() or time.monotonic() > deadline:
                        break
        except Exception as exc:  # noqa: BLE001 - e.g. headphones unplugged mid-sentence
            raise AudioDeviceError("The speakers stopped working while I was talking.") from exc
        finally:
            self._playing.clear()
            if level_callback is not None:
                level_callback(0.0)
        return not self._stop.is_set()


def tone(frequency: float, seconds: float, sample_rate: int = 24_000, volume: float = 0.18) -> np.ndarray:
    """A short sine beep with soft edges (no clicks)."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    wave = np.sin(2 * np.pi * frequency * t) * volume
    fade = min(len(wave) // 4, int(sample_rate * 0.01))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave.astype(np.float32)


def play_cue(kind: str, player: Player | None = None) -> None:
    """Play the 'start listening' / 'stop listening' chime. Failures are ignored."""
    sample_rate = 24_000
    if kind == "start":
        sound = np.concatenate([tone(660, 0.07, sample_rate), tone(880, 0.09, sample_rate)])
    else:
        sound = np.concatenate([tone(880, 0.07, sample_rate), tone(660, 0.09, sample_rate)])
    try:
        (player or Player()).play(sound, sample_rate)
    except AudioDeviceError as exc:
        log.debug("Cue sound failed: %s", exc)
