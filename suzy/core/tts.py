"""Text-to-speech: Suzy's voice.

* Primary voice: Microsoft Edge neural voices via ``edge-tts`` (free, needs
  internet). Default "en-IN-NeerjaNeural", fallback "en-US-AriaNeural".
* Offline fallback: the Windows built-in voice through ``pyttsx3`` (SAPI5),
  used automatically when edge-tts can't be reached.

Replies are split into sentences. The next sentence is synthesized while the
current one plays, so Suzy starts talking after the first sentence is ready
instead of waiting for the whole reply. ``stop()`` interrupts instantly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core.audio import AudioDeviceError, LevelCallback, Player

log = logging.getLogger("suzy.tts")

EDGE_SAMPLE_RATE = 24_000
EDGE_RETRY_AFTER_SECONDS = 60  # after a network failure, use the offline voice for a while


class SynthesisError(RuntimeError):
    """Synthesis failed. ``network`` is True when the cause was connectivity."""

    def __init__(self, message: str, network: bool = False) -> None:
        super().__init__(message)
        self.network = network


class Synthesizer(Protocol):
    name: str

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        """Return (mono float32 samples, sample rate). ``voice`` overrides the default
        (e.g. the Hindi voice for Devanagari text). Raises SynthesisError."""


# =========================================================================== text prep
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*]\s+)", re.MULTILINE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def has_devanagari(text: str) -> bool:
    """True if the text contains Hindi (Devanagari) characters."""
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def clean_for_speech(text: str) -> str:
    """Remove things that sound bad when read aloud (markdown, raw URLs)."""
    text = _URL.sub("the link", text)
    text = _MARKDOWN.sub("", text)
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str, min_chars: int = 30, max_chars: int = 250) -> list[str]:
    """Split into speakable chunks: short sentences are merged, very long ones split at commas."""
    pieces = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(", ", 0, max_chars)
            cut = cut + 1 if cut > min_chars else max_chars
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            chunks.append(piece)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) < min_chars and len(merged[-1]) + len(chunk) < max_chars:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def speech_chunks(text: str, whole_up_to: int = 320, group_up_to: int = 400) -> list[str]:
    """How a reply is sent to the voice service.

    Short replies (most of Suzy's) go in ONE request, so the intonation flows naturally
    across sentences. Longer ones: the first sentence alone (so she starts talking
    quickly), then the rest in large groups.
    """
    if len(text) <= whole_up_to:
        return [text] if text else []
    sentences = split_sentences(text, max_chars=250)
    chunks = [sentences[0]]
    for sentence in sentences[1:]:
        if len(chunks) > 1 and len(chunks[-1]) + len(sentence) < group_up_to:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
    return chunks


def normalize_rate(value: Any) -> str:
    """Accept 10, "10", "+10%" or "-5%" and return edge-tts format ("+10%")."""
    text = str(value).strip().rstrip("%") or "0"
    number = int(float(text))
    return f"{number:+d}%"


def normalize_pitch(value: Any) -> str:
    """Accept 5, "5", "+5Hz" and return edge-tts format ("+5Hz")."""
    text = str(value).strip()
    text = text[:-2] if text.lower().endswith("hz") else text
    number = int(float(text or 0))
    return f"{number:+d}Hz"


# =========================================================================== synthesizers
class EdgeSynthesizer:
    """Neural voices from Microsoft Edge's online TTS service."""

    name = "edge"

    def __init__(self, voice: str, fallback_voice: str | None = None, rate: str = "+0%",
                 pitch: str = "+0Hz", volume: str = "+0%", timeout: int = 10) -> None:
        self.voice = voice
        self.fallback_voice = fallback_voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.timeout = timeout

    async def _fetch(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(
            text, voice, rate=self.rate, pitch=self.pitch, volume=self.volume,
            connect_timeout=self.timeout, receive_timeout=max(self.timeout * 3, 30),
        )
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk["data"])
        return bytes(audio)

    def _fetch_sync(self, text: str, voice: str) -> bytes:
        try:
            import aiohttp
            import edge_tts.exceptions as edge_errors
        except ImportError as exc:
            raise SynthesisError("edge-tts is not installed") from exc
        try:
            return asyncio.run(self._fetch(text, voice))
        except edge_errors.NoAudioReceived as exc:
            raise SynthesisError(f"no audio for voice {voice}") from exc
        except (TimeoutError, aiohttp.ClientError, OSError, edge_errors.WebSocketError,
                edge_errors.SkewAdjustmentError) as exc:
            raise SynthesisError(f"can't reach the voice service: {exc}", network=True) from exc
        except (edge_errors.EdgeTTSException, ValueError) as exc:
            raise SynthesisError(f"voice service error: {exc}") from exc

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        voice = voice or self.voice
        try:
            mp3 = self._fetch_sync(text, voice)
        except SynthesisError as exc:
            if exc.network or not self.fallback_voice or self.fallback_voice == voice:
                raise
            log.warning("Voice %s failed (%s); trying %s", voice, exc, self.fallback_voice)
            mp3 = self._fetch_sync(text, self.fallback_voice)
        if not mp3:
            raise SynthesisError("the voice service returned no audio")
        return decode_mp3(mp3), EDGE_SAMPLE_RATE


def decode_mp3(data: bytes, sample_rate: int = EDGE_SAMPLE_RATE) -> np.ndarray:
    """Decode MP3 bytes to mono float32 samples."""
    try:
        import miniaudio
    except ImportError as exc:
        raise SynthesisError("miniaudio is not installed") from exc
    try:
        decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32,
                                   nchannels=1, sample_rate=sample_rate)
    except miniaudio.DecodeError as exc:
        raise SynthesisError(f"couldn't decode the voice audio: {exc}") from exc
    return np.frombuffer(decoded.samples, dtype=np.float32).copy()


class OfflineSynthesizer:
    """The Windows built-in voice (SAPI5) via pyttsx3. Works without internet."""

    name = "offline"
    PREFERRED_VOICES = ("heera", "zira", "female")  # Indian English female first
    HINDI_VOICES = ("kalpana", "swara", "hindi", "hemant")  # Windows Hindi voices, if installed

    def __init__(self, rate_percent: int = 0) -> None:
        self.rate_percent = rate_percent

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        hindi = bool(voice and voice.startswith("hi-")) or has_devanagari(text)
        try:
            import pyttsx3
        except ImportError as exc:
            raise SynthesisError("the offline voice (pyttsx3) is not installed") from exc
        try:  # SAPI is a COM API: every thread must initialize COM first.
            import comtypes

            comtypes.CoInitialize()
        except (ImportError, OSError):
            pass

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="suzy-tts-")
        os.close(fd)
        try:
            engine = pyttsx3.init()
            try:
                self._pick_voice(engine, hindi)
                base_rate = engine.getProperty("rate") or 200
                engine.setProperty("rate", int(base_rate * (1 + self.rate_percent / 100)))
                engine.save_to_file(text, path)
                engine.runAndWait()
            finally:
                engine.stop()
            return read_wav(path)
        except SynthesisError:
            raise
        except Exception as exc:  # noqa: BLE001 - SAPI/COM errors vary
            raise SynthesisError(f"the offline voice failed: {exc}") from exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _pick_voice(self, engine: Any, hindi: bool = False) -> None:
        voices = engine.getProperty("voices") or []
        wanted_order = (self.HINDI_VOICES + self.PREFERRED_VOICES) if hindi else self.PREFERRED_VOICES
        for wanted in wanted_order:
            for voice in voices:
                label = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                if wanted in label or (wanted == "female" and getattr(voice, "gender", "") == "female"):
                    engine.setProperty("voice", voice.id)
                    return


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file into mono float32."""
    try:
        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        raise SynthesisError(f"couldn't read the offline voice audio: {exc}") from exc
    if width != 2:
        raise SynthesisError(f"unexpected WAV sample width {width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


# =========================================================================== speaker
@dataclass
class TTSSettings:
    enabled: bool = True
    voice: str = "en-IN-NeerjaExpressiveNeural"
    fallback_voice: str = "en-IN-NeerjaNeural"
    hindi_voice: str | None = "hi-IN-SwaraNeural"  # speaks any reply that contains Hindi (Devanagari)
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    offline_fallback: bool = True

    @classmethod
    def from_config(cls, config: Any) -> TTSSettings:
        defaults = cls()

        def safe(normalizer: Any, key: str, default: str) -> str:
            try:
                return normalizer(config.get(key, default))
            except (TypeError, ValueError):
                log.warning("Invalid %s in config.yaml; using %s", key, default)
                return default

        return cls(
            enabled=bool(config.get("tts.enabled", True)),
            voice=str(config.get("tts.voice", defaults.voice)),
            fallback_voice=str(config.get("tts.fallback_voice", defaults.fallback_voice)),
            rate=safe(normalize_rate, "tts.rate", defaults.rate),
            pitch=safe(normalize_pitch, "tts.pitch", defaults.pitch),
            volume=safe(normalize_rate, "tts.volume", defaults.volume),
            offline_fallback=bool(config.get("tts.offline_fallback", True)),
            hindi_voice=config.get("tts.hindi_voice", defaults.hindi_voice) or None,
        )


class TextToSpeech:
    """Speaks text aloud. ``speak()`` blocks until done; ``stop()`` interrupts it from any thread."""

    def __init__(self, settings: TTSSettings | None = None, player: Player | None = None,
                 primary: Synthesizer | None = None, offline: Synthesizer | None = None) -> None:
        self.settings = settings or TTSSettings()
        self.player = player or Player()
        self.primary: Synthesizer = primary or EdgeSynthesizer(
            self.settings.voice, self.settings.fallback_voice, self.settings.rate,
            self.settings.pitch, self.settings.volume,
        )
        self.offline: Synthesizer | None = offline
        if self.offline is None and self.settings.offline_fallback:
            rate_number = int(self.settings.rate.rstrip("%"))
            self.offline = OfflineSynthesizer(rate_number)
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._lock = threading.Lock()
        self._primary_down_until = 0.0
        self.last_error: str | None = None
        # Called once when the natural voice can't be reached (e.g. to tell Sagar why she sounds robotic).
        self.on_notice: Any = None
        self._noticed = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def update_voice(self, voice: str | None = None, rate: str | None = None, pitch: str | None = None) -> None:
        """Change the voice, speed or pitch for the next things Suzy says."""
        if voice:
            self.settings.voice = voice
        if rate is not None:
            self.settings.rate = normalize_rate(rate)
        if pitch is not None:
            self.settings.pitch = normalize_pitch(pitch)
        if isinstance(self.primary, EdgeSynthesizer):
            self.primary.voice = self.settings.voice
            self.primary.rate = self.settings.rate
            self.primary.pitch = self.settings.pitch
        self._primary_down_until = 0.0  # try the natural voice again right away

    def stop(self) -> None:
        """Interrupt the current speech immediately."""
        self._stop.set()
        self.player.stop()

    def voice_for(self, text: str) -> str | None:
        """The Hindi voice for text containing Hindi script (None = the normal voice)."""
        return self.settings.hindi_voice if self.settings.hindi_voice and has_devanagari(text) else None

    def _synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize one chunk, falling back to the offline voice. Returns None on failure."""
        voice = self.voice_for(text)
        if time.monotonic() >= self._primary_down_until:
            try:
                return self.primary.synthesize(text, voice) if voice else self.primary.synthesize(text)
            except SynthesisError as exc:
                self.last_error = str(exc)
                log.warning("%s voice failed: %s", self.primary.name, exc)
                if exc.network:
                    self._primary_down_until = time.monotonic() + EDGE_RETRY_AFTER_SECONDS
                    if not self._noticed and self.on_notice is not None:
                        self._noticed = True
                        self.on_notice("My natural voice service isn't reachable right now, so I'm using the "
                                       "Windows voice. The natural voice comes back by itself when it can.")
        if self.offline is not None:
            try:
                return self.offline.synthesize(text, voice) if voice else self.offline.synthesize(text)
            except SynthesisError as exc:
                self.last_error = str(exc)
                log.warning("Offline voice failed: %s", exc)
        return None

    def speak(self, text: str, level_callback: LevelCallback | None = None) -> bool:
        """Say ``text``. Returns True if it was spoken completely. Never raises."""
        if not self.settings.enabled:
            return False
        chunks = speech_chunks(clean_for_speech(text or ""))
        if not chunks:
            return True

        with self._lock:  # one utterance at a time
            self._stop.clear()
            self.player.reset()
            self._speaking.set()
            ready: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=2)
            synthesis_failed = threading.Event()

            def offer(item: tuple[np.ndarray, int] | None) -> None:
                # Never block forever: give up as soon as playback has been stopped.
                while not self._stop.is_set():
                    try:
                        ready.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue

            def produce() -> None:
                try:
                    for chunk in chunks:
                        if self._stop.is_set():
                            return
                        audio = self._synthesize(chunk)
                        if audio is None:
                            synthesis_failed.set()
                            return
                        offer(audio)
                except Exception:  # noqa: BLE001 - never let the worker die silently
                    log.exception("TTS synthesis thread crashed")
                    synthesis_failed.set()
                finally:
                    offer(None)  # end-of-speech marker

            worker = threading.Thread(target=produce, name="TTSSynth", daemon=True)
            worker.start()
            completed = True
            try:
                # One continuous stream for the whole reply: no gaps or clicks between chunks.
                completed = self.player.play_queue(ready, EDGE_SAMPLE_RATE, level_callback)
            except AudioDeviceError as exc:
                self.last_error = str(exc)
                log.warning("Playback failed: %s", exc)
                completed = False
            except Exception:  # noqa: BLE001 - speech must never crash Suzy
                log.exception("Unexpected TTS failure")
                completed = False
            finally:
                if self._stop.is_set() or synthesis_failed.is_set():
                    completed = False
                self._stop.set()  # tell the producer to finish
                worker.join(timeout=2)
                self._speaking.clear()
            return completed
