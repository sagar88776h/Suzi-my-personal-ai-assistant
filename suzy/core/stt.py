"""Speech-to-text: Suzy's ears, using faster-whisper on the CPU.

The Whisper model (``small`` by default, ~480 MB) is downloaded once into
``models/`` the first time it is needed and then works fully offline. Loading
happens on a background thread so Suzy starts instantly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.paths import APP_DIR

log = logging.getLogger("suzy.stt")

# Phrases Whisper tends to "hear" in silence or noise (from its YouTube training
# data). Real phrases like "thank you" are kept - people do say them to Suzy.
HALLUCINATIONS = {
    "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "please subscribe.", "subscribe.", "you", ".", "...",
}

# Biases Whisper toward Latin-script Hinglish and the words Suzy hears most.
DEFAULT_PROMPT = "Hey Suzy, open Chrome. Volume thoda kam karo. Aaj ka weather kya hai?"


class STTError(RuntimeError):
    """Speech recognition is unavailable; the message is safe to show the user."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def retry_language(detected: str | None, forced: str | None, allowed: tuple[str, ...]) -> str | None:
    """Which language to re-run recognition in, or None to keep the result.

    Whisper often labels spoken Hindi as Urdu (and writes Urdu script), and on short
    clips may pick an unrelated language: Urdu -> Hindi, anything else -> English.
    """
    if forced or not allowed or detected in allowed:
        return None
    if detected == "ur" and "hi" in allowed:
        return "hi"
    return "en" if "en" in allowed else allowed[0]


@dataclass
class Transcript:
    text: str
    language: str | None = None
    language_probability: float = 0.0
    seconds: float = 0.0  # processing time


@dataclass
class STTSettings:
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None                  # None = auto-detect
    allowed_languages: tuple[str, ...] = ("en", "hi")
    beam_size: int = 1
    initial_prompt: str | None = DEFAULT_PROMPT
    models_dir: str = "models"
    engine: str = "groq"                          # "groq" (cloud, fast) or "local"
    cloud_model: str = "whisper-large-v3-turbo"

    @classmethod
    def from_config(cls, config: Any) -> STTSettings:
        defaults = cls()
        language = config.get("stt.language", None)
        allowed = config.get("stt.allowed_languages", list(defaults.allowed_languages)) or []
        return cls(
            model=str(config.get("stt.model", defaults.model)),
            device=str(config.get("stt.device", defaults.device)),
            compute_type=str(config.get("stt.compute_type", defaults.compute_type)),
            language=str(language).strip().lower() if language else None,
            allowed_languages=tuple(str(code).lower() for code in allowed),
            beam_size=max(1, int(config.get("stt.beam_size", defaults.beam_size))),
            initial_prompt=config.get("stt.initial_prompt", defaults.initial_prompt),
            models_dir=str(config.get("stt.models_dir", defaults.models_dir)),
            engine=str(config.get("stt.engine", defaults.engine)).strip().lower(),
            cloud_model=str(config.get("stt.cloud_model", defaults.cloud_model)),
        )


class SpeechToText:
    """Thread-safe wrapper around a lazily loaded faster-whisper model."""

    def __init__(self, settings: STTSettings | None = None, base_dir: Path = APP_DIR) -> None:
        self.settings = settings or STTSettings()
        models_dir = Path(self.settings.models_dir)
        self.model_dir = (models_dir if models_dir.is_absolute() else base_dir / models_dir) / (
            f"faster-whisper-{self.settings.model}"
        )
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self._loaded = threading.Event()
        self.load_error: str | None = None

    # ------------------------------------------------------------------ loading
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_downloaded(self) -> bool:
        return (self.model_dir / "model.bin").exists()

    def load(self) -> None:
        """Download (first run only) and load the model. Raises STTError."""
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
                from faster_whisper.utils import download_model
            except ImportError as exc:
                self.load_error = "The speech recognition library (faster-whisper) isn't installed."
                raise STTError(self.load_error) from exc
            try:
                if not self.is_downloaded:
                    log.info("Downloading Whisper '%s' model to %s", self.settings.model, self.model_dir)
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    download_model(self.settings.model, output_dir=str(self.model_dir))
                started = time.perf_counter()
                self._model = WhisperModel(
                    str(self.model_dir), device=self.settings.device, compute_type=self.settings.compute_type,
                )
                log.info("Whisper model loaded in %.1fs", time.perf_counter() - started)
                self.load_error = None
            except Exception as exc:  # noqa: BLE001 - network, disk or model errors
                log.exception("Loading the Whisper model failed")
                if not self.is_downloaded:
                    self.load_error = ("I couldn't download my speech model. I need internet once for that - "
                                       "check your connection and restart me.")
                else:
                    self.load_error = f"I couldn't load my speech model ({exc})."
                raise STTError(self.load_error) from exc
            finally:
                self._loaded.set()

    def load_async(self) -> threading.Thread:
        """Start loading in the background; errors are kept in ``load_error``."""

        def run() -> None:
            try:
                self.load()
            except STTError:
                pass

        thread = threading.Thread(target=run, name="WhisperLoader", daemon=True)
        thread.start()
        return thread

    def wait_until_loaded(self, timeout: float | None = None) -> bool:
        """Wait for a background load to finish. True if the model is ready."""
        self._loaded.wait(timeout)
        return self.is_ready

    # ------------------------------------------------------------------ transcription
    def _run(self, audio: np.ndarray, language: str | None) -> tuple[str, Any]:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.settings.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.settings.initial_prompt or None,
        )
        text = "".join(segment.text for segment in segments).strip()
        return text, info

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Turn 16 kHz mono float32 audio into text. Raises STTError."""
        if self._model is None:
            self.load()
        started = time.perf_counter()
        try:
            with self._transcribe_lock:
                text, info = self._run(audio, self.settings.language)
                language = getattr(info, "language", None)
                # Auto-detect sometimes picks an unrelated language on short clips; retry.
                retry = retry_language(language, self.settings.language, self.settings.allowed_languages)
                if retry:
                    log.debug("Detected %s; retrying as %s", language, retry)
                    text, info = self._run(audio, retry)
                    language = retry
        except Exception as exc:  # noqa: BLE001 - CTranslate2 / decoding errors
            log.exception("Transcription failed")
            raise STTError("Sorry, my speech recognition hit a problem. Could you try again?") from exc

        if text.lower() in HALLUCINATIONS:
            log.debug("Dropping likely hallucination: %r", text)
            text = ""
        return Transcript(
            text=text,
            language=language,
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            seconds=time.perf_counter() - started,
        )


# =========================================================================== cloud (Groq)
LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "urdu": "ur", "marathi": "mr", "bengali": "bn", "punjabi": "pa",
    "gujarati": "gu", "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml", "nepali": "ne",
}


def language_code(value: Any) -> str | None:
    """Whisper APIs report "english" or "en"; normalise to the two-letter code."""
    if not value:
        return None
    text = str(value).strip().lower()
    return text if len(text) == 2 else LANGUAGE_CODES.get(text, text[:2])


def to_wav_bytes(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """16-bit mono WAV in memory (what speech APIs accept)."""
    import io
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class GroqSpeechToText:
    """Whisper large-v3-turbo on Groq: ~0.3-0.5 s per request, very accurate (Hindi too).

    Free with the same GROQ_API_KEY (about 2,000 requests a day). Used for requests
    only - the always-on "Suzy" check stays on the PC.
    """

    def __init__(self, api_key: str | None, model: str = "whisper-large-v3-turbo",
                 language: str | None = None, allowed_languages: tuple[str, ...] = ("en", "hi"),
                 prompt: str | None = DEFAULT_PROMPT, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.allowed_languages = allowed_languages
        self.prompt = prompt
        self.timeout = timeout
        self.disabled_reason: str | None = None
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.disabled_reason is None

    def _request(self, wav: bytes, language: str | None) -> tuple[str, str | None]:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        kwargs: dict[str, Any] = {"model": self.model, "file": ("speech.wav", wav),
                                  "response_format": "verbose_json", "temperature": 0.0}
        if self.prompt:
            kwargs["prompt"] = self.prompt
        if language:
            kwargs["language"] = language
        response = self._client.audio.transcriptions.create(**kwargs)
        extra = getattr(response, "model_extra", None) or {}
        detected = getattr(response, "language", None) or extra.get("language")
        return (getattr(response, "text", "") or "").strip(), language_code(detected)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if not self.available:
            raise STTError(self.disabled_reason or "No Groq key for cloud speech recognition.", retryable=False)
        started = time.perf_counter()
        wav = to_wav_bytes(audio)
        try:
            text, language = self._request(wav, self.language)
            retry = retry_language(language, self.language, self.allowed_languages)
            if retry:
                log.debug("Cloud STT detected %s; retrying as %s", language, retry)
                text, _ = self._request(wav, retry)
                language = retry
        except Exception as exc:  # noqa: BLE001 - classified below
            raise self._error(exc) from exc
        if text.lower() in HALLUCINATIONS:
            text = ""
        return Transcript(text, language, 1.0, time.perf_counter() - started)

    def _error(self, exc: Exception) -> STTError:
        try:
            import groq
        except ImportError:
            return STTError(f"cloud speech recognition failed: {exc}")
        if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
            self.disabled_reason = "The Groq key was rejected for speech recognition."
            return STTError(self.disabled_reason, retryable=False)
        if isinstance(exc, groq.NotFoundError):
            self.disabled_reason = f"Groq no longer offers the {self.model} speech model."
            return STTError(self.disabled_reason, retryable=False)
        return STTError(f"cloud speech recognition failed: {str(exc)[:200]}")


class HybridSpeechToText:
    """Cloud speech recognition first (fast); the local Whisper model as backup.

    Has the same interface as :class:`SpeechToText`, so the rest of Suzy doesn't care.
    After a cloud failure (offline, rate limit) the local model is used for a minute.
    """

    COOLDOWN_SECONDS = 60

    def __init__(self, cloud: GroqSpeechToText, local: SpeechToText) -> None:
        self.cloud = cloud
        self.local = local
        self._cloud_down_until = 0.0

    # Pass-through for setup code that manages the local model.
    @property
    def settings(self) -> STTSettings:
        return self.local.settings

    @property
    def is_downloaded(self) -> bool:
        return self.local.is_downloaded

    def load_async(self) -> threading.Thread:
        return self.local.load_async()

    def load(self) -> None:
        self.local.load()

    @property
    def is_ready(self) -> bool:
        return self.cloud.available or self.local.is_ready

    @property
    def load_error(self) -> str | None:
        return None if self.cloud.available else self.local.load_error

    def wait_until_loaded(self, timeout: float | None = None) -> bool:
        return True if self.cloud.available else self.local.wait_until_loaded(timeout)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self.cloud.available and time.monotonic() >= self._cloud_down_until:
            try:
                return self.cloud.transcribe(audio)
            except STTError as exc:
                log.warning("Cloud speech recognition failed (%s); using the local model", exc)
                if exc.retryable:
                    self._cloud_down_until = time.monotonic() + self.COOLDOWN_SECONDS
        if not self.local.is_ready and not self.local.wait_until_loaded(0) and self.local.load_error:
            raise STTError(self.local.load_error)
        return self.local.transcribe(audio)


def create_speech_to_text(settings: STTSettings, api_key: str | None) -> SpeechToText | HybridSpeechToText:
    """The recogniser for requests: Groq cloud first (if enabled and a key exists), local as backup."""
    local = SpeechToText(settings)
    if settings.engine == "groq" and api_key:
        cloud = GroqSpeechToText(api_key, settings.cloud_model, settings.language,
                                 settings.allowed_languages, settings.initial_prompt)
        return HybridSpeechToText(cloud, local)
    return local
