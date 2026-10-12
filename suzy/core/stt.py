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
                # Auto-detect sometimes picks an unrelated language on short clips; retry in English.
                allowed = self.settings.allowed_languages
                if self.settings.language is None and allowed and language not in allowed:
                    log.debug("Detected %s, not in %s; retrying as English", language, allowed)
                    text, info = self._run(audio, "en")
                    language = "en"
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
