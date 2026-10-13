"""Tests for cloud speech recognition (Groq Whisper) and the cloud->local fallback."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import groq
import httpx
import numpy as np
import pytest

from core.stt import (
    GroqSpeechToText,
    HybridSpeechToText,
    SpeechToText,
    STTError,
    STTSettings,
    Transcript,
    create_speech_to_text,
    language_code,
    retry_language,
    to_wav_bytes,
)

AUDIO = (np.sin(np.linspace(0, 200, 16_000)) * 0.3).astype(np.float32)


class FakeTranscriptions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        text, language = result
        return SimpleNamespace(text=text, language=language, model_extra={})


def cloud(*results, **kwargs):
    stt = GroqSpeechToText("gsk_test", **kwargs)
    fake = FakeTranscriptions(results)
    stt._client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake))
    return stt, fake


def _error(cls, code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


def test_wav_encoding_roundtrip():
    data = to_wav_bytes(AUDIO)
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16_000, 16_000)


def test_cloud_transcription_request():
    stt, fake = cloud(("Open Notepad please.", "english"))
    result = stt.transcribe(AUDIO)
    assert result.text == "Open Notepad please." and result.language == "en"
    call = fake.calls[0]
    assert call["model"] == "whisper-large-v3-turbo" and call["response_format"] == "verbose_json"
    assert call["file"][0] == "speech.wav" and call["file"][1][:4] == b"RIFF"
    assert "Volume thoda kam karo" in call["prompt"] and "language" not in call


def test_urdu_is_redone_as_hindi():
    stt, fake = cloud(("والیوم کم کرو", "urdu"), ("volume kam karo", "hindi"))
    assert stt.transcribe(AUDIO).text == "volume kam karo"
    assert fake.calls[1]["language"] == "hi"


def test_hallucination_dropped():
    stt, _ = cloud(("Thanks for watching!", "english"))
    assert stt.transcribe(AUDIO).text == ""


def test_bad_key_disables_cloud():
    stt, _ = cloud(_error(groq.AuthenticationError, 401))
    with pytest.raises(STTError) as info:
        stt.transcribe(AUDIO)
    assert not info.value.retryable and not stt.available


def test_rate_limit_is_retryable():
    stt, _ = cloud(_error(groq.RateLimitError, 429))
    with pytest.raises(STTError) as info:
        stt.transcribe(AUDIO)
    assert info.value.retryable and stt.available


@pytest.mark.parametrize("value, code", [("english", "en"), ("Hindi", "hi"), ("en", "en"), (None, None),
                                         ("urdu", "ur"), ("swahili", "sw")])
def test_language_code(value, code):
    assert language_code(value) == code


@pytest.mark.parametrize("detected, forced, allowed, expected", [
    ("en", None, ("en", "hi"), None), ("hi", None, ("en", "hi"), None), ("ur", None, ("en", "hi"), "hi"),
    ("fr", None, ("en", "hi"), "en"), ("fr", "en", ("en", "hi"), None), ("fr", None, (), None),
])
def test_retry_language(detected, forced, allowed, expected):
    assert retry_language(detected, forced, allowed) == expected


# ----------------------------------------------------------------- hybrid
class FakeLocal:
    def __init__(self, ready=True, load_error=None):
        self.ready, self.load_error, self.calls = ready, load_error, 0
        self.settings = STTSettings()

    @property
    def is_ready(self):
        return self.ready

    @property
    def is_downloaded(self):
        return True

    def wait_until_loaded(self, timeout=None):
        return self.ready

    def load_async(self):
        return None

    def transcribe(self, audio):
        self.calls += 1
        return Transcript("local text", "en")


def test_hybrid_prefers_cloud():
    stt, _ = cloud(("cloud text", "english"))
    hybrid = HybridSpeechToText(stt, FakeLocal(ready=False))
    assert hybrid.is_ready and hybrid.load_error is None and hybrid.wait_until_loaded(0)
    assert hybrid.transcribe(AUDIO).text == "cloud text"


def test_hybrid_falls_back_and_cools_down():
    stt, fake = cloud(groq.APIConnectionError(request=httpx.Request("POST", "https://x")))
    local = FakeLocal()
    hybrid = HybridSpeechToText(stt, local)
    assert hybrid.transcribe(AUDIO).text == "local text"
    assert hybrid.transcribe(AUDIO).text == "local text"  # cooling down: cloud not tried again
    assert len(fake.calls) == 1 and local.calls == 2


def test_hybrid_reports_local_problems_when_cloud_is_gone():
    stt, _ = cloud(_error(groq.AuthenticationError, 401))
    hybrid = HybridSpeechToText(stt, FakeLocal(ready=False, load_error="Couldn't download the model."))
    with pytest.raises(STTError, match="Couldn't download"):
        hybrid.transcribe(AUDIO)
    assert not hybrid.is_ready and hybrid.load_error == "Couldn't download the model."


def test_create_speech_to_text():
    assert isinstance(create_speech_to_text(STTSettings(), "gsk_x"), HybridSpeechToText)
    assert type(create_speech_to_text(STTSettings(), None)) is SpeechToText
    assert type(create_speech_to_text(STTSettings(engine="local"), "gsk_x")) is SpeechToText
