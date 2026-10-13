"""Tests for speech-to-text (faster-whisper mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import faster_whisper
import faster_whisper.utils
import numpy as np
import pytest

from core.config import Config
from core.stt import SpeechToText, STTError, STTSettings


class FakeWhisper:
    """Records calls; returns scripted (text, language) per call."""

    instances: list[FakeWhisper] = []
    script: list[tuple[str, str]] = []

    def __init__(self, path, device, compute_type):
        self.path, self.device, self.compute_type, self.calls = path, device, compute_type, []
        FakeWhisper.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        text, language = FakeWhisper.script.pop(0)
        segments = iter([SimpleNamespace(text=" " + text)] if text else [])
        return segments, SimpleNamespace(language=language, language_probability=0.9)


@pytest.fixture()
def whisper(monkeypatch):
    FakeWhisper.instances.clear()
    FakeWhisper.script = []
    downloads = []

    def fake_download(size, output_dir):
        downloads.append((size, output_dir))
        (__import__("pathlib").Path(output_dir) / "model.bin").write_bytes(b"x")
        return output_dir

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(faster_whisper.utils, "download_model", fake_download)
    return downloads


AUDIO = np.zeros(16_000, np.float32)


def test_downloads_once_then_loads(whisper, tmp_path):
    stt = SpeechToText(STTSettings(), base_dir=tmp_path)
    assert not stt.is_downloaded
    stt.load()
    assert whisper == [("small", str(tmp_path / "models" / "faster-whisper-small"))]
    assert stt.is_ready and FakeWhisper.instances[0].compute_type == "int8"
    # A second instance finds the model on disk and does not download again.
    SpeechToText(STTSettings(), base_dir=tmp_path).load()
    assert len(whisper) == 1


def test_transcribe_text_and_options(whisper, tmp_path):
    FakeWhisper.script = [("Open Chrome please.", "en")]
    stt = SpeechToText(STTSettings(beam_size=2), base_dir=tmp_path)
    result = stt.transcribe(AUDIO)
    assert result.text == "Open Chrome please." and result.language == "en"
    call = FakeWhisper.instances[0].calls[0]
    assert call["vad_filter"] is True and call["beam_size"] == 2 and call["language"] is None
    assert "Volume thoda kam karo" in call["initial_prompt"]


def test_unexpected_language_is_retried_as_english(whisper, tmp_path):
    FakeWhisper.script = [("Ouvrir chrome", "fr"), ("Open chrome", "en")]
    result = SpeechToText(base_dir=tmp_path).transcribe(AUDIO)
    assert result.text == "Open chrome" and result.language == "en"
    assert [c["language"] for c in FakeWhisper.instances[0].calls] == [None, "en"]


def test_hindi_is_accepted(whisper, tmp_path):
    FakeWhisper.script = [("volume kam karo", "hi")]
    assert SpeechToText(base_dir=tmp_path).transcribe(AUDIO).language == "hi"


def test_forced_language(whisper, tmp_path):
    FakeWhisper.script = [("hello", "en")]
    SpeechToText(STTSettings(language="en"), base_dir=tmp_path).transcribe(AUDIO)
    assert FakeWhisper.instances[0].calls[0]["language"] == "en"


@pytest.mark.parametrize("text", ["Thanks for watching!", "you", "", "..."])
def test_hallucinations_are_dropped(whisper, tmp_path, text):
    FakeWhisper.script = [(text, "en")]
    assert SpeechToText(base_dir=tmp_path).transcribe(AUDIO).text == ""


def test_thank_you_is_kept(whisper, tmp_path):
    FakeWhisper.script = [("Thank you.", "en")]
    assert SpeechToText(base_dir=tmp_path).transcribe(AUDIO).text == "Thank you."


def test_download_failure_message(monkeypatch, tmp_path):
    def offline(size, output_dir):
        raise OSError("Name or service not known")

    monkeypatch.setattr(faster_whisper.utils, "download_model", offline)
    stt = SpeechToText(base_dir=tmp_path)
    with pytest.raises(STTError, match="need internet once"):
        stt.load()
    assert "internet" in stt.load_error and stt.wait_until_loaded(0) is False


def test_load_failure_with_model_on_disk(monkeypatch, whisper, tmp_path):
    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("corrupt model")

    model_dir = tmp_path / "models" / "faster-whisper-small"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(faster_whisper, "WhisperModel", Broken)
    stt = SpeechToText(base_dir=tmp_path)
    stt.load_async().join(5)
    assert not stt.is_ready and "corrupt model" in stt.load_error


def test_transcription_crash_is_friendly(whisper, tmp_path):
    stt = SpeechToText(base_dir=tmp_path)
    stt.load()
    FakeWhisper.script = []  # pop from empty list -> IndexError inside transcribe
    with pytest.raises(STTError, match="speech recognition"):
        stt.transcribe(AUDIO)


def test_settings_from_config():
    settings = STTSettings.from_config(Config({"stt": {"model": "base", "language": "EN", "beam_size": 0}}))
    assert settings.model == "base" and settings.language == "en" and settings.beam_size == 1
    assert STTSettings.from_config(Config()).allowed_languages == ("en", "hi")


def test_absolute_models_dir(tmp_path):
    stt = SpeechToText(STTSettings(models_dir=str(tmp_path / "m")), base_dir=tmp_path / "ignored")
    assert stt.model_dir == tmp_path / "m" / "faster-whisper-small"
