"""Tests for text-to-speech: text prep, synthesizers and the speaking pipeline."""

from __future__ import annotations

import io
import threading
import time
import wave

import numpy as np
import pytest

from core import tts as tts_module
from core.audio import AudioDeviceError
from core.config import Config
from core.tts import (
    EdgeSynthesizer,
    SynthesisError,
    TextToSpeech,
    TTSSettings,
    clean_for_speech,
    decode_mp3,
    normalize_pitch,
    normalize_rate,
    read_wav,
    split_sentences,
)


# ----------------------------------------------------------------- text prep
def test_clean_for_speech():
    text = "**Done!** Opened `notepad`.\n# Heading\n- see https://example.com/x?y=1 now"
    assert clean_for_speech(text) == "Done! Opened notepad. Heading see the link now"


def test_split_sentences_merges_short_and_splits_long():
    assert split_sentences("Hi. Opening Chrome now for you, Sagar. Anything else?") == [
        "Hi. Opening Chrome now for you, Sagar.", "Anything else?"]
    long = ", ".join(["this is a fairly long clause"] * 15) + "."
    chunks = split_sentences(long, max_chars=120)
    assert len(chunks) > 1 and all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == long
    assert split_sentences("") == []
    assert split_sentences("Volume kam kar diya। Aur kuch?") == ["Volume kam kar diya। Aur kuch?"]


@pytest.mark.parametrize("raw, expected", [(0, "+0%"), ("10", "+10%"), ("+15%", "+15%"), ("-5%", "-5%"), (12.7, "+12%")])
def test_normalize_rate(raw, expected):
    assert normalize_rate(raw) == expected


@pytest.mark.parametrize("raw, expected", [(0, "+0Hz"), ("5", "+5Hz"), ("+5Hz", "+5Hz"), ("-3hz", "-3Hz")])
def test_normalize_pitch(raw, expected):
    assert normalize_pitch(raw) == expected


def test_settings_from_config_with_bad_values():
    config = Config({"tts": {"voice": "en-US-AriaNeural", "rate": "fast", "pitch": 4, "enabled": False}})
    settings = TTSSettings.from_config(config)
    assert settings.voice == "en-US-AriaNeural" and settings.rate == "+0%"
    assert settings.pitch == "+4Hz" and settings.enabled is False


# ----------------------------------------------------------------- decoding
def _mp3_tone() -> bytes:
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    container = av.open(buffer, "w", format="mp3")
    stream = container.add_stream("mp3", rate=24_000)
    stream.layout = "mono"
    t = np.arange(24_000) / 24_000
    frame = av.AudioFrame.from_ndarray((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, -1),
                                       format="flt", layout="mono")
    frame.sample_rate = 24_000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def test_decode_mp3_real_audio():
    samples = decode_mp3(_mp3_tone())
    assert samples.dtype == np.float32 and len(samples) == pytest.approx(24_000, abs=1200)
    assert np.sqrt(np.mean(samples ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)


def test_decode_mp3_garbage():
    with pytest.raises(SynthesisError):
        decode_mp3(b"definitely not an mp3")


def test_read_wav(tmp_path):
    path = tmp_path / "x.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(22_050)
        wav.writeframes(np.array([16384, 16384, -16384, -16384], dtype="<i2").tobytes())
    samples, rate = read_wav(str(path))
    assert rate == 22_050 and np.allclose(samples, [0.5, -0.5])
    with pytest.raises(SynthesisError):
        read_wav(str(tmp_path / "missing.wav"))


# ----------------------------------------------------------------- edge synthesizer
def test_edge_uses_fallback_voice(monkeypatch):
    calls = []

    def fake_fetch(self, text, voice):
        calls.append(voice)
        if voice == "bad-voice":
            raise SynthesisError("no audio for voice bad-voice")
        return _mp3_tone()

    monkeypatch.setattr(EdgeSynthesizer, "_fetch_sync", fake_fetch)
    samples, rate = EdgeSynthesizer("bad-voice", "en-US-AriaNeural").synthesize("hi")
    assert calls == ["bad-voice", "en-US-AriaNeural"] and rate == 24_000 and len(samples) > 0


def test_edge_network_error_skips_fallback_voice(monkeypatch):
    calls = []

    def offline(self, text, voice):
        calls.append(voice)
        raise SynthesisError("can't reach", network=True)

    monkeypatch.setattr(EdgeSynthesizer, "_fetch_sync", offline)
    with pytest.raises(SynthesisError) as info:
        EdgeSynthesizer("a", "b").synthesize("hi")
    assert info.value.network and calls == ["a"]


def test_edge_maps_connection_errors(monkeypatch):
    import aiohttp
    import edge_tts

    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            raise aiohttp.ClientConnectionError("DNS failure")
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(edge_tts, "Communicate", Broken)
    with pytest.raises(SynthesisError) as info:
        EdgeSynthesizer("en-IN-NeerjaNeural").synthesize("hello")
    assert info.value.network


def test_edge_collects_audio_chunks(monkeypatch):
    import edge_tts

    mp3 = _mp3_tone()
    seen = {}

    class Fake:
        def __init__(self, text, voice, **kwargs):
            seen.update(kwargs, voice=voice, text=text)

        async def stream(self):
            yield {"type": "WordBoundary"}
            yield {"type": "audio", "data": mp3[:1000]}
            yield {"type": "audio", "data": mp3[1000:]}

    monkeypatch.setattr(edge_tts, "Communicate", Fake)
    samples, _ = EdgeSynthesizer("en-IN-NeerjaNeural", rate="+10%", pitch="-2Hz").synthesize("hello")
    assert len(samples) > 20_000
    assert seen["voice"] == "en-IN-NeerjaNeural" and seen["rate"] == "+10%" and seen["pitch"] == "-2Hz"


# ----------------------------------------------------------------- speaking pipeline
class FakeSynth:
    def __init__(self, name="fake", fail_with=None, delay=0.0):
        self.name, self.fail_with, self.delay, self.texts = name, fail_with, delay, []

    def synthesize(self, text):
        self.texts.append(text)
        time.sleep(self.delay)
        if self.fail_with is not None:
            raise self.fail_with
        return np.full(100, 0.1, np.float32), 24_000


class FakePlayer:
    def __init__(self, block_seconds=0.0, error=None):
        self.block_seconds, self.error, self.played = block_seconds, error, []
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def reset(self):
        self._stop.clear()

    def play(self, samples, rate, level_callback=None):
        if self.error:
            raise self.error
        if self._stop.wait(self.block_seconds) or self._stop.is_set():
            return False
        self.played.append(len(samples))
        return True


def make_tts(primary=None, offline=None, player=None, **settings):
    return TextToSpeech(TTSSettings(**settings), player=player or FakePlayer(),
                        primary=primary or FakeSynth(), offline=offline)


def test_speaks_every_sentence_in_order():
    primary = FakeSynth()
    speaker = make_tts(primary)
    text = "Opening Chrome now for you, Sagar. The volume is at thirty percent. Anything else I can do?"
    assert speaker.speak(text) is True
    assert primary.texts == split_sentences(text)
    assert len(speaker.player.played) == len(primary.texts) and not speaker.is_speaking


def test_offline_fallback_and_cooldown(monkeypatch):
    primary = FakeSynth("edge", fail_with=SynthesisError("offline", network=True))
    offline = FakeSynth("offline")
    speaker = make_tts(primary, offline)
    assert speaker.speak("First reply here. Second sentence of it.") is True
    assert len(primary.texts) == 1        # edge marked down after the first network failure
    assert len(offline.texts) == 1         # both sentences were merged into one chunk
    assert speaker.speak("Another reply.") is True
    assert len(primary.texts) == 1        # still in cooldown
    monkeypatch.setattr(tts_module, "EDGE_RETRY_AFTER_SECONDS", 0)
    speaker._primary_down_until = 0
    speaker.speak("And again.")
    assert len(primary.texts) == 2


def test_non_network_error_tries_offline_without_cooldown():
    primary = FakeSynth("edge", fail_with=SynthesisError("bad voice"))
    offline = FakeSynth("offline")
    speaker = make_tts(primary, offline)
    speaker.speak("One.")
    speaker.speak("Two.")
    assert len(primary.texts) == 2 and len(offline.texts) == 2


def test_total_synthesis_failure_returns_false():
    speaker = make_tts(FakeSynth(fail_with=SynthesisError("x")), offline=None, offline_fallback=False)
    assert speaker.speak("Hello there.") is False
    assert "x" in speaker.last_error


def test_stop_interrupts_quickly():
    speaker = make_tts(FakeSynth(), player=FakePlayer(block_seconds=2.0))
    threading.Timer(0.2, speaker.stop).start()
    started = time.monotonic()
    assert speaker.speak("A long sentence number one. Another long sentence number two. And three.") is False
    assert time.monotonic() - started < 1.5
    # The next utterance works normally again.
    speaker.player.block_seconds = 0
    assert speaker.speak("Back again.") is True


def test_playback_device_error_is_contained():
    speaker = make_tts(player=FakePlayer(error=AudioDeviceError("The speakers stopped working")))
    assert speaker.speak("Hello.") is False and "speakers" in speaker.last_error


def test_disabled_and_empty():
    primary = FakeSynth()
    assert make_tts(primary, enabled=False).speak("hi") is False
    assert make_tts(primary).speak("   ") is True
    assert primary.texts == []
