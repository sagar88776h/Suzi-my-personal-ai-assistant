"""Tests for microphone capture, playback and device checks (fake sound card)."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from core import audio
from core.audio import (
    AudioDeviceError,
    Player,
    Recorder,
    RecordingSettings,
    check_devices,
    rms_level,
    tone,
)
from tests.fake_sounddevice import install_fake_sounddevice, speech_signal


def test_rms_level():
    assert rms_level(np.zeros(100, np.float32)) == 0.0
    assert rms_level(np.array([], np.float32)) == 0.0
    assert rms_level(np.full(100, 0.5, np.float32)) == pytest.approx(0.5)


def test_tone_has_soft_edges():
    beep = tone(440, 0.1)
    assert beep.dtype == np.float32 and len(beep) == 2400
    assert abs(beep[0]) < 1e-6 and abs(beep[-1]) < 0.01


# ----------------------------------------------------------------- devices
def test_check_devices_ok(monkeypatch):
    install_fake_sounddevice(monkeypatch)
    report = check_devices()
    assert (report.microphone, report.speaker, report.problems) == ("Fake Mic", "Fake Speakers", [])


def test_check_devices_missing_mic(monkeypatch):
    install_fake_sounddevice(monkeypatch, input_device=None)
    report = check_devices()
    assert report.microphone is None and "microphone" in report.problems[0]


def test_check_devices_without_portaudio(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import raises ImportError
    report = check_devices()
    assert report.microphone is None and report.speaker is None
    assert "audio system" in report.problems[0]


# ----------------------------------------------------------------- recording
def fast_settings(**overrides) -> RecordingSettings:
    values = dict(max_seconds=5, silence_seconds=0.6, no_speech_timeout=2, min_speech_seconds=0.2)
    values.update(overrides)
    return RecordingSettings(**values)


def test_records_until_pause(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 1.0, 3.0))
    levels: list[float] = []
    captured = Recorder(fast_settings()).record(level_callback=levels.append)
    assert captured is not None
    seconds = len(captured) / audio.SAMPLE_RATE
    # 0.5 s lead-in + 1 s speech + ~0.6 s pause, not the full 4.5 s signal.
    assert 1.8 <= seconds <= 2.6
    assert max(levels) > 0.1


def test_returns_none_when_nothing_said(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 0.0, 0.0))
    assert Recorder(fast_settings(no_speech_timeout=0.5)).record() is None


def test_ignores_short_clicks(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 0.05, 1.0))
    assert Recorder(fast_settings(no_speech_timeout=0.8)).record() is None


def test_stop_ends_recording_early(monkeypatch):
    # Real-time pacing: one 30 ms block every 30 ms, with endless "speech".
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.3, 30.0, 0.0), realtime=audio.BLOCK_SECONDS)
    recorder = Recorder(fast_settings(max_seconds=20))
    threading.Timer(0.8, recorder.stop).start()
    started = time.monotonic()
    captured = recorder.record()
    assert time.monotonic() - started < 3
    assert captured is not None and not recorder.is_recording


def test_max_length(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.3, 30.0, 0.0), realtime=audio.BLOCK_SECONDS)
    captured = Recorder(fast_settings(max_seconds=0.8)).record()
    assert captured is not None and len(captured) / audio.SAMPLE_RATE < 2


def test_microphone_open_failure(monkeypatch):
    install_fake_sounddevice(monkeypatch, fail_open_input=True)
    with pytest.raises(AudioDeviceError, match="couldn't open the microphone"):
        Recorder(fast_settings()).record()


# ----------------------------------------------------------------- playback
def test_player_plays_everything(monkeypatch):
    fake = install_fake_sounddevice(monkeypatch)
    sound = tone(440, 0.2)
    levels: list[float] = []
    assert Player().play(sound, 24_000, level_callback=levels.append) is True
    played = fake.played[0]
    np.testing.assert_allclose(played[:len(sound)], sound)
    assert max(levels) > 0.05 and levels[-1] == 0.0


def test_player_can_be_stopped(monkeypatch):
    install_fake_sounddevice(monkeypatch, realtime=0.01)
    player = Player()
    threading.Timer(0.1, player.stop).start()
    started = time.monotonic()
    assert player.play(tone(440, 5.0), 24_000) is False
    assert time.monotonic() - started < 1.5
    assert player.play(tone(440, 0.1), 24_000) is False  # stays stopped until reset()
    player.reset()
    assert player.play(tone(440, 0.05), 24_000) is True


def test_player_empty_and_device_errors(monkeypatch):
    install_fake_sounddevice(monkeypatch, fail_open_output=True)
    assert Player().play(np.zeros(0, np.float32), 24_000) is True
    with pytest.raises(AudioDeviceError, match="speakers"):
        Player().play(tone(440, 0.1), 24_000)


def test_play_cue_never_raises(monkeypatch):
    install_fake_sounddevice(monkeypatch, fail_open_output=True)
    audio.play_cue("start")
    audio.play_cue("stop")
