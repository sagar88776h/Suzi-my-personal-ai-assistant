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


def test_slow_consumer_does_not_append_backlog(monkeypatch):
    """Regression: on Windows the mic thread can run ahead of the recorder.
    Blocks queued after the pause must not be glued onto the recording."""
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 1.0, 3.0))
    captured = Recorder(fast_settings()).record(level_callback=lambda _level: time.sleep(0.002))
    assert captured is not None
    assert 1.8 <= len(captured) / audio.SAMPLE_RATE <= 2.6


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


# ----------------------------------------------------------------- diagnostics
def test_recorder_reports_peak_for_quiet_mic(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 1.0, 1.0, loudness=0.004))
    recorder = Recorder(fast_settings(no_speech_timeout=0.5))
    assert recorder.record() is None
    assert 0.002 < recorder.last_peak < recorder.last_threshold


def test_quiet_mic_is_now_heard_with_default_sensitivity(monkeypatch):
    """A laptop mic at RMS ~0.01 was ignored by the old 0.015 threshold."""
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.5, 1.0, 2.0, loudness=0.014))
    assert Recorder(RecordingSettings(silence_seconds=0.6, no_speech_timeout=2)).record() is not None


@pytest.mark.parametrize(
    "peak, phrase",
    [(0.0, "pure silence"), (0.002, "very quiet"), (0.05, "didn't hear anything.")],
)
def test_describe_quiet_recording(peak, phrase):
    assert phrase in audio.describe_quiet_recording(peak, 0.006)


def test_list_and_resolve_input_devices(monkeypatch):
    install_fake_sounddevice(monkeypatch)
    assert audio.list_input_devices() == [(0, "Fake Mic", True), (1, "USB Headset Microphone", False)]
    assert audio.resolve_input_device(None) is None
    assert audio.resolve_input_device("  ") is None
    assert audio.resolve_input_device(1) == 1
    assert audio.resolve_input_device("1") == 1
    assert audio.resolve_input_device("usb headset") == 1
    with pytest.raises(AudioDeviceError, match="Bluetooth"):
        audio.resolve_input_device("Bluetooth")


def test_measure_microphone(monkeypatch):
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.1, 0.5, 0.0), realtime=0.01)
    levels = []
    result = audio.measure_microphone(0.4, level_callback=levels.append)
    assert result.peak > 0.1 and 0 < result.average < result.peak and levels
    assert result.audio.dtype == np.float32 and len(result.audio) > 0


def test_measure_microphone_open_failure(monkeypatch):
    install_fake_sounddevice(monkeypatch, fail_open_input=True)
    with pytest.raises(AudioDeviceError):
        audio.measure_microphone(0.1)


def test_speaking_right_at_the_chime_is_still_heard(monkeypatch):
    """Regression: speech during the 0.3 s noise calibration made the threshold so high
    that the rest of the sentence counted as silence ("Suzy talks but doesn't listen")."""
    install_fake_sounddevice(monkeypatch, mic_signal=speech_signal(0.0, 1.5, 2.0, loudness=0.08))
    captured = Recorder(fast_settings()).record()
    assert captured is not None
    assert 1.8 <= len(captured) / audio.SAMPLE_RATE <= 2.8  # ends at the pause, not at max length


def test_loud_room_does_not_make_suzy_deaf(monkeypatch):
    """Even with a noisy calibration, the speech threshold is capped at a level normal speech reaches."""
    import numpy as np

    rng = np.random.default_rng(1)
    noisy = (rng.standard_normal(16_000 * 3) * 0.02).astype(np.float32)  # fan / traffic noise
    noisy[8_000:32_000] += speech_signal(0.0, 1.5, 0.0, loudness=0.1)[:24_000]
    install_fake_sounddevice(monkeypatch, mic_signal=noisy)
    recorder = Recorder(fast_settings(max_seconds=3))
    assert recorder.record() is not None and recorder.last_threshold <= 0.03


# ----------------------------------------------------------------- continuous playback
def _queue(*items):
    import queue as queue_module

    q = queue_module.Queue()
    for item in items:
        q.put(item)
    return q


def test_play_queue_is_one_continuous_stream(monkeypatch):
    fake = install_fake_sounddevice(monkeypatch)
    a, b = np.full(2400, 0.5, np.float32), np.full(2400, -0.5, np.float32)
    assert Player().play_queue(_queue((a, 24_000), (b, 24_000), None), 24_000) is True
    assert len(fake.played) == 1                       # ONE stream for both chunks
    played = fake.played[0]
    assert np.count_nonzero(played) >= 4700           # no silent gap between the chunks
    assert played[0] == 0.0 and abs(played[1199]) == pytest.approx(0.5)  # faded in, then full level


def test_play_queue_resamples_offline_voice(monkeypatch):
    fake = install_fake_sounddevice(monkeypatch)
    assert Player().play_queue(_queue((np.full(22_050, 0.3, np.float32), 22_050), None), 24_000)
    assert abs(np.count_nonzero(fake.played[0]) - 24_000) < 300


def test_play_queue_waits_for_slow_synthesis(monkeypatch):
    import queue as queue_module

    fake = install_fake_sounddevice(monkeypatch, realtime=0.005)
    q = queue_module.Queue()
    q.put((np.full(2400, 0.4, np.float32), 24_000))
    threading.Timer(0.2, lambda: (q.put((np.full(2400, 0.4, np.float32), 24_000)), q.put(None))).start()
    assert Player().play_queue(q, 24_000) is True
    assert np.count_nonzero(fake.played[0]) >= 4700   # both chunks played, silence in between


def test_play_queue_stop_and_stall(monkeypatch):
    import queue as queue_module

    install_fake_sounddevice(monkeypatch, realtime=0.005)
    player = Player()
    threading.Timer(0.15, player.stop).start()
    started = time.monotonic()
    assert player.play_queue(queue_module.Queue(), 24_000) is False   # nothing ever arrives; stopped
    assert time.monotonic() - started < 1.5
    player.reset()
    assert player.play_queue(queue_module.Queue(), 24_000, stall_seconds=0.2) is True  # gives up on a stall


def test_resample_and_smooth_edges():
    assert len(audio.resample(np.ones(22_050, np.float32), 22_050, 24_000)) == 24_000
    same = np.ones(10, np.float32)
    assert audio.resample(same, 24_000, 24_000) is same
    faded = audio.smooth_edges(np.ones(2400, np.float32), 24_000)
    assert faded[0] == 0.0 and faded[-1] == 0.0 and faded[1200] == 1.0
