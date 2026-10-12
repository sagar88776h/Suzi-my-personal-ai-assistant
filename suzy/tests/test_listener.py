"""Tests for always-on listening: VAD, segmenting, wake word and the listener's modes."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core import listener as listener_module
from core.listener import (
    FRAME,
    EnergyVAD,
    ListenerRecorder,
    ListenerSettings,
    Mode,
    SegmenterSettings,
    SileroVAD,
    SpeechSegmenter,
    WakeListener,
    WakeResult,
    WakeWordDetector,
    create_vad,
    find_wake,
    has_request,
    strip_wake,
)
from core.stt import STTError, Transcript
from tests.fake_sounddevice import install_fake_sounddevice

RATE = 16_000


# ============================================================================ wake text
@pytest.mark.parametrize("text", [
    "Suzy", "Hi Suzy.", "Hello Suzy!", "Hey Susie, open Notepad", "hii suzie", "OK Suzi what's the time",
    "सूज़ी, volume kam karo", "Suzy?",
])
def test_wake_phrases_found(text):
    assert find_wake(text)


@pytest.mark.parametrize("text", ["I am busy", "Lucy is here", "sushi for dinner", "suzuki car", ""])
def test_not_wake_phrases(text):
    assert not find_wake(text)


@pytest.mark.parametrize("text, expected", [
    ("Hey Suzy, open Notepad.", "open Notepad"),
    ("Hello Suzy", ""),
    ("Suzy stop", "stop"),
    ("Okay Susie... what's the weather?", "what's the weather"),
    ("open chrome", "open chrome"),  # no wake word: unchanged
])
def test_strip_wake(text, expected):
    assert strip_wake(text) == expected


def test_has_request():
    assert has_request("open notepad") and has_request("stop")
    assert not has_request("") and not has_request("please") and not has_request("um, ji")


# ============================================================================ VAD
def test_silero_rejects_noise_and_tones():
    vad = SileroVAD()
    rng = np.random.default_rng(0)
    noise = [vad((rng.standard_normal(FRAME) * 0.01).astype(np.float32)) for _ in range(40)]
    t = np.arange(FRAME * 40) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(40, FRAME)
    vad.reset()
    tones = [vad(frame) for frame in tone]
    assert max(noise) < 0.5 and max(tones) < 0.5


def test_energy_vad_adapts_to_noise():
    vad = EnergyVAD(min_level=0.01)
    quiet = np.full(FRAME, 0.004, np.float32)
    loud = np.full(FRAME, 0.2, np.float32)
    assert vad(quiet) == 0.0 and vad(loud) == 1.0


def test_create_vad_falls_back(monkeypatch):
    def broken():
        raise RuntimeError("onnxruntime missing")

    monkeypatch.setattr(listener_module, "SileroVAD", broken)
    assert isinstance(create_vad(), EnergyVAD)


# ============================================================================ segmenting
def frames(seconds: float, level: float) -> list[tuple[np.ndarray, float]]:
    count = int(seconds * RATE / FRAME)
    return [(np.full(FRAME, level, np.float32), 1.0 if level > 0.05 else 0.0) for _ in range(count)]


def run_segmenter(segmenter, pieces):
    out, now = [], 100.0
    for frame, prob in pieces:
        now += FRAME / RATE
        result = segmenter.feed(frame, prob, now)
        if result is not None:
            out.append(result)
    return out


def test_segmenter_finds_utterances_with_pre_roll():
    seg = SpeechSegmenter(SegmenterSettings(end_silence=0.5, pre_roll=0.3))
    found = run_segmenter(seg, frames(1, 0) + frames(1.0, 0.2) + frames(1, 0) + frames(0.6, 0.2) + frames(1, 0))
    assert len(found) == 2
    first = len(found[0]) / RATE
    assert 1.6 <= first <= 2.0  # 0.3 pre-roll + 1.0 speech + 0.5 pause
    assert seg.last_started > 100


def test_segmenter_ignores_short_clicks():
    seg = SpeechSegmenter(SegmenterSettings(min_speech=0.25))
    assert run_segmenter(seg, frames(1, 0) + frames(0.15, 0.2) + frames(1, 0)) == []


def test_segmenter_max_length_and_flush():
    seg = SpeechSegmenter(SegmenterSettings(max_seconds=1.0))
    found = run_segmenter(seg, frames(2.5, 0.2))
    assert len(found) == 2 and all(len(f) / RATE <= 1.05 for f in found)
    run_segmenter(seg, frames(0.5, 0.2))
    assert seg.active and seg.flush() is not None and not seg.active
    assert seg.flush() is None


# ============================================================================ listener
class FakeDetector:
    """Decides what was 'said' in each utterance from a script."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def detect(self, audio):
        self.calls += 1
        return self.results.pop(0) if self.results else WakeResult(False, "background chatter")


class LoudVAD:
    name = "test"

    def __call__(self, frame):
        return 1.0 if float(np.sqrt(np.mean(frame ** 2))) > 0.05 else 0.0


def signal(*parts: tuple[float, float]) -> np.ndarray:
    """parts = (seconds, level): level 0 = silence, >0 = 'speech'."""
    return np.concatenate([np.full(int(s * RATE), lvl, np.float32) for s, lvl in parts])


def start_listener(monkeypatch, mic, detector, speaking=lambda: False, spoken=lambda: "", **settings):
    install_fake_sounddevice(monkeypatch, mic_signal=mic, realtime=0.004)
    events = []
    defaults = dict(command_timeout=0.4, followup_seconds=0.6,
                    segment=SegmenterSettings(end_silence=0.3, min_speech=0.2))
    defaults.update(settings)
    lst = WakeListener(detector, on_event=lambda k, p: events.append((k, p)), vad=LoudVAD(),
                       is_speaking=speaking, spoken_text=spoken, settings=ListenerSettings(**defaults))
    lst.start()
    return lst, events


def wait_for(events, kind, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for k, payload in events:
            if k == kind:
                return payload
        time.sleep(0.01)
    raise AssertionError(f"no '{kind}' event; got {[k for k, _ in events]}")


def test_wake_and_request_in_one_breath(monkeypatch):
    detector = FakeDetector(WakeResult(True, "Hey Suzy, open Notepad", "open Notepad"))
    lst, events = start_listener(monkeypatch, signal((0.3, 0), (1.0, 0.2), (1.0, 0)), detector)
    try:
        command = wait_for(events, "command")
        assert command.via == "wake" and command.wake_text == "Hey Suzy, open Notepad"
        assert lst.mode is Mode.WAKE
    finally:
        lst.stop()


def test_name_only_then_request(monkeypatch):
    detector = FakeDetector(WakeResult(True, "Suzy", ""))
    mic = signal((0.3, 0), (0.5, 0.2), (0.4, 0), (0.8, 0.2), (1.0, 0))
    lst, events = start_listener(monkeypatch, mic, detector, command_timeout=2)
    try:
        assert wait_for(events, "wake") == "Suzy"
        command = wait_for(events, "command")
        assert command.via == "wake" and command.wake_text == "" and detector.calls == 1
    finally:
        lst.stop()


def test_name_only_then_silence_times_out(monkeypatch):
    lst, events = start_listener(monkeypatch, signal((0.3, 0), (0.5, 0.2), (3, 0)),
                                 FakeDetector(WakeResult(True, "Hi Suzy", "")))
    try:
        wait_for(events, "timeout")
        assert lst.mode is Mode.WAKE
    finally:
        lst.stop()


def test_speech_without_wake_word_is_ignored(monkeypatch):
    lst, events = start_listener(monkeypatch, signal((0.3, 0), (0.8, 0.2), (1.5, 0)), FakeDetector())
    try:
        time.sleep(1.5)
        assert [k for k, _ in events] == [] and lst.detector.calls == 1
    finally:
        lst.stop()


def test_followup_needs_no_wake_word(monkeypatch):
    # Real-time pacing, so speech start times are realistic.
    install_fake_sounddevice(monkeypatch, mic_signal=signal((0.3, 0), (0.8, 0.2), (1.0, 0)),
                             realtime=FRAME / RATE)
    events = []
    lst = WakeListener(FakeDetector(), on_event=lambda k, p: events.append((k, p)), vad=LoudVAD(),
                       settings=ListenerSettings(followup_seconds=3,
                                                 segment=SegmenterSettings(end_silence=0.3, min_speech=0.2)))
    lst.start_followup()  # Suzy just answered
    assert lst.mode is Mode.FOLLOWUP
    lst.start()
    try:
        command = wait_for(events, "command")
        assert command.via == "followup" and lst.detector.calls == 0  # no wake-word check needed
        assert lst.mode is Mode.WAKE
    finally:
        lst.stop()


def test_followup_expires(monkeypatch):
    lst, events = start_listener(monkeypatch, signal((0.2, 0)), FakeDetector(), followup_seconds=0.2)
    try:
        lst.start_followup()
        time.sleep(0.6)
        assert lst.mode is Mode.WAKE and events == []
    finally:
        lst.stop()


def test_followup_disabled(monkeypatch):
    lst, _ = start_listener(monkeypatch, signal((0.2, 0)), FakeDetector(), followup_seconds=0)
    try:
        lst.start_followup()
        assert lst.mode is Mode.WAKE
    finally:
        lst.stop()


def test_saying_suzy_interrupts_her(monkeypatch):
    detector = FakeDetector(WakeResult(True, "Suzy stop", "stop"))
    lst, events = start_listener(monkeypatch, signal((0.2, 0), (0.6, 0.2), (1, 0)), detector,
                                 speaking=lambda: True, spoken=lambda: "Here's a long story about dragons.")
    try:
        wait_for(events, "command")
        assert [k for k, _ in events][:2] == ["interrupt", "command"]
    finally:
        lst.stop()


def test_her_own_voice_saying_suzy_is_ignored(monkeypatch):
    detector = FakeDetector(WakeResult(True, "Suzy here, what can I do", "what can I do"))
    lst, events = start_listener(monkeypatch, signal((0.2, 0), (0.6, 0.2), (1, 0)), detector,
                                 speaking=lambda: True,
                                 spoken=lambda: "Good evening, Sagar. Suzy here - what can I do for you?")
    try:
        time.sleep(1.5)
        assert events == [] and detector.calls == 1
    finally:
        lst.stop()


def test_push_to_talk_through_shared_microphone(monkeypatch):
    mic = signal((0.4, 0), (0.8, 0.2), (1.0, 0))
    lst, events = start_listener(monkeypatch, mic, FakeDetector())
    recorder = ListenerRecorder(lst, no_speech_timeout=2)
    try:
        audio = recorder.record()
        assert audio is not None and 0.9 <= len(audio) / RATE <= 1.6
        assert lst.detector.calls == 0 and events == []  # captured audio is not a wake check
        assert recorder.last_peak > 0.1 and not recorder.is_recording
    finally:
        lst.stop()


def test_push_to_talk_nothing_said(monkeypatch):
    lst, _ = start_listener(monkeypatch, signal((0.2, 0)), FakeDetector())
    try:
        assert ListenerRecorder(lst, no_speech_timeout=0.3).record() is None
    finally:
        lst.stop()


def test_push_to_talk_stop_early(monkeypatch):
    lst, _ = start_listener(monkeypatch, signal((0.2, 0), (20, 0.2)), FakeDetector())
    recorder = ListenerRecorder(lst, no_speech_timeout=2)
    try:
        threading.Timer(0.6, recorder.stop).start()
        started = time.monotonic()
        audio = recorder.record()
        assert audio is not None and time.monotonic() - started < 2
    finally:
        lst.stop()


def test_capture_when_not_running():
    lst = WakeListener(FakeDetector(), on_event=lambda k, p: None, vad=LoudVAD())
    with pytest.raises(Exception, match="isn't listening"):
        lst.capture(0.1)


def test_microphone_failure_is_reported(monkeypatch):
    install_fake_sounddevice(monkeypatch, fail_open_input=True)
    events = []
    lst = WakeListener(FakeDetector(), on_event=lambda k, p: events.append((k, p)), vad=LoudVAD())
    lst.start()
    assert "microphone stopped working" in wait_for(events, "error")
    assert not lst.running


def test_handler_errors_do_not_stop_listening(monkeypatch):
    def broken(kind, payload):
        raise RuntimeError("UI bug")

    install_fake_sounddevice(monkeypatch, mic_signal=signal((0.2, 0), (0.6, 0.2), (0.6, 0)), realtime=0.004)
    lst = WakeListener(FakeDetector(WakeResult(True, "Suzy hi", "hi")), on_event=broken, vad=LoudVAD(),
                       settings=ListenerSettings(segment=SegmenterSettings(end_silence=0.3)))
    lst.start()
    try:
        time.sleep(1.5)
        assert lst.running
    finally:
        lst.stop()


# ============================================================================ detector
class FakeWakeSTT:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def transcribe(self, audio):
        if self.error:
            raise self.error
        return Transcript(self.text, "en")


def test_detector_results():
    audio = np.zeros(16_000, np.float32)
    assert WakeWordDetector(FakeWakeSTT("Hey Suzy, open Notepad.")).detect(audio) == \
        WakeResult(True, "Hey Suzy, open Notepad.", "open Notepad")
    assert WakeWordDetector(FakeWakeSTT("Hello Suzy")).detect(audio) == WakeResult(True, "Hello Suzy", "")
    assert not WakeWordDetector(FakeWakeSTT("pass the salt")).detect(audio).found
    assert not WakeWordDetector(FakeWakeSTT(error=STTError("boom"))).detect(audio).found
