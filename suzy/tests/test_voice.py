"""Tests for the voice conversation loop with fake ears, voice and brain."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.assistant import AssistantReply
from core.audio import AudioDeviceError
from core.stt import STTError, Transcript
from core.tts import TTSSettings
from core.voice import VoiceSession


class FakeAssistant:
    def __init__(self):
        self.requests = []

    def process(self, text):
        self.requests.append(text)
        return AssistantReply(f"Reply to {text}")


class FakeTTS:
    def __init__(self, speak_seconds=0.0):
        self.settings = TTSSettings()
        self.spoken, self.speak_seconds = [], speak_seconds
        self._stop = threading.Event()
        self._speaking = threading.Event()

    @property
    def is_speaking(self):
        return self._speaking.is_set()

    def speak(self, text, level_callback=None):
        self._stop.clear()
        self._speaking.set()
        try:
            interrupted = self._stop.wait(self.speak_seconds)
            self.spoken.append((text, not interrupted))
            return not interrupted
        finally:
            self._speaking.clear()

    def stop(self):
        self._stop.set()


class FakeSTT:
    def __init__(self, text="open notepad", ready=True, load_error=None, error=None):
        self.text, self.ready, self.load_error, self.error = text, ready, load_error, error
        self.received = []

    @property
    def is_ready(self):
        return self.ready

    def wait_until_loaded(self, timeout=None):
        return self.ready

    def transcribe(self, audio):
        self.received.append(audio)
        if self.error:
            raise self.error
        return Transcript(self.text, "en")


SPEECH = np.ones(16_000, np.float32)


class FakeRecorder:
    def __init__(self, audio=SPEECH, error=None, hold_seconds=0.0):
        self.audio, self.error, self.hold_seconds = audio, error, hold_seconds
        self._stop = threading.Event()
        self._recording = threading.Event()
        self.calls = 0

    @property
    def is_recording(self):
        return self._recording.is_set()

    def stop(self):
        self._stop.set()

    def record(self, level_callback=None):
        self.calls += 1
        if self.error:
            raise self.error
        self._stop.clear()
        self._recording.set()
        try:
            self._stop.wait(self.hold_seconds)
            return self.audio
        finally:
            self._recording.clear()


def make_session(stt=None, recorder=None, tts=None, assistant=None):
    events = []
    session = VoiceSession(
        assistant or FakeAssistant(), tts or FakeTTS(), stt if stt is not None else FakeSTT(),
        recorder if recorder is not None else FakeRecorder(),
        on_event=lambda kind, payload: events.append((kind, payload)), cues=False,
    )
    return session, events


def kinds(events):
    return [k for k, _ in events]


def test_full_voice_turn():
    session, events = make_session()
    reply = session.listen_and_respond()
    assert reply.text == "Reply to open notepad"
    assert kinds(events) == ["listening", "transcribing", "heard", "thinking", "reply", "speaking", "idle"]
    assert session.tts.spoken == [("Reply to open notepad", True)]
    assert session.assistant.requests == ["open notepad"]


def test_nothing_heard():
    session, events = make_session(recorder=FakeRecorder(audio=None))
    assert session.listen_and_respond() is None
    assert ("info", "I didn't hear anything.") in events
    assert session.assistant.requests == [] and session.tts.spoken == []


def test_unclear_speech():
    session, events = make_session(stt=FakeSTT(text=""))
    assert session.listen_and_respond() is None
    assert session.tts.spoken == [("Sorry, I didn't catch that.", True)]


def test_microphone_failure_is_spoken():
    session, events = make_session(recorder=FakeRecorder(error=AudioDeviceError("I couldn't open the microphone.")))
    assert session.listen_and_respond() is None
    assert ("error", "I couldn't open the microphone.") in events
    assert session.tts.spoken[0][0] == "I couldn't open the microphone."


def test_stt_failure():
    session, _ = make_session(stt=FakeSTT(error=STTError("Speech recognition hit a problem.")))
    assert session.listen_and_respond() is None
    assert session.tts.spoken[0][0] == "Speech recognition hit a problem."


def test_model_failed_to_load():
    session, events = make_session(stt=FakeSTT(ready=False, load_error="I couldn't download my speech model."))
    assert session.listen_and_respond() is None
    assert session.recorder.calls == 0
    assert ("error", "I couldn't download my speech model.") in events


def test_model_still_loading_waits():
    stt = FakeSTT(ready=False)
    stt.wait_until_loaded = lambda timeout=None: setattr(stt, "ready", True) or True
    session, events = make_session(stt=stt)
    assert session.listen_and_respond() is not None
    assert events[0][0] == "info" and "loading" in events[0][1]


def test_voice_disabled():
    session = VoiceSession(FakeAssistant(), FakeTTS(), None, None, cues=False)
    assert not session.can_listen and session.listen_and_respond() is None


def test_typed_text_turn_and_mute():
    session, events = make_session()
    session.handle_text("what time is it")
    assert kinds(events) == ["thinking", "reply", "speaking", "idle"]
    session.tts.settings.enabled = False
    session.handle_text("again")
    assert len(session.tts.spoken) == 1  # muted: printed but not spoken


def test_only_one_turn_at_a_time():
    session, _ = make_session(recorder=FakeRecorder(hold_seconds=0.5))
    worker = threading.Thread(target=session.listen_and_respond)
    worker.start()
    time.sleep(0.1)
    assert session.busy
    assert session.listen_and_respond(wait=False) is None  # rejected while busy
    worker.join()
    assert session.recorder.calls == 1


def test_hotkey_stops_listening_early():
    session, _ = make_session(recorder=FakeRecorder(hold_seconds=10))
    worker = threading.Thread(target=session.listen_and_respond)
    worker.start()
    deadline = time.monotonic() + 2
    while not session.recorder.is_recording and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    session.hotkey_pressed()  # second press = "I'm done talking"
    worker.join(3)
    assert time.monotonic() - started < 1.5 and session.assistant.requests == ["open notepad"]


def test_hotkey_interrupts_speech_and_listens():
    tts = FakeTTS(speak_seconds=10)
    session, _ = make_session(tts=tts)
    talking = threading.Thread(target=session.handle_text, args=("tell me a long story",))
    talking.start()
    deadline = time.monotonic() + 2
    while not tts.is_speaking and time.monotonic() < deadline:
        time.sleep(0.01)
    tts.speak_seconds = 0
    started = time.monotonic()
    session.hotkey_pressed()  # stops the story, then listens
    talking.join(3)
    assert time.monotonic() - started < 2
    assert tts.spoken[0] == ("Reply to tell me a long story", False)  # interrupted
    assert session.assistant.requests == ["tell me a long story", "open notepad"]


def test_broken_event_handler_does_not_break_turn():
    def explode(kind, payload):
        raise RuntimeError("UI bug")

    session = VoiceSession(FakeAssistant(), FakeTTS(), FakeSTT(), FakeRecorder(), on_event=explode, cues=False)
    assert session.listen_and_respond().text == "Reply to open notepad"


@pytest.mark.parametrize("enabled", [True, False])
def test_say_async(enabled):
    session, _ = make_session()
    session.tts.settings.enabled = enabled
    session.say_async("Hello Sagar").join(2)
    assert (session.tts.spoken == [("Hello Sagar", True)]) is enabled
