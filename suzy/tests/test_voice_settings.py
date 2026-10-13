"""Tests for changing Suzy's voice, saving settings, and the voice preview."""

from __future__ import annotations

import numpy as np
import pytest

import main
from core.config import Config, replace_yaml_value
from core.tts import EdgeSynthesizer, SynthesisError, TextToSpeech, TTSSettings
from tools import voice_settings

YAML = """user:
  name: Sagar             # how Suzy addresses you

tts:                      # Suzy's voice
  enabled: true
  voice: en-IN-NeerjaNeural      # pick with --voices
  rate: "+0%"             # speed
  nested:
    deep: 1
"""


# ----------------------------------------------------------------- config persistence
def test_replace_yaml_value_keeps_comments():
    out = replace_yaml_value(YAML, ["tts", "voice"], "hi-IN-SwaraNeural")
    old_line = next(line for line in YAML.splitlines() if "voice:" in line and "#" in line)
    new_line = next(line for line in out.splitlines() if "voice:" in line and "#" in line)
    assert new_line.startswith("  voice: hi-IN-SwaraNeural ") and new_line.endswith("# pick with --voices")
    assert new_line.index("#") == old_line.index("#")  # comment stays in the same column
    out = replace_yaml_value(out, ["tts", "rate"], "+15%")
    assert "  rate: +15%" in out and "# speed" in out
    assert replace_yaml_value(YAML, ["tts", "nested", "deep"], 2).count("deep: 2") == 1
    assert replace_yaml_value(YAML, ["tts", "nested"], 5) is None      # a section, not a value
    assert replace_yaml_value(YAML, ["tts", "missing"], 5) is None


def test_persist_writes_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(YAML, encoding="utf-8")
    config = Config.load(path, tmp_path / ".env")
    config.persist("tts.voice", "hi-IN-SwaraNeural")
    config.persist("tts.pitch", "+2Hz")  # not in the file yet -> whole-file save
    reloaded = Config.load(path, tmp_path / ".env")
    assert reloaded.get("tts.voice") == "hi-IN-SwaraNeural" and reloaded.get("tts.pitch") == "+2Hz"
    assert config.get("tts.voice") == "hi-IN-SwaraNeural"


# ----------------------------------------------------------------- change_voice tool
@pytest.fixture()
def configured(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(YAML, encoding="utf-8")
    config = Config.load(path, tmp_path / ".env")
    tts = TextToSpeech(TTSSettings(voice="en-IN-NeerjaNeural"))
    monkeypatch.setattr(voice_settings, "TTS", tts)
    monkeypatch.setattr(voice_settings, "CONFIG", config)
    return tts, path


def test_change_voice_by_friendly_name(configured):
    tts, path = configured
    result = voice_settings.change_voice(voice="Swara")
    assert result.success and tts.settings.voice == "hi-IN-SwaraNeural"
    assert tts.primary.voice == "hi-IN-SwaraNeural"
    assert "voice: hi-IN-SwaraNeural" in path.read_text(encoding="utf-8")


def test_change_speed_and_pitch(configured):
    tts, path = configured
    assert voice_settings.change_voice(speed=150, pitch=-5).success
    assert tts.settings.rate == "+100%" and tts.settings.pitch == "-5Hz"  # speed clamped to 100
    assert tts.primary.rate == "+100%"


def test_change_voice_errors(configured):
    assert not voice_settings.change_voice().success
    assert "don't know a voice" in voice_settings.change_voice(voice="Bob the robot").message
    assert voice_settings.change_voice(voice="en-AU-NatashaNeural").success  # any valid full name


def test_change_voice_unavailable(monkeypatch):
    monkeypatch.setattr(voice_settings, "TTS", None)
    assert not voice_settings.change_voice(voice="Swara").success


# ----------------------------------------------------------------- TTS notice
class Offline:
    name = "edge"

    def synthesize(self, text):
        raise SynthesisError("can't reach", network=True)


class Windows:
    name = "offline"

    def synthesize(self, text):
        return np.zeros(100, np.float32), 24_000


def test_robotic_voice_notice_is_given_once():
    from tests.test_tts import FakePlayer

    notices = []
    tts = TextToSpeech(TTSSettings(), player=FakePlayer(), primary=Offline(), offline=Windows())
    tts.on_notice = notices.append
    tts.speak("One.")
    tts._primary_down_until = 0
    tts.speak("Two.")
    assert len(notices) == 1 and "Windows voice" in notices[0]
    tts.update_voice(voice="hi-IN-SwaraNeural")
    assert tts._primary_down_until == 0.0


# ----------------------------------------------------------------- voice preview
def test_voice_preview_saves_choice(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(YAML, encoding="utf-8")
    config = Config.load(path, tmp_path / ".env")
    voices = [{"name": "en-IN-NeerjaExpressiveNeural", "locale": "en-IN"},
              {"name": "hi-IN-SwaraNeural", "locale": "hi-IN"}]
    played = []
    monkeypatch.setattr(main, "fetch_voice_choices", lambda: voices)
    monkeypatch.setattr(EdgeSynthesizer, "synthesize", lambda self, text: (np.zeros(10, np.float32), 24_000))
    monkeypatch.setattr(main.Player, "play", lambda self, samples, rate: played.append(rate) or True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    assert main.run_voice_preview(config) == 0
    assert len(played) == 2 and "Saved!" in capsys.readouterr().out
    assert "voice: hi-IN-SwaraNeural" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("answer, code", [("", 0), ("9", 1), ("abc", 1)])
def test_voice_preview_other_answers(tmp_path, monkeypatch, answer, code):
    config = Config({}, tmp_path / "config.yaml")
    monkeypatch.setattr(main, "fetch_voice_choices", lambda: [{"name": "hi-IN-SwaraNeural", "locale": "hi-IN"}])
    monkeypatch.setattr(EdgeSynthesizer, "synthesize", lambda self, text: (np.zeros(10, np.float32), 24_000))
    monkeypatch.setattr(main.Player, "play", lambda self, samples, rate: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert main.run_voice_preview(config) == code


def test_voice_preview_offline(monkeypatch, capsys):
    def offline():
        raise OSError("no internet")

    monkeypatch.setattr(main, "fetch_voice_choices", offline)
    assert main.run_voice_preview(Config()) == 1
    assert "couldn't get the voice list" in capsys.readouterr().out
