"""Voice tools: Suzy changes how she sounds (voice, speed, pitch) when asked."""

from __future__ import annotations

import re
from typing import Any

from tools.registry import ToolResult, tool

# Set by main.py at startup.
TTS: Any = None
CONFIG: Any = None

# Natural-sounding female voices that suit Suzy (Microsoft Edge neural voices, free).
RECOMMENDED_VOICES = {
    "en-IN-NeerjaExpressiveNeural": "Indian English, expressive and warm (default)",
    "en-IN-NeerjaNeural": "Indian English, calm and clear",
    "hi-IN-SwaraNeural": "Hindi, best for Hinglish",
    "en-US-AvaMultilingualNeural": "very natural, speaks English and Hindi (American accent)",
    "en-US-EmmaMultilingualNeural": "very natural, friendly (American accent)",
    "en-GB-SoniaNeural": "British English",
}
VOICE_NAME = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-[A-Za-z]+Neural$")


def configure(tts: Any, config: Any) -> None:
    global TTS, CONFIG
    TTS, CONFIG = tts, config


def _resolve(name: str) -> str | None:
    """Accept a full voice name or a friendly one like "Swara" or "neerja expressive"."""
    name = name.strip()
    if VOICE_NAME.match(name):
        return name
    wanted = re.sub(r"[^a-z]", "", name.lower())
    for voice in RECOMMENDED_VOICES:
        if wanted and wanted in voice.lower().replace("-", ""):
            return voice
    return None


@tool(
    description=(
        "Change how you (Suzy) sound: your voice, speaking speed and/or pitch. Voices: "
        + "; ".join(f"{name} = {about}" for name, about in RECOMMENDED_VOICES.items())
        + ". speed and pitch are changes from normal, e.g. speed 15 = 15% faster, -10 = slower."
    ),
    parameters={
        "type": "object",
        "properties": {
            "voice": {"type": "string", "description": "Voice name, e.g. 'hi-IN-SwaraNeural' or 'Swara'."},
            "speed": {"type": "integer", "description": "Percent faster (+) or slower (-), -50 to 100."},
            "pitch": {"type": "integer", "description": "Pitch change in Hz, -20 to 20."},
        },
    },
    category="voice",
)
def change_voice(voice: str | None = None, speed: int | None = None, pitch: int | None = None) -> ToolResult:
    if TTS is None:
        return ToolResult.fail("My voice settings aren't available right now.")
    if voice is None and speed is None and pitch is None:
        return ToolResult.fail("Tell me which voice, or how much faster, slower, higher or lower.")
    chosen = None
    if voice is not None:
        chosen = _resolve(voice)
        if chosen is None:
            return ToolResult.fail(f"I don't know a voice called {voice}. Try Neerja, Swara or Ava.")
    rate = f"{max(-50, min(100, speed)):+d}%" if speed is not None else None
    hz = f"{max(-20, min(20, pitch)):+d}Hz" if pitch is not None else None
    TTS.update_voice(chosen, rate, hz)
    if CONFIG is not None:
        if chosen:
            CONFIG.persist("tts.voice", chosen)
        if rate is not None:
            CONFIG.persist("tts.rate", rate)
        if hz is not None:
            CONFIG.persist("tts.pitch", hz)
    return ToolResult.ok("Done - how do I sound now?", voice=TTS.settings.voice, rate=TTS.settings.rate,
                         pitch=TTS.settings.pitch)
