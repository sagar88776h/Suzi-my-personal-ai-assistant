"""Personality moods and communication styles for Suzy."""

from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger("suzy.moods")


class Mood(str, Enum):
    NATURAL = "natural"
    PROFESSIONAL = "professional"
    FOCUS = "focus"
    COACH = "coach"
    CREATIVE = "creative"
    QUIET = "quiet"


MOOD_DESCRIPTIONS: dict[Mood, str] = {
    Mood.NATURAL: "Friendly, conversational, calm, and naturally helpful.",
    Mood.PROFESSIONAL: "Crisp, structured, business-focused, and objective.",
    Mood.FOCUS: "Ultra-concise, minimal distraction, pure signal.",
    Mood.COACH: "Direct, honest, action-oriented, and motivating.",
    Mood.CREATIVE: "Imaginative, expressive, colorful, and engaging.",
    Mood.QUIET: "Quiet mode with minimal spoken output and muted style.",
}

MOOD_PROMPT_INSTRUCTIONS: dict[Mood, str] = {
    Mood.NATURAL: (
        "COMMUNICATION STYLE: NATURAL\n"
        "- Friendly, conversational, relaxed, and warm.\n"
        "- Speak naturally with everyday warmth while remaining efficient."
    ),
    Mood.PROFESSIONAL: (
        "COMMUNICATION STYLE: PROFESSIONAL\n"
        "- Crisp, structured, polite, and business-focused.\n"
        "- Emphasize clarity, actionable structure, and professional tone. Avoid slang."
    ),
    Mood.FOCUS: (
        "COMMUNICATION STYLE: FOCUS (MAXIMUM BREVITY)\n"
        "- Strictly 1 short, direct sentence whenever possible. No pleasantries or conversational filler.\n"
        "- Pure signal and instant answers."
    ),
    Mood.COACH: (
        "COMMUNICATION STYLE: COACH\n"
        "- Direct, constructive, action-oriented, and motivating.\n"
        "- Challenge excuses gently, highlight priorities, and suggest immediate next steps."
    ),
    Mood.CREATIVE: (
        "COMMUNICATION STYLE: CREATIVE\n"
        "- Imaginative, expressive, vivid, and engaging.\n"
        "- Feel free to offer analogies, innovative angles, and lively phrasing."
    ),
    Mood.QUIET: (
        "COMMUNICATION STYLE: QUIET\n"
        "- Muted spoken delivery. Keep verbal answers as short as 1-5 words or text-only.\n"
        "- Do not suppress critical error messages or safety alerts."
    ),
}

_CURRENT_MOOD: Mood = Mood.NATURAL


def get_current_mood() -> Mood:
    return _CURRENT_MOOD


def set_current_mood(mood_input: Mood | str) -> Mood:
    global _CURRENT_MOOD
    val = str(mood_input).strip().lower()
    for m in Mood:
        if m.value == val or m.name.lower() == val:
            _CURRENT_MOOD = m
            log.info("Switched mood to %s", m.value)
            return m
    # Try fuzzy matching common terms
    if "prof" in val or "formal" in val or "business" in val:
        _CURRENT_MOOD = Mood.PROFESSIONAL
    elif "focus" in val or "short" in val or "brief" in val or "concise" in val:
        _CURRENT_MOOD = Mood.FOCUS
    elif "coach" in val or "motivat" in val:
        _CURRENT_MOOD = Mood.COACH
    elif "creat" in val or "fun" in val or "imagin" in val:
        _CURRENT_MOOD = Mood.CREATIVE
    elif "quiet" in val or "silent" in val or "mute" in val:
        _CURRENT_MOOD = Mood.QUIET
    else:
        _CURRENT_MOOD = Mood.NATURAL

    log.info("Resolved and switched mood to %s", _CURRENT_MOOD.value)
    return _CURRENT_MOOD


def get_mood_instructions(mood: Mood | None = None) -> str:
    m = mood or _CURRENT_MOOD
    return MOOD_PROMPT_INSTRUCTIONS.get(m, MOOD_PROMPT_INSTRUCTIONS[Mood.NATURAL])
