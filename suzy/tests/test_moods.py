"""Unit tests for conversation moods."""

from __future__ import annotations

from core.moods import Mood, get_current_mood, get_mood_instructions, set_current_mood
from tools import mood_tools


def test_mood_transitions():
    assert set_current_mood("focus") == Mood.FOCUS
    assert get_current_mood() == Mood.FOCUS
    assert "COMMUNICATION STYLE: FOCUS" in get_mood_instructions()

    assert set_current_mood("professional") == Mood.PROFESSIONAL
    assert "COMMUNICATION STYLE: PROFESSIONAL" in get_mood_instructions()

    assert set_current_mood("natural") == Mood.NATURAL
    assert "COMMUNICATION STYLE: NATURAL" in get_mood_instructions()


def test_mood_fuzzy_matching():
    assert set_current_mood("be brief and short") == Mood.FOCUS
    assert set_current_mood("business formal") == Mood.PROFESSIONAL
    assert set_current_mood("silent") == Mood.QUIET


def test_mood_tools():
    res = mood_tools.set_mood("coach")
    assert res.success
    assert "Switched to coach mode" in res.message
    assert res.data["mood"] == "coach"

    res_get = mood_tools.get_mood()
    assert res_get.success
    assert "coach" in res_get.message
