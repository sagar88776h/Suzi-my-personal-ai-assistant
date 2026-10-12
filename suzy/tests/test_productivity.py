"""Tests for time and date tools (clock frozen)."""

from __future__ import annotations

from datetime import datetime

import pytest

from tools import productivity


@pytest.mark.parametrize(
    "moment, text",
    [
        (datetime(2026, 10, 12, 0, 5), "12:05 AM"),
        (datetime(2026, 10, 12, 9, 0), "9:00 AM"),
        (datetime(2026, 10, 12, 12, 30), "12:30 PM"),
        (datetime(2026, 10, 12, 21, 7), "9:07 PM"),
    ],
)
def test_format_time(moment, text):
    assert productivity.format_time(moment) == text


def test_tell_time(monkeypatch):
    monkeypatch.setattr(productivity, "now", lambda: datetime(2026, 10, 12, 15, 45))
    result = productivity.tell_time()
    assert result.success and result.message == "It's 3:45 PM." and result.data["time"] == "15:45"


def test_tell_date(monkeypatch):
    monkeypatch.setattr(productivity, "now", lambda: datetime(2026, 10, 2, 8, 0))
    result = productivity.tell_date()
    assert result.message == "Today is Friday, 2 October 2026."
    assert result.data["date"] == "2026-10-02"
