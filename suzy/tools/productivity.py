"""Productivity tools: telling the time and date."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from tools.registry import ToolResult, tool

# Indirection so tests can freeze the clock.
now: Callable[[], datetime] = datetime.now


def format_time(moment: datetime) -> str:
    """12-hour time without a leading zero, e.g. ``"9:05 PM"``."""
    hour = moment.hour % 12 or 12
    suffix = "AM" if moment.hour < 12 else "PM"
    return f"{hour}:{moment.minute:02d} {suffix}"


def format_date(moment: datetime) -> str:
    """Spoken-friendly date, e.g. ``"Monday, 12 October 2026"``."""
    return f"{moment:%A}, {moment.day} {moment:%B %Y}"


@tool(description="Tell the current local time.", category="productivity")
def tell_time() -> ToolResult:
    moment = now()
    return ToolResult.ok(f"It's {format_time(moment)}.", time=moment.strftime("%H:%M"))


@tool(description="Tell today's date and day of the week.", category="productivity")
def tell_date() -> ToolResult:
    moment = now()
    return ToolResult.ok(f"Today is {format_date(moment)}.", date=moment.date().isoformat())
