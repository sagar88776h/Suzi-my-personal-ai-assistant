"""Tools for switching and inspecting conversation moods."""

from __future__ import annotations

from core.moods import MOOD_DESCRIPTIONS, Mood, get_current_mood, set_current_mood
from tools.registry import ToolResult, tool
from tools.safety import SafetyLevel


@tool(
    description="Change the conversation mood and personality style (natural, professional, focus, coach, creative, quiet).",
    parameters={
        "type": "object",
        "properties": {
            "mood": {
                "type": "string",
                "enum": ["natural", "professional", "focus", "coach", "creative", "quiet"],
                "description": "The desired conversation mood/tone.",
            },
        },
        "required": ["mood"],
    },
    safety=SafetyLevel.SAFE,
    category="personality",
)
def set_mood(mood: str) -> ToolResult:
    new_mood = set_current_mood(mood)
    desc = MOOD_DESCRIPTIONS.get(new_mood, "")
    return ToolResult.ok(
        f"Switched to {new_mood.value} mode. {desc}",
        mood=new_mood.value,
        description=desc,
    )


@tool(
    description="Check the currently active conversation mood and style.",
    safety=SafetyLevel.SAFE,
    category="personality",
)
def get_mood() -> ToolResult:
    current = get_current_mood()
    desc = MOOD_DESCRIPTIONS.get(current, "")
    return ToolResult.ok(
        f"Current mode is {current.value}. {desc}",
        mood=current.value,
        description=desc,
    )
