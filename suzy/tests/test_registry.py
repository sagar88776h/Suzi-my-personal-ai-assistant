"""Tests for tool registration, argument validation and safe execution."""

from __future__ import annotations

import logging

import pytest

from tools.registry import (
    ToolArgumentError,
    ToolRegistry,
    ToolResult,
    tool,
    validate_arguments,
)
from tools.safety import SafetyLevel
from tools.winutils import PlatformNotSupportedError


@pytest.fixture()
def reg():
    registry = ToolRegistry()

    @tool(
        description="Add numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "number"},
                "loud": {"type": "boolean"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
            "required": ["a"],
        },
        registry=registry,
    )
    def add(a: int, b: float = 0, loud: bool = False, mode: str = "fast") -> ToolResult:
        return ToolResult.ok(f"{a + b}", total=a + b, loud=loud, mode=mode)

    @tool(description="Always explodes", registry=registry, safety=SafetyLevel.CONFIRM)
    def boom() -> ToolResult:
        raise RuntimeError("kaboom")

    @tool(description="Windows only", registry=registry)
    def winonly() -> ToolResult:
        raise PlatformNotSupportedError("Locking the screen")

    @tool(description="Returns a string", registry=registry)
    def plain() -> str:
        return "plain text"

    return registry


def test_registration_and_schema(reg):
    assert reg.names() == ["add", "boom", "plain", "winonly"]
    schema = reg.get("add").schema()
    assert schema["name"] == "add"
    assert schema["parameters"]["required"] == ["a"]
    assert reg.get("boom").safety is SafetyLevel.CONFIRM
    # Parameter-less tools still get a valid object schema.
    assert reg.get("boom").parameters == {"type": "object", "properties": {}}


def test_execute_success_and_coercion(reg):
    result = reg.execute("add", {"a": "5", "b": "2.5", "loud": "yes"})
    assert result.success
    assert result.data == {"total": 7.5, "loud": True, "mode": "fast"}


def test_float_to_integer_and_percent_strings(reg):
    assert reg.execute("add", {"a": 4.6}).data["total"] == 5
    assert reg.execute("add", {"a": "30%"}).data["total"] == 30


def test_missing_required_argument(reg):
    result = reg.execute("add", {})
    assert not result.success
    assert "Missing required" in result.message


def test_bad_types_and_enum(reg):
    assert not reg.execute("add", {"a": "lots"}).success
    assert not reg.execute("add", {"a": True}).success
    assert not reg.execute("add", {"a": 1, "loud": "maybe"}).success
    result = reg.execute("add", {"a": 1, "mode": "medium"})
    assert not result.success and "fast, slow" in result.message


def test_unknown_and_null_arguments_are_dropped(reg):
    result = reg.execute("add", {"a": 1, "colour": "red", "b": None})
    assert result.success and result.data["total"] == 1


def test_unknown_tool(reg):
    result = reg.execute("does_not_exist", {})
    assert not result.success and "does_not_exist" in result.message


def test_crashing_tool_is_contained(reg):
    result = reg.execute("boom")
    assert not result.success and "kaboom" in result.message


def test_platform_error_message(reg):
    result = reg.execute("winonly")
    assert not result.success and result.message == "Locking the screen only works on Windows."


def test_string_return_is_wrapped(reg):
    result = reg.execute("plain")
    assert result.success and result.message == "plain text"


def test_actions_are_logged(reg, caplog):
    logger = logging.getLogger("suzy.actions")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="suzy.actions"):
            reg.execute("add", {"a": 2})
            reg.execute("boom")
    finally:
        logger.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if r.name == "suzy.actions"]
    assert any(line.startswith("OK") and "add(" in line for line in lines)
    assert any(line.startswith("FAIL") and "boom(" in line for line in lines)


def test_validate_arguments_direct(reg):
    with pytest.raises(ToolArgumentError):
        validate_arguments(reg.get("add"), {"a": [1]})


def test_to_llm_payload():
    assert ToolResult.ok("hi").to_llm_payload() == {"success": True, "message": "hi"}
    assert ToolResult.fail("no", x=1).to_llm_payload() == {"success": False, "message": "no", "data": {"x": 1}}


def test_all_phase1_tools_registered(registry):
    for name in ("open_app", "set_volume", "tell_time", "tell_date", "google_search", "get_system_info"):
        tool_obj = registry.get(name)
        assert tool_obj is not None, name
        assert tool_obj.description
        assert tool_obj.parameters["type"] == "object"
