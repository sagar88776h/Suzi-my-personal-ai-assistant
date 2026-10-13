"""Tests for the assistant loop using a scripted fake LLM and a fake registry."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.assistant import Assistant
from core.config import Config
from core.llm import LLMError, LLMErrorKind, LLMResponse, LLMUnavailableError, ToolCall
from tools.registry import ToolRegistry, ToolResult, tool


class FakeLLM:
    """Plays back scripted LLMResponses (or raises scripted errors)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.seen: list[list[dict]] = []
        self.prompts: list[str] = []

    def chat(self, messages, tools, system_prompt, options=None):
        self.options = options
        self.seen.append([dict(m) for m in messages])
        self.prompts.append(system_prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def text(reply):
    return LLMResponse(reply, [], "fake")


def calls(*tool_calls, extras=None):
    return LLMResponse("", [ToolCall(f"id{i}", n, a) for i, (n, a) in enumerate(tool_calls)], "fake",
                       message_extras=extras or {})


@pytest.fixture()
def reg():
    registry = ToolRegistry()

    @tool(description="open", registry=registry,
          parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    def open_app(name: str) -> ToolResult:
        return ToolResult.ok(f"Opening {name}.")

    @tool(description="time", registry=registry)
    def tell_time() -> ToolResult:
        return ToolResult.ok("It's 3:45 PM.")

    return registry


def make(llm, reg, quick=False, **config):
    """Most tests exercise the full tool loop, so quick replies are off unless asked for."""
    config.setdefault("assistant", {}).setdefault("quick_replies", quick)
    return Assistant(Config(config), llm, reg, clock=lambda: datetime(2026, 10, 12, 15, 45))


def test_plain_chat(reg):
    llm = FakeLLM(text("Hey Sagar!"))
    assistant = make(llm, reg)
    reply = assistant.process("hello")
    assert reply.text == "Hey Sagar!" and reply.actions == [] and not reply.error
    assert assistant.history == [{"role": "user", "content": "hello"},
                                 {"role": "assistant", "content": "Hey Sagar!"}]
    assert "Sagar" in llm.prompts[0] and "3:45 PM" in llm.prompts[0]


def test_tool_loop(reg):
    llm = FakeLLM(calls(("open_app", {"name": "Chrome"}), ("tell_time", {}), extras={"_raw": 1}),
                  text("Chrome's open and it's 3:45."))
    assistant = make(llm, reg)
    reply = assistant.process("open chrome and tell the time")
    assert reply.text == "Chrome's open and it's 3:45."
    assert [(a.tool, a.result.success) for a in reply.actions] == [("open_app", True), ("tell_time", True)]

    roles = [m["role"] for m in assistant.history]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    assert assistant.history[1]["_raw"] == 1
    assert json.loads(assistant.history[1]["tool_calls"][0]["function"]["arguments"]) == {"name": "Chrome"}
    assert json.loads(assistant.history[2]["content"]) == {"success": True, "message": "Opening Chrome."}
    # The second LLM call saw the tool results.
    assert [m["role"] for m in llm.seen[1]] == ["user", "assistant", "tool", "tool"]


def test_failed_tool_is_reported_to_llm(reg):
    llm = FakeLLM(calls(("open_app", {})), text("Which app?"))
    assistant = make(llm, reg)
    reply = assistant.process("open")
    assert not reply.actions[0].result.success
    assert json.loads(assistant.history[2]["content"])["success"] is False


def test_empty_llm_text_falls_back_to_tool_messages(reg):
    llm = FakeLLM(calls(("tell_time", {})), text(""))
    assert make(llm, reg).process("time?").text == "It's 3:45 PM."


def test_empty_llm_text_without_actions(reg):
    reply = make(FakeLLM(text("  ")), reg).process("hmm")
    assert "rephrase" in reply.text


def test_max_tool_rounds(reg):
    llm = FakeLLM(*[calls(("tell_time", {}))] * 2)
    assistant = make(llm, reg, assistant={"max_tool_rounds": 2})
    reply = assistant.process("loop forever")
    assert len(reply.actions) == 2 and reply.text == "It's 3:45 PM. It's 3:45 PM."
    assert assistant.history[-1]["role"] == "assistant"


def test_llm_unavailable_rolls_back_history(reg):
    error = LLMUnavailableError([LLMError("groq", LLMErrorKind.NETWORK, "down")])
    assistant = make(FakeLLM(text("first"), error), reg)
    assistant.process("hi")
    reply = assistant.process("open chrome")
    assert reply.error and "internet" in reply.text
    assert len(assistant.history) == 2  # failed turn removed


def test_llm_fails_after_tools_ran(reg):
    error = LLMUnavailableError([LLMError("groq", LLMErrorKind.RATE_LIMIT, "429")])
    assistant = make(FakeLLM(calls(("open_app", {"name": "Notepad"})), error), reg)
    reply = assistant.process("open notepad")
    assert reply.error and reply.text == "Opening Notepad."
    assert assistant.history[-1] == {"role": "assistant", "content": "Opening Notepad."}


def test_unexpected_exception_never_escapes(reg):
    assistant = make(FakeLLM(ZeroDivisionError("bug")), reg)
    reply = assistant.process("hi")
    assert reply.error and "something went wrong" in reply.text
    assert assistant.history == []


def test_single_llm_error_is_explained(reg):
    assistant = make(FakeLLM(LLMError("groq", LLMErrorKind.AUTH, "bad")), reg)
    reply = assistant.process("hi")
    assert "rejected" in reply.text and reply.detail == "groq: auth - bad"


def test_blank_input(reg):
    llm = FakeLLM()
    reply = make(llm, reg).process("   ")
    assert reply.error and llm.seen == []


def test_history_trimmed_to_max_turns(reg):
    llm = FakeLLM(*[text(f"r{i}") for i in range(5)])
    assistant = make(llm, reg, assistant={"max_context_turns": 3})
    for i in range(5):
        assistant.process(f"u{i}")
    users = [m["content"] for m in assistant.history if m["role"] == "user"]
    assert users == ["u2", "u3", "u4"]
    assert assistant.history[0]["role"] == "user"


def test_trim_never_orphans_tool_messages(reg):
    outcomes = []
    for _ in range(4):
        outcomes += [calls(("tell_time", {})), text("ok")]
    assistant = make(FakeLLM(*outcomes), reg, assistant={"max_context_turns": 2})
    for i in range(4):
        assistant.process(f"time {i}")
    assert assistant.history[0]["role"] == "user"
    assert [m["role"] for m in assistant.history].count("user") == 2


def test_reset(reg):
    assistant = make(FakeLLM(text("hi")), reg)
    assistant.process("hello")
    assistant.reset()
    assert assistant.history == []


def test_chat_may_use_another_model_when_main_one_is_busy(reg):
    llm = FakeLLM(text("hi"))
    make(llm, reg).process("hello")
    assert llm.options == {"rotate_on_rate_limit": True}


# ----------------------------------------------------------------- quick replies
def test_quick_reply_skips_second_llm_call(reg):
    llm = FakeLLM(calls(("open_app", {"name": "Notepad"})))
    assistant = make(llm, reg, quick=True)
    reply = assistant.process("open notepad")
    assert reply.text == "Opening Notepad." and len(llm.seen) == 1
    assert [m["role"] for m in assistant.history] == ["user", "assistant", "tool", "assistant"]


def test_quick_reply_joins_several_actions(reg):
    llm = FakeLLM(calls(("open_app", {"name": "Chrome"}), ("tell_time", {})))
    assert make(llm, reg, quick=True).process("open chrome and tell the time").text == \
        "Opening Chrome. It's 3:45 PM."


def test_failed_action_still_goes_to_the_llm(reg):
    llm = FakeLLM(calls(("open_app", {})), text("Which app should I open?"))
    assert make(llm, reg, quick=True).process("open").text == "Which app should I open?"
    assert len(llm.seen) == 2


def test_hinglish_gets_an_llm_reply(reg):
    llm = FakeLLM(calls(("open_app", {"name": "Notepad"})), text("Notepad khol diya!"))
    assert make(llm, reg, quick=True).process("notepad kholo").text == "Notepad khol diya!"


def test_data_tools_are_explained_by_the_llm(reg):
    @tool(description="stats", registry=reg, speak_result=False)
    def stats() -> ToolResult:
        return ToolResult.ok("CPU 12%; RAM 40%")

    llm = FakeLLM(calls(("stats", {})), text("Your PC is relaxed."))
    assert make(llm, reg, quick=True).process("how is my pc").text == "Your PC is relaxed."


@pytest.mark.parametrize("text, expected", [
    ("volume thoda kam karo", True), ("notepad kholo", True), ("mujhe time batao", True),
    ("सूज़ी समय बताओ", True), ("open notepad", False), ("what time is it", False),
    ("do it", False), ("set volume to 30 and open chrome", False),
])
def test_looks_like_hinglish(text, expected):
    from core.assistant import looks_like_hinglish

    assert looks_like_hinglish(text) is expected
