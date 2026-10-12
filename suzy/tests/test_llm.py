"""Tests for the LLM providers (SDK clients mocked) and the failover router."""

from __future__ import annotations

import json
from types import SimpleNamespace

import groq
import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from core.config import Config
from core.llm import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    LLMError,
    LLMErrorKind,
    LLMResponse,
    LLMRouter,
    LLMUnavailableError,
    ToolCall,
    build_provider,
)

TOOLS = [{
    "name": "open_app",
    "description": "Open an app",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
}]

CONVERSATION = [
    {"role": "user", "content": "open chrome and tell me the time"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "open_app", "arguments": '{"name": "chrome"}'}},
        {"id": "call_2", "type": "function", "function": {"name": "tell_time", "arguments": "{}"}},
    ], "_gemini_content": None},
    {"role": "tool", "tool_call_id": "call_1", "name": "open_app", "content": '{"success": true, "message": "Opening Google Chrome."}'},
    {"role": "tool", "tool_call_id": "call_2", "name": "tell_time", "content": '{"success": true, "message": "It\'s 3:45 PM."}'},
    {"role": "assistant", "content": "Chrome's open, and it's 3:45 PM."},
    {"role": "user", "content": "thanks"},
]


# ============================================================================ Groq
class FakeGroqClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply


def groq_reply(content=None, tool_calls=None):
    calls = [SimpleNamespace(id=c[0], function=SimpleNamespace(name=c[1], arguments=c[2])) for c in tool_calls or []]
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_groq(client) -> GroqProvider:
    provider = GroqProvider(api_key="gsk_test", model="llama-3.3-70b-versatile")
    provider._client = client
    return provider


def test_groq_text_reply_and_request_shape():
    client = FakeGroqClient(reply=groq_reply("Hello Sourab!"))
    response = make_groq(client).chat(CONVERSATION, TOOLS, "SYSTEM")
    assert response == LLMResponse("Hello Sourab!", [], "groq")

    sent = client.calls[0]
    assert sent["model"] == "llama-3.3-70b-versatile" and sent["tool_choice"] == "auto"
    assert sent["tools"] == [{"type": "function", "function": TOOLS[0]}]
    assert sent["messages"][0] == {"role": "system", "content": "SYSTEM"}
    # Private keys are stripped, tool messages pass through.
    assert all(not k.startswith("_") for m in sent["messages"] for k in m)
    assert sent["messages"][3]["role"] == "tool" and sent["messages"][3]["tool_call_id"] == "call_1"


def test_groq_tool_calls_parsed():
    client = FakeGroqClient(reply=groq_reply(None, [("c1", "open_app", '{"name": "notepad"}'),
                                                    ("c2", "tell_time", ""),
                                                    ("c3", "open_app", "{broken json")]))
    response = make_groq(client).chat(CONVERSATION[:1], TOOLS, "S")
    assert response.text == ""
    assert response.tool_calls == [ToolCall("c1", "open_app", {"name": "notepad"}),
                                   ToolCall("c2", "tell_time", {}),
                                   ToolCall("c3", "open_app", {})]


def test_groq_without_tools_omits_tool_choice():
    client = FakeGroqClient(reply=groq_reply("hi"))
    make_groq(client).chat(CONVERSATION[:1], [], "S")
    assert "tools" not in client.calls[0] and "tool_choice" not in client.calls[0]


def _status_error(cls, code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    "error, kind",
    [
        (_status_error(groq.AuthenticationError, 401), LLMErrorKind.AUTH),
        (_status_error(groq.RateLimitError, 429), LLMErrorKind.RATE_LIMIT),
        (_status_error(groq.InternalServerError, 503), LLMErrorKind.SERVER),
        (_status_error(groq.BadRequestError, 400), LLMErrorKind.BAD_REQUEST),
        (groq.APIConnectionError(request=httpx.Request("POST", "https://x")), LLMErrorKind.NETWORK),
        (groq.APITimeoutError(request=httpx.Request("POST", "https://x")), LLMErrorKind.NETWORK),
        (ValueError("weird"), LLMErrorKind.UNKNOWN),
    ],
)
def test_groq_error_classification(error, kind):
    with pytest.raises(LLMError) as info:
        make_groq(FakeGroqClient(error=error)).chat(CONVERSATION[:1], TOOLS, "S")
    assert info.value.kind is kind and info.value.provider == "groq"


def test_groq_not_configured():
    with pytest.raises(LLMError) as info:
        GroqProvider(api_key=None, model="m").chat([], [], "S")
    assert info.value.kind is LLMErrorKind.NOT_CONFIGURED and not info.value.retryable


# ============================================================================ Gemini
class FakeGeminiClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def gemini_response(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=list(parts)))])


def make_gemini(client, thinking_budget=0) -> GeminiProvider:
    provider = GeminiProvider(api_key="AIza_test", model="gemini-2.5-flash", thinking_budget=thinking_budget)
    provider._client = client
    return provider


def test_gemini_message_conversion():
    contents = GeminiProvider.convert_messages(CONVERSATION)
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    calls = [p.function_call for p in contents[1].parts]
    assert [(c.name, c.args) for c in calls] == [("open_app", {"name": "chrome"}), ("tell_time", {})]
    responses = [p.function_response for p in contents[2].parts]  # both results grouped in one turn
    assert [r.name for r in responses] == ["open_app", "tell_time"]
    assert responses[0].response["message"] == "Opening Google Chrome."
    assert contents[3].parts[0].text == "Chrome's open, and it's 3:45 PM."


def test_gemini_reuses_raw_content():
    raw = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="x", args={}))])
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [], "_gemini_content": raw}]
    assert GeminiProvider.convert_messages(messages)[1] is raw


def test_gemini_text_reply_and_config():
    client = FakeGeminiClient(gemini_response(types.Part(text="thinking...", thought=True),
                                              types.Part(text="Namaste Sourab!")))
    response = make_gemini(client).chat(CONVERSATION[:1], TOOLS, "SYSTEM")
    assert response.text == "Namaste Sourab!" and response.tool_calls == [] and response.message_extras == {}
    config = client.calls[0]["config"]
    assert config.system_instruction == "SYSTEM"
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_budget == 0
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "open_app" and declaration.parameters_json_schema == TOOLS[0]["parameters"]


def test_gemini_tool_call_reply():
    client = FakeGeminiClient(gemini_response(
        types.Part(function_call=types.FunctionCall(name="open_app", args={"name": "chrome"}))))
    response = make_gemini(client, thinking_budget=None).chat(CONVERSATION[:1], TOOLS, "S")
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "open_app" and call.arguments == {"name": "chrome"} and call.id.startswith("call_")
    assert response.message_extras["_gemini_content"].parts[0].function_call.name == "open_app"
    assert client.calls[0]["config"].thinking_config is None


def test_gemini_empty_candidates():
    client = FakeGeminiClient(SimpleNamespace(candidates=[]))
    assert make_gemini(client).chat(CONVERSATION[:1], TOOLS, "S").text == ""


@pytest.mark.parametrize(
    "error, kind",
    [
        (genai_errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}}),
         LLMErrorKind.AUTH),
        (genai_errors.ClientError(403, {"error": {"message": "forbidden"}}), LLMErrorKind.AUTH),
        (genai_errors.ClientError(429, {"error": {"message": "quota"}}), LLMErrorKind.RATE_LIMIT),
        (genai_errors.ServerError(503, {"error": {"message": "overloaded"}}), LLMErrorKind.SERVER),
        (genai_errors.ClientError(404, {"error": {"message": "model not found"}}), LLMErrorKind.BAD_REQUEST),
        (httpx.ConnectError("no route"), LLMErrorKind.NETWORK),
        (httpx.ReadTimeout("slow"), LLMErrorKind.NETWORK),
        (RuntimeError("weird"), LLMErrorKind.UNKNOWN),
    ],
)
def test_gemini_error_classification(error, kind):
    with pytest.raises(LLMError) as info:
        make_gemini(FakeGeminiClient(error=error)).chat(CONVERSATION[:1], TOOLS, "S")
    assert info.value.kind is kind


# ============================================================================ router
class ScriptedProvider(BaseProvider):
    """Returns/raises the scripted outcomes in order."""

    def __init__(self, name, outcomes, configured=True):
        super().__init__(api_key="k" if configured else None, model="m")
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat(self, messages, tools, system_prompt):
        self.calls += 1
        if not self.is_configured:
            raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED, "no key")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(outcome, [], self.name)


def router(*providers):
    sleeps: list[float] = []
    return LLMRouter(list(providers), sleep=sleeps.append), sleeps


def test_router_primary_success():
    primary, fallback = ScriptedProvider("groq", ["hi"]), ScriptedProvider("gemini", ["unused"])
    r, sleeps = router(primary, fallback)
    assert r.chat([], [], "S").text == "hi"
    assert (primary.calls, fallback.calls, sleeps, r.last_provider) == (1, 0, [], "groq")


def test_router_retries_once_then_succeeds():
    primary = ScriptedProvider("groq", [LLMError("groq", LLMErrorKind.SERVER, "503"), "second try"])
    r, sleeps = router(primary, ScriptedProvider("gemini", []))
    assert r.chat([], [], "S").text == "second try"
    assert primary.calls == 2 and sleeps == [0.5]


def test_router_falls_back_after_retry():
    net = LLMError("groq", LLMErrorKind.NETWORK, "down")
    primary, fallback = ScriptedProvider("groq", [net, net]), ScriptedProvider("gemini", ["from gemini"])
    r, _ = router(primary, fallback)
    response = r.chat([], [], "S")
    assert response.provider == "gemini" and primary.calls == 2 and fallback.calls == 1


def test_router_auth_error_skips_retry():
    primary = ScriptedProvider("groq", [LLMError("groq", LLMErrorKind.AUTH, "bad key")])
    fallback = ScriptedProvider("gemini", ["ok"])
    r, sleeps = router(primary, fallback)
    assert r.chat([], [], "S").text == "ok" and primary.calls == 1 and sleeps == []


def test_router_rate_limit_uses_longer_delay():
    rl = LLMError("groq", LLMErrorKind.RATE_LIMIT, "429")
    r, sleeps = router(ScriptedProvider("groq", [rl, "ok"]))
    r.chat([], [], "S")
    assert sleeps == [2.0]


def test_router_unexpected_exception_moves_on():
    r, _ = router(ScriptedProvider("groq", [KeyError("sdk bug")]), ScriptedProvider("gemini", ["ok"]))
    assert r.chat([], [], "S").provider == "gemini"


def test_router_all_fail():
    net = LLMError("groq", LLMErrorKind.NETWORK, "down")
    r, _ = router(ScriptedProvider("groq", [net, net]), ScriptedProvider("gemini", [], configured=False))
    with pytest.raises(LLMUnavailableError) as info:
        r.chat([], [], "S")
    assert "internet" in info.value.user_message()
    assert info.value.kinds() == {LLMErrorKind.NETWORK, LLMErrorKind.NOT_CONFIGURED}


@pytest.mark.parametrize(
    "kinds, phrase",
    [
        ([], "API key"),
        ([LLMErrorKind.NOT_CONFIGURED], "API key yet"),
        ([LLMErrorKind.AUTH, LLMErrorKind.NOT_CONFIGURED], "rejected"),
        ([LLMErrorKind.RATE_LIMIT, LLMErrorKind.AUTH], "usage limit"),
        ([LLMErrorKind.NETWORK, LLMErrorKind.RATE_LIMIT], "internet"),
        ([LLMErrorKind.SERVER], "trouble"),
    ],
)
def test_unavailable_messages(kinds, phrase):
    error = LLMUnavailableError([LLMError("x", k, "") for k in kinds])
    assert phrase in error.user_message()


def test_from_config_order_and_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = Config({"llm": {"provider": "gemini", "fallback_provider": "groq",
                             "gemini": {"model": "gemini-x", "thinking_budget": None}}})
    r = LLMRouter.from_config(config)
    assert [p.name for p in r.providers] == ["gemini", "groq"]
    assert r.configured_providers == ["groq"]
    assert r.providers[0].model == "gemini-x" and r.providers[0].thinking_budget is None


def test_from_config_dedupes_and_ignores_unknown():
    config = Config({"llm": {"provider": "groq", "fallback_provider": "groq"}})
    assert [p.name for p in LLMRouter.from_config(config).providers] == ["groq"]
    config = Config({"llm": {"provider": "openai", "fallback_provider": "gemini"}})
    assert [p.name for p in LLMRouter.from_config(config).providers] == ["gemini"]
    with pytest.raises(ValueError):
        build_provider("openai", config)


def test_tool_call_arguments_roundtrip_json():
    # The assistant stores arguments as JSON strings; both providers must parse them back.
    contents = GeminiProvider.convert_messages([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "set_volume", "arguments": json.dumps({"level": 30})}}]},
    ])
    assert contents[0].parts[0].function_call.args == {"level": 30}
