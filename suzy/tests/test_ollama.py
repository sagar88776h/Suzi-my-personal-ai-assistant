"""Tests for the Ollama provider (Ollama Cloud or local) against a fake Ollama server."""

from __future__ import annotations

import json

import httpx
import pytest

from core.config import Config
from core.llm import (
    LLMError,
    LLMErrorKind,
    LLMRouter,
    OllamaProvider,
    ToolCall,
    build_provider,
)

TOOLS = [{"name": "open_app", "description": "Open an app",
          "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}]
HISTORY = [
    {"role": "user", "content": "open chrome"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "open_app", "arguments": '{"name": "chrome"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "open_app", "content": '{"success": true}'},
    {"role": "assistant", "content": "Opening Chrome."},
    {"role": "user", "content": "thanks"},
]


def provider_with(handler, **kwargs) -> tuple[OllamaProvider, list]:
    requests = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    provider = OllamaProvider(kwargs.pop("api_key", "ollama_key"), kwargs.pop("model", ["gpt-oss:120b"]), **kwargs)
    provider._client = httpx.Client(transport=httpx.MockTransport(record))
    return provider, requests


def reply(message: dict, status: int = 200):
    return lambda request: httpx.Response(status, json={"model": "x", "message": message, "done": True})


def test_text_reply_and_request_shape():
    provider, requests = provider_with(reply({"role": "assistant", "content": " Hi Sagar! "}))
    response = provider.chat(HISTORY, TOOLS, "SYSTEM")
    assert response.text == "Hi Sagar!" and response.tool_calls == [] and response.provider == "ollama"

    request = requests[0]
    assert str(request.url) == "https://ollama.com/api/chat"
    assert request.headers["Authorization"] == "Bearer ollama_key"
    body = json.loads(request.content)
    assert body["model"] == "gpt-oss:120b" and body["stream"] is False and body["think"] == "low"
    assert body["tools"] == [{"type": "function", "function": TOOLS[0]}]
    assert body["options"]["num_predict"] == 1024
    messages = body["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[2]["tool_calls"] == [{"function": {"name": "open_app", "arguments": {"name": "chrome"}}}]
    assert messages[3] == {"role": "tool", "content": '{"success": true}', "tool_name": "open_app"}


def test_tool_calls_parsed():
    provider, _ = provider_with(reply({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "open_app", "arguments": {"name": "notepad"}}},
        {"function": {"name": "tell_time", "arguments": "{}"}},
    ]}))
    response = provider.chat(HISTORY[:1], TOOLS, "S")
    assert [(c.name, c.arguments) for c in response.tool_calls] == [("open_app", {"name": "notepad"}),
                                                                      ("tell_time", {})]
    assert all(isinstance(c, ToolCall) and c.id.startswith("call_") for c in response.tool_calls)


def test_options_override_tokens_and_thinking():
    provider, requests = provider_with(reply({"content": "ok"}))
    provider.chat(HISTORY[:1], [], "S", options={"max_tokens": 3000, "reasoning_effort": "medium"})
    body = json.loads(requests[0].content)
    assert body["options"]["num_predict"] == 3000 and body["think"] == "medium" and "tools" not in body


def test_no_think_for_other_models():
    provider, requests = provider_with(reply({"content": "ok"}), model=["qwen3:8b"])
    provider.chat(HISTORY[:1], [], "S")
    assert "think" not in json.loads(requests[0].content)


@pytest.mark.parametrize("status, body, kind", [
    (401, "unauthorized", LLMErrorKind.AUTH),
    (429, "too many requests", LLMErrorKind.RATE_LIMIT),
    (404, 'model "x" not found', LLMErrorKind.MODEL_UNAVAILABLE),
    (500, "internal error", LLMErrorKind.SERVER),
    (400, "invalid request", LLMErrorKind.BAD_REQUEST),
])
def test_errors_are_classified(status, body, kind):
    provider, _ = provider_with(lambda request: httpx.Response(status, text=body))
    with pytest.raises(LLMError) as info:
        provider.chat(HISTORY[:1], TOOLS, "S")
    assert info.value.kind is kind and info.value.provider == "ollama"


def test_retired_model_moves_to_next():
    def handler(request):
        if json.loads(request.content)["model"] == "old:1b":
            return httpx.Response(404, text='model "old:1b" not found')
        return httpx.Response(200, json={"message": {"content": "from the new model"}})

    provider, _ = provider_with(handler, model=["old:1b", "gpt-oss:20b"])
    assert provider.chat(HISTORY[:1], [], "S").text == "from the new model"
    assert provider.model == "gpt-oss:20b"


def test_offline_and_local_hint():
    def offline(request):
        raise httpx.ConnectError("connection refused")

    provider, _ = provider_with(offline, api_key=None, host="http://localhost:11434")
    with pytest.raises(LLMError) as info:
        provider.chat(HISTORY[:1], [], "S")
    assert info.value.kind is LLMErrorKind.NETWORK and "is Ollama running" in info.value.detail


def test_local_ollama_needs_no_key():
    provider, requests = provider_with(reply({"content": "local hello"}), api_key=None,
                                       host="http://localhost:11434/")
    assert provider.is_configured and provider.is_local
    assert provider.chat(HISTORY[:1], [], "S").text == "local hello"
    assert "Authorization" not in requests[0].headers
    assert str(requests[0].url) == "http://localhost:11434/api/chat"


def test_cloud_without_key_is_not_configured():
    provider = OllamaProvider(None, "gpt-oss:120b")
    assert not provider.is_configured
    with pytest.raises(LLMError) as info:
        provider.chat(HISTORY[:1], [], "S")
    assert info.value.kind is LLMErrorKind.NOT_CONFIGURED


def test_bad_json_reply():
    provider, _ = provider_with(lambda request: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(LLMError) as info:
        provider.chat(HISTORY[:1], [], "S")
    assert info.value.kind is LLMErrorKind.UNKNOWN


# ----------------------------------------------------------------- configuration
def test_build_provider_from_config(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ok_key")
    provider = build_provider("ollama", Config())
    assert isinstance(provider, OllamaProvider)
    assert provider.api_key == "ok_key" and provider.host == "https://ollama.com"
    assert provider.models == ["gpt-oss:120b", "gpt-oss:20b"]
    local = build_provider("ollama", Config({"llm": {"ollama": {"host": "http://localhost:11434",
                                                                "models": ["llama3.2"]}}}))
    assert local.is_local and local.models == ["llama3.2"]


def test_router_adds_ollama_when_a_key_exists(monkeypatch):
    assert [p.name for p in LLMRouter.from_config(Config()).providers] == ["groq", "gemini"]
    monkeypatch.setenv("OLLAMA_API_KEY", "ok_key")
    assert [p.name for p in LLMRouter.from_config(Config()).providers] == ["groq", "gemini", "ollama"]


def test_router_order_with_ollama_first_and_fallback_list(monkeypatch):
    config = Config({"llm": {"provider": "ollama", "fallback_provider": ["groq", "gemini"]}})
    assert [p.name for p in LLMRouter.from_config(config).providers] == ["ollama", "groq", "gemini"]
