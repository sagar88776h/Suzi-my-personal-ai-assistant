"""Suzy's brain: LLM providers with native tool calling and automatic failover.

Conversation messages use one provider-neutral format (OpenAI-style dicts),
which each provider converts to its own API:

    {"role": "user", "content": "open chrome"}
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "open_app", "arguments": "{\\"name\\": \\"chrome\\"}"}}]}
    {"role": "tool", "tool_call_id": "call_1", "name": "open_app", "content": "{...}"}
    {"role": "assistant", "content": "Opening Google Chrome."}

Keys starting with ``_`` hold provider-private state (e.g. Gemini's raw reply)
and are stripped before a message is sent to a different provider.

:class:`LLMRouter` tries the primary provider, retries it once on failure, then
switches to the fallback provider.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.config import Config

log = logging.getLogger("suzy.llm")


# =========================================================================== data types
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    # Extra keys to store on the assistant message (provider-private state).
    message_extras: dict[str, Any] = field(default_factory=dict)


class LLMErrorKind(str, Enum):
    NOT_CONFIGURED = "not_configured"  # no API key / SDK missing
    AUTH = "auth"                      # invalid or revoked API key
    RATE_LIMIT = "rate_limit"          # free-tier quota hit
    NETWORK = "network"                # offline, DNS failure, timeout
    SERVER = "server"                  # provider outage (5xx)
    BAD_REQUEST = "bad_request"        # malformed tool call, bad model name, ...
    UNKNOWN = "unknown"


class LLMError(Exception):
    """A single provider failed."""

    def __init__(self, provider: str, kind: LLMErrorKind, message: str) -> None:
        super().__init__(f"[{provider}] {kind.value}: {message}")
        self.provider = provider
        self.kind = kind
        self.detail = message

    @property
    def retryable(self) -> bool:
        return self.kind not in (LLMErrorKind.NOT_CONFIGURED, LLMErrorKind.AUTH)


class LLMUnavailableError(Exception):
    """Every provider failed. ``user_message()`` explains why in plain words."""

    def __init__(self, errors: list[LLMError]) -> None:
        super().__init__("; ".join(str(e) for e in errors) or "no LLM providers configured")
        self.errors = errors

    def kinds(self) -> set[LLMErrorKind]:
        return {e.kind for e in self.errors}

    def user_message(self) -> str:
        kinds = self.kinds()
        if not kinds or kinds == {LLMErrorKind.NOT_CONFIGURED}:
            return ("I don't have a working API key yet. Add your Groq or Gemini key to the "
                    ".env file and restart me.")
        if LLMErrorKind.NETWORK in kinds:
            return "I can't reach the internet right now, so my brain is offline. Check your connection?"
        if LLMErrorKind.RATE_LIMIT in kinds:
            return "I've hit my free usage limit for the moment. Give me a minute and try again."
        if LLMErrorKind.AUTH in kinds:
            return "My API key was rejected. Please check the keys in the .env file."
        return "My brain is having trouble right now. Please try again in a moment."


# =========================================================================== helpers
def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Parse a tool call's arguments, tolerating empty or malformed JSON."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        log.warning("Unparseable tool arguments: %r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _public(message: dict[str, Any]) -> dict[str, Any]:
    """Strip provider-private keys (those starting with ``_``)."""
    return {k: v for k, v in message.items() if not k.startswith("_")}


def _parse_tool_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(content or "{}")
    except (TypeError, ValueError):
        return {"result": str(content)}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


# =========================================================================== providers
class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, api_key: str | None, model: str, temperature: float = 0.6,
                 max_tokens: int = 1024, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str) -> LLMResponse:
        """Send the conversation and return text and/or tool calls. Raises LLMError."""


class GroqProvider(BaseProvider):
    """Groq's OpenAI-compatible chat API (Llama models, generous free tier)."""

    name = "groq"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED, "groq package not installed") from exc
            # Retries are handled by LLMRouter so failover stays fast.
            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client

    @staticmethod
    def convert_messages(messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            clean = _public(message)
            if clean.get("role") == "assistant" and not clean.get("tool_calls"):
                clean.pop("tool_calls", None)
                clean["content"] = clean.get("content") or ""
            converted.append(clean)
        return converted

    @staticmethod
    def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"type": "function", "function": t} for t in tools]

    def _classify(self, exc: Exception) -> LLMError:
        import groq

        if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
            kind = LLMErrorKind.AUTH
        elif isinstance(exc, groq.RateLimitError):
            kind = LLMErrorKind.RATE_LIMIT
        elif isinstance(exc, groq.APIConnectionError):  # includes timeouts
            kind = LLMErrorKind.NETWORK
        elif isinstance(exc, groq.InternalServerError):
            kind = LLMErrorKind.SERVER
        elif isinstance(exc, (groq.BadRequestError, groq.NotFoundError, groq.UnprocessableEntityError)):
            kind = LLMErrorKind.BAD_REQUEST
        elif isinstance(exc, (ConnectionError, TimeoutError)):
            kind = LLMErrorKind.NETWORK
        else:
            kind = LLMErrorKind.UNKNOWN
        return LLMError(self.name, kind, str(exc)[:500])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str) -> LLMResponse:
        if not self.is_configured:
            raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED, "GROQ_API_KEY is not set")
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.convert_messages(messages, system_prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
            kwargs["tool_choice"] = "auto"
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below
            raise self._classify(exc) from exc

        message = completion.choices[0].message
        calls = [
            ToolCall(
                id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                name=tc.function.name,
                arguments=_parse_arguments(tc.function.arguments),
            )
            for tc in (message.tool_calls or [])
        ]
        return LLMResponse(text=message.content or "", tool_calls=calls, provider=self.name)


class GeminiProvider(BaseProvider):
    """Google Gemini via the google-genai SDK."""

    name = "gemini"

    def __init__(self, *args: Any, thinking_budget: int | None = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thinking_budget = thinking_budget

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED,
                               "google-genai package not installed") from exc
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        return self._client

    @staticmethod
    def convert_messages(messages: list[dict[str, Any]]) -> list[Any]:
        """Convert neutral messages to a list of ``types.Content``."""
        from google.genai import types

        contents: list[Any] = []
        pending_responses: list[Any] = []

        def flush_responses() -> None:
            if pending_responses:
                contents.append(types.Content(role="user", parts=list(pending_responses)))
                pending_responses.clear()

        for message in messages:
            role = message.get("role")
            if role == "tool":
                pending_responses.append(types.Part.from_function_response(
                    name=message.get("name") or "tool",
                    response=_parse_tool_content(message.get("content")),
                ))
                continue
            flush_responses()
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=message.get("content") or "")]))
            elif role == "assistant":
                raw = message.get("_gemini_content")
                if raw is not None:  # reuse Gemini's own reply (keeps thought signatures)
                    contents.append(raw)
                    continue
                parts = []
                if message.get("content"):
                    parts.append(types.Part(text=message["content"]))
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=function.get("name"), args=_parse_arguments(function.get("arguments")),
                    )))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
        flush_responses()
        return contents

    def _classify(self, exc: Exception) -> LLMError:
        from google.genai import errors

        text = str(exc)
        kind = LLMErrorKind.UNKNOWN
        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
                kind = LLMErrorKind.AUTH
            elif code == 429:
                kind = LLMErrorKind.RATE_LIMIT
            elif isinstance(exc, errors.ServerError) or (code or 0) >= 500:
                kind = LLMErrorKind.SERVER
            elif code in (400, 404):
                kind = LLMErrorKind.BAD_REQUEST
        else:
            try:
                import httpx

                network_errors: tuple[type[BaseException], ...] = (
                    httpx.TransportError, ConnectionError, TimeoutError)
            except ImportError:
                network_errors = (ConnectionError, TimeoutError)
            if isinstance(exc, network_errors):
                kind = LLMErrorKind.NETWORK
        return LLMError(self.name, kind, text[:500])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str) -> LLMResponse:
        if not self.is_configured:
            raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED, "GEMINI_API_KEY is not set")
        client = self._get_client()
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tools:
            config_kwargs["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"], description=t["description"], parameters_json_schema=t["parameters"],
                ) for t in tools
            ])]
        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=self.convert_messages(messages),
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            raise self._classify(exc) from exc

        candidates = getattr(response, "candidates", None) or []
        content = candidates[0].content if candidates else None
        if content is None or not content.parts:
            # Usually a safety block or an empty reply - treat as a polite non-answer.
            return LLMResponse(text="", tool_calls=[], provider=self.name)

        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in content.parts:
            if part.function_call is not None:
                calls.append(ToolCall(
                    id=part.function_call.id or f"call_{uuid.uuid4().hex[:12]}",
                    name=part.function_call.name or "",
                    arguments=dict(part.function_call.args or {}),
                ))
            elif part.text and not part.thought:
                texts.append(part.text)
        return LLMResponse(
            text="".join(texts).strip(),
            tool_calls=calls,
            provider=self.name,
            message_extras={"_gemini_content": content} if calls else {},
        )


PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def build_provider(name: str, config: Config) -> BaseProvider:
    """Create a provider from config.yaml settings and the .env API key."""
    cls = PROVIDER_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose one of: {', '.join(PROVIDER_CLASSES)}")
    common = {
        "api_key": Config.api_key(name),
        "model": config.get(f"llm.{name}.model"),
        "temperature": float(config.get("llm.temperature", 0.6)),
        "max_tokens": int(config.get("llm.max_tokens", 1024)),
        "timeout": float(config.get("llm.request_timeout", 20)),
    }
    if cls is GeminiProvider:
        return GeminiProvider(thinking_budget=config.get("llm.gemini.thinking_budget", 0), **common)
    return cls(**common)


# =========================================================================== router
class LLMRouter:
    """Primary provider -> one retry -> fallback provider."""

    def __init__(self, providers: list[BaseProvider], retry_delay: float = 0.5,
                 rate_limit_delay: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.providers = providers
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self._sleep = sleep
        self.last_provider: str | None = None

    @classmethod
    def from_config(cls, config: Config) -> LLMRouter:
        order: list[str] = []
        for key in ("llm.provider", "llm.fallback_provider"):
            name = str(config.get(key) or "").strip().lower()
            if name and name not in order:
                order.append(name)
        providers = []
        for name in order:
            try:
                providers.append(build_provider(name, config))
            except ValueError as exc:
                log.error("%s", exc)
        return cls(providers)

    @property
    def configured_providers(self) -> list[str]:
        return [p.name for p in self.providers if p.is_configured]

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str) -> LLMResponse:
        errors: list[LLMError] = []
        for provider in self.providers:
            attempts = 2  # first try + one retry
            for attempt in range(1, attempts + 1):
                try:
                    started = time.perf_counter()
                    response = provider.chat(messages, tools, system_prompt)
                    log.debug("%s answered in %.2fs (%d tool calls)", provider.name,
                              time.perf_counter() - started, len(response.tool_calls))
                    self.last_provider = provider.name
                    return response
                except LLMError as exc:
                    errors.append(exc)
                    log.warning("LLM attempt %d/%d failed: %s", attempt, attempts, exc)
                    if not exc.retryable or attempt == attempts:
                        break
                    delay = self.rate_limit_delay if exc.kind is LLMErrorKind.RATE_LIMIT else self.retry_delay
                    self._sleep(delay)
                except Exception as exc:  # noqa: BLE001 - unexpected SDK behaviour
                    log.exception("Unexpected error from %s", provider.name)
                    errors.append(LLMError(provider.name, LLMErrorKind.UNKNOWN, str(exc)))
                    break
        raise LLMUnavailableError(errors)
