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
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    MODEL_UNAVAILABLE = "model_unavailable"  # model retired, renamed or not enabled for this key
    BAD_REQUEST = "bad_request"        # malformed request or tool call
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
        return self.kind not in (LLMErrorKind.NOT_CONFIGURED, LLMErrorKind.AUTH,
                                 LLMErrorKind.MODEL_UNAVAILABLE)


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
        if LLMErrorKind.MODEL_UNAVAILABLE in kinds:
            return ("The AI models in my settings aren't available any more. Re-run the installer "
                    "to update me, or set a current model in config.yaml.")
        return "My brain is having trouble right now. Please try again in a moment."

    def retry_after(self) -> float | None:
        """Seconds until a rate limit resets, if the provider said so ("try again in 1m7.5s")."""
        waits = [parse_retry_after(e.detail) for e in self.errors if e.kind is LLMErrorKind.RATE_LIMIT]
        known = [w for w in waits if w is not None]
        return min(known) if known else None

    def technical_summary(self, max_len: int = 160) -> str:
        """One short line per provider with the last error it reported."""
        latest: dict[str, LLMError] = {}
        for error in self.errors:
            latest[error.provider] = error
        parts = []
        for provider, error in latest.items():
            detail = " ".join(error.detail.split())
            if len(detail) > max_len:
                detail = detail[: max_len - 3] + "..."
            parts.append(f"{provider}: {error.kind.value} - {detail}")
        return "; ".join(parts)


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


_RETRY_AFTER = re.compile(r"try again in\s+((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", re.IGNORECASE)


def parse_retry_after(text: str) -> float | None:
    """Read "Please try again in 1m7.5s" / "in 450ms" from a rate-limit message, in seconds."""
    match = _RETRY_AFTER.search(text or "")
    if not match or not match.group(1):
        return None
    total = 0.0
    for amount, unit in re.findall(r"([\d.]+)(ms|h|m|s)", match.group(1)):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


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


def _mentions_missing_model(text: str) -> bool:
    """True if an API error message says the requested model doesn't exist/is retired."""
    lowered = text.lower()
    if "model" not in lowered:
        return False
    return any(marker in lowered for marker in (
        "decommissioned", "model_not_found", "does not exist", "not found", "no longer",
        "not supported", "not available", "deprecated", "retired",
    ))


# =========================================================================== providers
class BaseProvider(ABC):
    """A cloud LLM with an ordered list of models to try.

    Providers retire models regularly. When the current model is reported as
    unavailable, the provider moves on to the next model in its list and keeps
    using whichever one works.
    """

    name: str = "base"

    def __init__(self, api_key: str | None, model: str | list[str], temperature: float = 0.6,
                 max_tokens: int = 1024, timeout: float = 20.0) -> None:
        self.api_key = api_key
        models = [model] if isinstance(model, str) else list(model)
        self.models = [m for m in dict.fromkeys(m.strip() for m in models if m and m.strip())]
        if not self.models:
            raise ValueError(f"No model configured for {self.name}.")
        self._model_index = 0
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    @property
    def model(self) -> str:
        """The model currently in use."""
        return self.models[self._model_index]

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str, options: dict[str, Any] | None = None) -> LLMResponse:
        """Send the conversation and return text and/or tool calls. Raises LLMError.

        ``options`` may override ``max_tokens`` and ``reasoning_effort`` for one call.
        """
        if not self.is_configured:
            raise LLMError(self.name, LLMErrorKind.NOT_CONFIGURED, f"no API key set for {self.name}")
        options = options or {}
        while True:
            model = self.model
            try:
                return self._chat_with_model(model, messages, tools, system_prompt, options)
            except LLMError as exc:
                if exc.kind is LLMErrorKind.RATE_LIMIT and options.get("rotate_on_rate_limit"):
                    response = self._try_other_models(model, messages, tools, system_prompt, options)
                    if response is not None:
                        return response
                    raise
                if exc.kind is not LLMErrorKind.MODEL_UNAVAILABLE or self._model_index + 1 >= len(self.models):
                    raise
                self._model_index += 1
                log.warning("%s model %s is unavailable (%s); switching to %s",
                            self.name, model, exc.detail[:200], self.model)

    def _try_other_models(self, busy_model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                          system_prompt: str, options: dict[str, Any]) -> LLMResponse | None:
        """The free tier limits each model separately: when one is at its limit, try the others once."""
        for model in self.models:
            if model == busy_model:
                continue
            try:
                log.info("%s is at its rate limit; trying %s for this request", busy_model, model)
                return self._chat_with_model(model, messages, tools, system_prompt, options)
            except LLMError as exc:
                if exc.kind not in (LLMErrorKind.RATE_LIMIT, LLMErrorKind.MODEL_UNAVAILABLE):
                    raise
        return None

    @abstractmethod
    def _chat_with_model(self, model: str, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]], system_prompt: str,
                         options: dict[str, Any]) -> LLMResponse:
        """Make one request with a specific model. Raises LLMError."""


class GroqProvider(BaseProvider):
    """Groq's OpenAI-compatible chat API (very fast, generous free tier)."""

    name = "groq"

    def __init__(self, *args: Any, reasoning_effort: str | None = "low", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort

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
        elif isinstance(exc, groq.NotFoundError) or (
            isinstance(exc, groq.BadRequestError) and _mentions_missing_model(str(exc))
        ):
            kind = LLMErrorKind.MODEL_UNAVAILABLE
        elif isinstance(exc, (groq.BadRequestError, groq.UnprocessableEntityError)) or (
            isinstance(exc, groq.APIStatusError) and getattr(exc, "status_code", None) == 413
        ):
            kind = LLMErrorKind.BAD_REQUEST  # includes "request too large" (413)
        elif isinstance(exc, (ConnectionError, TimeoutError)):
            kind = LLMErrorKind.NETWORK
        else:
            kind = LLMErrorKind.UNKNOWN
        return LLMError(self.name, kind, str(exc)[:500])

    def _chat_with_model(self, model: str, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]], system_prompt: str,
                         options: dict[str, Any]) -> LLMResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self.convert_messages(messages, system_prompt),
            "temperature": self.temperature,
            "max_tokens": int(options.get("max_tokens", self.max_tokens)),
        }
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
            kwargs["tool_choice"] = "auto"
        effort = options.get("reasoning_effort", self.reasoning_effort)
        if effort and "gpt-oss" in model:
            # gpt-oss models "think" before answering; low effort keeps replies snappy.
            kwargs["reasoning_effort"] = effort
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

    def __init__(self, *args: Any, thinking_budget: int | None = None, **kwargs: Any) -> None:
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
            key_problem = any(marker in text for marker in (
                "API_KEY", "API key", "UNAUTHENTICATED", "ACCESS_TOKEN", "credentials"))
            if code == 401 or key_problem:
                kind = LLMErrorKind.AUTH
            elif code == 404 or (code in (400, 403) and _mentions_missing_model(text)):
                kind = LLMErrorKind.MODEL_UNAVAILABLE
            elif code == 403:
                kind = LLMErrorKind.AUTH
            elif code == 429:
                kind = LLMErrorKind.RATE_LIMIT
            elif isinstance(exc, errors.ServerError) or (code or 0) >= 500:
                kind = LLMErrorKind.SERVER
            elif code == 400:
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

    def _chat_with_model(self, model: str, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]], system_prompt: str,
                         options: dict[str, Any]) -> LLMResponse:
        client = self._get_client()
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": self.temperature,
            "max_output_tokens": int(options.get("max_tokens", self.max_tokens)),
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
                model=model,
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
    # "model" (a single name, from older config files) is tried first, then "models".
    models: list[str] = []
    legacy = config.get(f"llm.{name}.model")
    if isinstance(legacy, str) and legacy.strip():
        models.append(legacy.strip())
    for model in config.get(f"llm.{name}.models") or []:
        if isinstance(model, str) and model.strip() and model.strip() not in models:
            models.append(model.strip())
    common = {
        "api_key": Config.api_key(name),
        "model": models,
        "temperature": float(config.get("llm.temperature", 0.6)),
        "max_tokens": int(config.get("llm.max_tokens", 2048)),
        "timeout": float(config.get("llm.request_timeout", 20)),
    }
    if cls is GeminiProvider:
        return GeminiProvider(thinking_budget=config.get("llm.gemini.thinking_budget"), **common)
    if cls is GroqProvider:
        return GroqProvider(reasoning_effort=config.get("llm.groq.reasoning_effort", "low"), **common)
    return cls(**common)


# =========================================================================== router
@dataclass
class ProviderStatus:
    provider: str
    ok: bool
    model: str
    error: LLMError | None = None

    def describe(self) -> str:
        """Plain-language status line, e.g. "working (openai/gpt-oss-120b)"."""
        if self.ok:
            return f"working ({self.model})"
        assert self.error is not None
        reasons = {
            LLMErrorKind.NOT_CONFIGURED: "no API key in the .env file",
            LLMErrorKind.AUTH: "the API key was rejected",
            LLMErrorKind.RATE_LIMIT: "free usage limit reached for now",
            LLMErrorKind.NETWORK: "can't connect (check the internet)",
            LLMErrorKind.SERVER: "the service is having an outage",
            LLMErrorKind.MODEL_UNAVAILABLE: "none of the configured models are available",
            LLMErrorKind.BAD_REQUEST: "the request was refused",
            LLMErrorKind.UNKNOWN: "unexpected error",
        }
        if self.error.kind is LLMErrorKind.AUTH and "ACCESS_TOKEN_TYPE_UNSUPPORTED" in self.error.detail:
            # Google currently refuses many new-format "AQ." Gemini keys; nothing the user can fix.
            return ("NOT working - Google is refusing new 'AQ.' keys right now "
                    "(a Google-side issue, not your setup)")
        return f"NOT working - {reasons[self.error.kind]}"


class LLMRouter:
    """Primary provider -> one retry -> fallback provider."""

    def __init__(self, providers: list[BaseProvider], retry_delay: float = 0.5,
                 rate_limit_delay: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.providers = providers
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self._sleep = sleep
        self.last_provider: str | None = None
        # Providers that failed in a way a retry can't fix (bad key, no model)
        # are tried last until they work again, so they don't slow every reply.
        self._broken: set[str] = set()

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

    def _ordered_providers(self) -> list[BaseProvider]:
        healthy = [p for p in self.providers if p.name not in self._broken]
        broken = [p for p in self.providers if p.name in self._broken]
        return healthy + broken

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             system_prompt: str, options: dict[str, Any] | None = None) -> LLMResponse:
        errors: list[LLMError] = []
        for provider in self._ordered_providers():
            attempts = 2  # first try + one retry
            for attempt in range(1, attempts + 1):
                try:
                    started = time.perf_counter()
                    response = provider.chat(messages, tools, system_prompt, options)
                    log.debug("%s answered in %.2fs (%d tool calls)", provider.name,
                              time.perf_counter() - started, len(response.tool_calls))
                    self.last_provider = provider.name
                    self._broken.discard(provider.name)
                    return response
                except LLMError as exc:
                    errors.append(exc)
                    log.warning("LLM attempt %d/%d failed: %s", attempt, attempts, exc)
                    if not exc.retryable:
                        self._broken.add(provider.name)
                        break
                    if attempt == attempts:
                        break
                    delay = self.rate_limit_delay if exc.kind is LLMErrorKind.RATE_LIMIT else self.retry_delay
                    self._sleep(delay)
                except Exception as exc:  # noqa: BLE001 - unexpected SDK behaviour
                    log.exception("Unexpected error from %s", provider.name)
                    errors.append(LLMError(provider.name, LLMErrorKind.UNKNOWN, str(exc)))
                    break
        raise LLMUnavailableError(errors)

    def check(self) -> list[ProviderStatus]:
        """Send a tiny request to every provider in parallel (startup self-check).

        This also lets each provider skip past retired models before the user
        asks anything.
        """
        probe = [{"role": "user", "content": "Reply with exactly one word: ready"}]

        def run(provider: BaseProvider) -> ProviderStatus:
            if not provider.is_configured:
                return ProviderStatus(provider.name, False, provider.model,
                                      LLMError(provider.name, LLMErrorKind.NOT_CONFIGURED, "no API key"))
            try:
                provider.chat(probe, [], "You are a connectivity check.")
                self._broken.discard(provider.name)
                return ProviderStatus(provider.name, True, provider.model)
            except LLMError as exc:
                if not exc.retryable:
                    self._broken.add(provider.name)
                return ProviderStatus(provider.name, False, provider.model, exc)
            except Exception as exc:  # noqa: BLE001 - unexpected SDK behaviour
                log.exception("Health check crashed for %s", provider.name)
                return ProviderStatus(provider.name, False, provider.model,
                                      LLMError(provider.name, LLMErrorKind.UNKNOWN, str(exc)))

        if not self.providers:
            return []
        with ThreadPoolExecutor(max_workers=len(self.providers), thread_name_prefix="LLMCheck") as pool:
            statuses = list(pool.map(run, self.providers))
        for status in statuses:
            if status.ok:
                log.info("Health check: %s OK with model %s", status.provider, status.model)
            else:
                log.error("Health check: %s failed: %s", status.provider, status.error)
        return statuses
