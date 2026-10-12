"""The assistant loop: user text -> LLM -> tools -> LLM -> reply.

:class:`Assistant` is deliberately UI-agnostic. The terminal chat, the voice
pipeline and the PyQt UI all call :meth:`Assistant.process` with plain text
and get back an :class:`AssistantReply`.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.config import Config
from core.llm import LLMError, LLMRouter, LLMUnavailableError
from core.prompts import build_system_prompt
from tools.registry import ToolRegistry, ToolResult

log = logging.getLogger("suzy.assistant")


@dataclass
class ActionRecord:
    tool: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass
class AssistantReply:
    text: str
    actions: list[ActionRecord] = field(default_factory=list)
    provider: str | None = None
    error: bool = False


class Assistant:
    def __init__(
        self,
        config: Config,
        llm: LLMRouter,
        registry: ToolRegistry,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.clock = clock
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ settings
    @property
    def max_turns(self) -> int:
        return max(1, int(self.config.get("assistant.max_context_turns", 20)))

    @property
    def max_tool_rounds(self) -> int:
        return max(1, int(self.config.get("assistant.max_tool_rounds", 5)))

    def system_prompt(self) -> str:
        return build_system_prompt(
            assistant_name=self.config.get("assistant.name", "Suzy"),
            user_name=self.config.get("user.name", "Sagar"),
            now=self.clock(),
        )

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        """Forget the conversation so far."""
        with self._lock:
            self.history.clear()

    def process(self, user_text: str) -> AssistantReply:
        """Handle one user utterance. Never raises."""
        text = (user_text or "").strip()
        if not text:
            return AssistantReply("Sorry, I didn't catch that.", error=True)

        with self._lock:
            checkpoint = len(self.history)
            self.history.append({"role": "user", "content": text})
            actions: list[ActionRecord] = []
            try:
                reply = self._run(actions)
            except (LLMUnavailableError, LLMError) as exc:
                log.warning("LLM unavailable: %s", exc)
                reply = self._recover(checkpoint, actions, exc)
            except Exception:  # noqa: BLE001 - the assistant must never crash
                log.exception("Unexpected error while processing %r", text)
                reply = self._recover(checkpoint, actions, None)
            self._trim_history()
            return reply

    # ------------------------------------------------------------------ internals
    def _run(self, actions: list[ActionRecord]) -> AssistantReply:
        tools = self.registry.schemas()
        provider: str | None = None

        for _round in range(self.max_tool_rounds):
            response = self.llm.chat(self.history, tools, self.system_prompt())
            provider = response.provider

            if not response.tool_calls:
                reply_text = response.text.strip() or self._summarize(actions)
                self.history.append({"role": "assistant", "content": reply_text})
                return AssistantReply(reply_text, actions, provider)

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in response.tool_calls
                ],
            }
            assistant_message.update(response.message_extras)
            self.history.append(assistant_message)

            for call in response.tool_calls:
                log.info("Tool call: %s(%s)", call.name, call.arguments)
                result = self.registry.execute(call.name, call.arguments)
                actions.append(ActionRecord(call.name, call.arguments, result))
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result.to_llm_payload(), ensure_ascii=False, default=str),
                })

        # Too many chained tool calls: stop and report what was done.
        log.warning("Reached max tool rounds (%d)", self.max_tool_rounds)
        reply_text = self._summarize(actions)
        self.history.append({"role": "assistant", "content": reply_text})
        return AssistantReply(reply_text, actions, provider)

    def _recover(self, checkpoint: int, actions: list[ActionRecord],
                 exc: Exception | None) -> AssistantReply:
        """Build a reply after a failure and leave the history consistent."""
        if actions:
            # Tools already ran - report their results rather than pretend nothing happened.
            reply_text = self._summarize(actions)
            self.history.append({"role": "assistant", "content": reply_text})
            return AssistantReply(reply_text, actions, None, error=True)

        del self.history[checkpoint:]  # drop the unanswered user message
        if isinstance(exc, LLMUnavailableError):
            message = exc.user_message()
        elif isinstance(exc, LLMError):
            message = LLMUnavailableError([exc]).user_message()
        else:
            message = "Oops, something went wrong on my side. Could you try that again?"
        return AssistantReply(message, [], None, error=True)

    @staticmethod
    def _summarize(actions: list[ActionRecord]) -> str:
        if not actions:
            return "Sorry, I'm not sure how to help with that. Could you rephrase it?"
        messages = [a.result.message for a in actions[-3:] if a.result.message]
        return " ".join(messages) or "Done."

    def _trim_history(self) -> None:
        """Keep only the last ``max_turns`` turns (a turn starts at a user message)."""
        user_indexes = [i for i, m in enumerate(self.history) if m.get("role") == "user"]
        excess = len(user_indexes) - self.max_turns
        if excess > 0:
            cut = user_indexes[excess]
            del self.history[:cut]
