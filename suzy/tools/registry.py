"""Tool registry: the bridge between the LLM's function calls and Python code.

Each tool is a plain, typed Python function decorated with :func:`tool`. The
decorator records the tool's description and JSON schema so the LLM can see
it, and :meth:`ToolRegistry.execute` runs it safely:

* arguments are validated and coerced against the schema,
* any exception is caught and turned into a friendly failure message,
* every execution is written to ``logs/actions.log``.
"""

from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tools.safety import SafetyLevel
from tools.winutils import PlatformNotSupportedError

log = logging.getLogger("suzy.tools")
action_log = logging.getLogger("suzy.actions")

# Modules in the tools/ package that are infrastructure, not tools.
NON_TOOL_MODULES = {"registry", "safety", "winutils", "context"}


@dataclass
class ToolResult:
    """Outcome of a tool call.

    ``message`` is a short, human-friendly sentence the LLM can relay to the
    user; ``data`` carries any structured details.
    """

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, message: str, **data: Any) -> ToolResult:
        return cls(True, message, data)

    @classmethod
    def fail(cls, message: str, **data: Any) -> ToolResult:
        return cls(False, message, data)

    def to_llm_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"success": self.success, "message": self.message}
        if self.data:
            payload["data"] = self.data
        return payload


class ToolArgumentError(ValueError):
    """Raised when the LLM passes arguments that do not fit a tool's schema."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult]
    safety: SafetyLevel = SafetyLevel.SAFE
    category: str = "general"

    def schema(self) -> dict[str, Any]:
        """Provider-neutral schema: name, description and JSON-schema parameters."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


# --------------------------------------------------------------------------- validation
def _coerce_value(key: str, value: Any, spec: dict[str, Any]) -> Any:
    """Coerce one argument to the JSON-schema type in ``spec``.

    LLMs sometimes send ``"50"`` instead of ``50`` or ``"true"`` instead of
    ``true``; those are fixed here instead of failing the call.
    """
    expected = spec.get("type")
    try:
        if expected == "integer":
            if isinstance(value, bool):
                raise ToolArgumentError(f"'{key}' must be a whole number.")
            if isinstance(value, str):
                value = float(value.strip().rstrip("%"))
            if isinstance(value, float):
                value = int(round(value))
            if not isinstance(value, int):
                raise ToolArgumentError(f"'{key}' must be a whole number.")
        elif expected == "number":
            if isinstance(value, bool):
                raise ToolArgumentError(f"'{key}' must be a number.")
            if isinstance(value, str):
                value = float(value.strip().rstrip("%"))
            if not isinstance(value, (int, float)):
                raise ToolArgumentError(f"'{key}' must be a number.")
        elif expected == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "yes", "1", "on"):
                    value = True
                elif lowered in ("false", "no", "0", "off"):
                    value = False
            if not isinstance(value, bool):
                raise ToolArgumentError(f"'{key}' must be true or false.")
        elif expected == "string":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            if not isinstance(value, str):
                raise ToolArgumentError(f"'{key}' must be text.")
        elif expected == "array" and not isinstance(value, list):
            raise ToolArgumentError(f"'{key}' must be a list.")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ToolArgumentError):
            raise
        raise ToolArgumentError(f"'{key}' has an invalid value: {value!r}.") from exc

    allowed = spec.get("enum")
    if allowed is not None and value not in allowed:
        raise ToolArgumentError(f"'{key}' must be one of: {', '.join(map(str, allowed))}.")
    return value


def validate_arguments(tool: Tool, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Return cleaned keyword arguments for ``tool`` or raise ToolArgumentError."""
    arguments = dict(arguments or {})
    properties: dict[str, Any] = tool.parameters.get("properties", {})
    required: list[str] = tool.parameters.get("required", [])

    unknown = [key for key in arguments if key not in properties]
    for key in unknown:
        log.debug("Dropping unknown argument %r for tool %s", key, tool.name)
        arguments.pop(key)

    # Treat explicit nulls as "not provided" so function defaults apply.
    arguments = {key: value for key, value in arguments.items() if value is not None}

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolArgumentError(f"Missing required argument(s): {', '.join(missing)}.")

    return {key: _coerce_value(key, value, properties[key]) for key, value in arguments.items()}


# --------------------------------------------------------------------------- registry
class ToolRegistry:
    """Holds every registered tool and executes them safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()

    def register(self, tool: Tool) -> Tool:
        with self._lock:
            if tool.name in self._tools:
                log.debug("Re-registering tool %s", tool.name)
            self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        with self._lock:
            return self._tools.get(name)

    def all(self) -> list[Tool]:
        with self._lock:
            return list(self._tools.values())

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.all()]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run a tool by name. Never raises - failures become a ToolResult."""
        tool = self.get(name)
        if tool is None:
            result = ToolResult.fail(f"I don't have a tool called '{name}'.")
            self._log_action(name, arguments, result, 0.0)
            return result

        started = time.perf_counter()
        try:
            kwargs = validate_arguments(tool, arguments)
            result = tool.func(**kwargs)
            if isinstance(result, str):
                result = ToolResult.ok(result)
            elif not isinstance(result, ToolResult):
                result = ToolResult.ok("Done.", value=result)
        except ToolArgumentError as exc:
            result = ToolResult.fail(f"I couldn't run {name}: {exc}")
        except PlatformNotSupportedError as exc:
            result = ToolResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must never crash Suzy
            log.exception("Tool %s crashed", name)
            result = ToolResult.fail(f"Something went wrong while running {name}: {exc}")

        self._log_action(name, arguments, result, time.perf_counter() - started)
        return result

    @staticmethod
    def _log_action(
        name: str, arguments: dict[str, Any] | None, result: ToolResult, elapsed: float
    ) -> None:
        try:
            args_text = json.dumps(arguments or {}, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            args_text = repr(arguments)
        status = "OK  " if result.success else "FAIL"
        action_log.info(
            "%s | %s(%s) | %.2fs | %s", status, name, args_text, elapsed, result.message
        )


REGISTRY = ToolRegistry()


def tool(
    *,
    description: str,
    parameters: dict[str, Any] | None = None,
    name: str | None = None,
    safety: SafetyLevel = SafetyLevel.SAFE,
    category: str = "general",
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """Decorator that registers a function as an LLM-callable tool."""

    def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        schema = parameters or {"type": "object", "properties": {}}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        (registry or REGISTRY).register(
            Tool(
                name=name or func.__name__,
                description=description,
                parameters=schema,
                func=func,
                safety=safety,
                category=category,
            )
        )
        return func

    return decorator


def tool_module_names() -> list[str]:
    """Every tool module in the tools/ package. A new file = a new set of abilities."""
    import tools

    names = []
    for info in pkgutil.iter_modules(tools.__path__):
        if not info.ispkg and info.name not in NON_TOOL_MODULES and not info.name.startswith("_"):
            names.append(f"tools.{info.name}")
    return sorted(names)


def load_all_tools() -> ToolRegistry:
    """Import every tool module so its tools register themselves.

    A module that fails to import is skipped (and logged) so one broken tool
    file can never stop Suzy from starting.
    """
    for module_name in tool_module_names():
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - one broken module must not kill the rest
            log.exception("Failed to load tool module %s", module_name)
    return REGISTRY
