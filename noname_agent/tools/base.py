"""Tool abstraction + registry.

A tool exposes a *name*, a *description* and a JSON-Schema *parameter spec*.
The registry renders those specs into OpenAI-style function definitions, which
is what the LLM uses to decide what to call.  Arguments are validated before
the tool runs, so a malformed model decision produces a clear error string
instead of a Python traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    """Raised by a tool when it cannot fulfil a request."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    requires_confirmation: bool = False

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Return an error string when ``arguments`` violates the schema."""
        if not isinstance(arguments, dict):
            return f"arguments for '{self.name}' must be an object"
        schema = self.parameters or {}
        props: Dict[str, Any] = schema.get("properties", {}) or {}
        required: List[str] = schema.get("required", []) or []

        for key in required:
            if key not in arguments or arguments[key] in (None, ""):
                return f"missing required argument '{key}' for tool '{self.name}'"
        if schema.get("additionalProperties", True) is False:
            unknown = set(arguments) - set(props)
            if unknown:
                return f"unexpected argument(s) {sorted(unknown)} for tool '{self.name}'"
        for key, value in arguments.items():
            if key not in props:
                continue
            expected = props[key].get("type")
            if expected and not _type_ok(expected, value):
                return (
                    f"argument '{key}' for tool '{self.name}' must be "
                    f"{expected}, got {type(value).__name__}"
                )
        return None

    def run(self, arguments: Dict[str, Any]) -> str:
        error = self.validate(arguments)
        if error:
            raise ToolError(error)
        try:
            result = self.func(**arguments)
        except TypeError as exc:  # signature mismatch -> model gave wrong args
            raise ToolError(f"bad arguments for '{self.name}': {exc}") from exc
        except ToolError:
            raise
        except Exception as exc:  # tool bugs must not kill the agent loop
            raise ToolError(f"tool '{self.name}' failed: {exc}") from exc
        return _stringify(result)


def _type_ok(expected: str, value: Any) -> bool:
    if value is None:
        return True
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    check = checks.get(expected)
    return True if check is None else check(value)


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


@dataclass
class ToolRegistry:
    """Name -> :class:`Tool` map with schema export and safe dispatch."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self.tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self.tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def names(self) -> List[str]:
        return sorted(self.tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self.tools[name].to_schema() for name in self.names()]

    def describe(self) -> str:
        lines = []
        for name in self.names():
            tool = self.tools[name]
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        """Run a tool, returning a human/LLM readable string.

        Errors are returned as ``ERROR: ...`` strings rather than raised, so
        the loop can feed them back to the model and let it recover.
        """
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            return f"ERROR: unknown tool '{name}'. Available tools: {available}."
        try:
            return tool.run(arguments or {})
        except ToolError as exc:
            return f"ERROR: {exc}"
