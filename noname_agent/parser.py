"""Parsing of raw model output into thought / tool call / final answer.

The runtime uses a ReAct-style textual protocol so it works with *any*
provider, including ones without native function calling:

    Thought: I should check the weather first.
    Action: {"name": "weather", "arguments": {"city": "Guangzhou"}}
    Action: {"name": "todo", "arguments": {"action": "add", "item": "buy milk"}}

    Final Answer: It is sunny in Guangzhou and your todo is saved.

Design notes
------------
* Parsing is deliberately tolerant: it accepts several common spellings
  (``Final Answer:`` / ``Answer:`` / ``Final:``), a JSON object or a bare
  ``tool(args)`` form, and multiple actions in one turn.
* If the model emits no markers at all we treat the text as a final answer
  instead of failing -- an LLM that ignores the format should still be useful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .messages import ToolCall

_THOUGHT_RE = re.compile(r"^\s*(?:Thought|思考)\s*[:：]\s*(.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*Action\s*[:：]\s*(.*)$", re.IGNORECASE)
_FINAL_RE = re.compile(
    r"^\s*(?:Final\s*Answer|Answer|Final|最终答案|答案)\s*[:：]\s*(.*)$",
    re.IGNORECASE,
)
_ACTION_BLOCK_RE = re.compile(
    r"Action\s*[:：]\s*(\{.*?\}|[A-Za-z_][\w]*\s*\(.*?\))",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_BLOCK_RE = re.compile(
    r"(?:Final\s*Answer|Answer|Final|最终答案|答案)\s*[:：]\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedOutput:
    thought: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    raw: str = ""
    error: Optional[str] = None

    @property
    def has_action(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


def parse_llm_output(raw: str) -> ParsedOutput:
    """Turn one raw assistant string into a :class:`ParsedOutput`."""
    result = ParsedOutput(raw=raw or "")
    text = (raw or "").strip()
    if not text:
        result.error = "empty model output"
        result.final_answer = ""
        return result

    lines = text.splitlines()
    thought_parts: List[str] = []
    action_payloads: List[str] = []
    final_answer: Optional[str] = None

    for line in lines:
        m_final = _FINAL_RE.match(line)
        if m_final is not None:
            final_answer = m_final.group(1).strip()
            continue
        m_action = _ACTION_RE.match(line)
        if m_action is not None:
            action_payloads.append(m_action.group(1).strip())
            continue
        m_thought = _THOUGHT_RE.match(line)
        if m_thought is not None:
            thought_parts.append(m_thought.group(1).strip())
            continue
        # Continuation line: append to thought, or to final answer if started.
        if final_answer is not None:
            final_answer += "\n" + line
        else:
            thought_parts.append(line)

    # Multi-line / block forms when markers were not at line starts.
    if not action_payloads:
        action_payloads = [m.group(1) for m in _ACTION_BLOCK_RE.finditer(text)]
    if final_answer is None:
        m_block = _FINAL_BLOCK_RE.search(text)
        if m_block is not None:
            final_answer = m_block.group(1).strip()

    result.thought = "\n".join(p for p in thought_parts if p).strip()

    for payload in action_payloads:
        call, err = _parse_action(payload)
        if call is not None:
            result.tool_calls.append(call)
        else:
            result.error = err

    if final_answer is not None:
        result.final_answer = final_answer.strip()
    elif not result.tool_calls and not result.error:
        # No protocol markers at all -> treat the whole text as the answer.
        result.final_answer = text
        if not result.thought:
            result.thought = ""
    return result


def _parse_action(payload: str) -> "tuple[Optional[ToolCall], Optional[str]]":
    """Parse ``{"name": ..., "arguments": {...}}`` or ``name({...})``."""
    payload = payload.strip()
    if not payload:
        return None, "empty action payload"

    if payload.startswith("{"):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"invalid action JSON: {exc}"
        if not isinstance(obj, dict) or "name" not in obj:
            return None, "action JSON must contain a 'name' field"
        args = obj.get("arguments", obj.get("args", {})) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            return None, "action 'arguments' must be an object"
        return ToolCall(name=obj["name"], arguments=args), None

    # name({...}) / name(arg=value, ...)
    m = re.match(r"([A-Za-z_][\w]*)\s*\((.*)\)\s*$", payload, re.DOTALL)
    if not m:
        return None, f"unrecognised action syntax: {payload[:80]}"
    name, inner = m.group(1), m.group(2).strip()
    if not inner:
        return ToolCall(name=name, arguments={}), None
    if inner.startswith("{"):
        try:
            return ToolCall(name=name, arguments=json.loads(inner)), None
        except json.JSONDecodeError as exc:
            return None, f"invalid action arguments: {exc}"
    args: Dict[str, Any] = {}
    for pair in re.split(r",\s*(?=[A-Za-z_]\w*\s*=)", inner):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        args[key.strip()] = _coerce(value.strip())
    if not args:
        return None, f"could not parse action arguments: {inner[:80]}"
    return ToolCall(name=name, arguments=args), None


def _coerce(value: str) -> Any:
    value = value.strip().strip("'\"")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
