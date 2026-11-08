"""Core data structures for the agent runtime.

Everything the agent ever sends to an LLM is a ``Message``.  Keeping the
message model tiny and explicit makes the "what goes into context" question
easy to answer and easy to test.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "call_id": self.call_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}) or {},
            call_id=data.get("call_id") or uuid.uuid4().hex[:8],
        )


@dataclass
class Message:
    """A single entry in the conversation context.

    ``metadata`` is used by the memory manager to make compression decisions
    (e.g. a compacted summary is marked so it is never compacted again).
    """

    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            metadata=data.get("metadata", {}) or {},
        )

    # ---- LLM-facing views ---------------------------------------------
    def to_chat(self) -> Dict[str, Any]:
        """Render as an OpenAI-style chat message."""
        msg: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.role is Role.TOOL:
            msg["tool_call_id"] = self.tool_call_id or self.name or "tool"
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    def approx_tokens(self) -> int:
        """Cheap token estimate: ~4 chars/token, minimum 1.

        Good enough for window/compression decisions and, importantly,
        deterministic so tests can assert on it.
        """
        size = len(self.content or "")
        for tc in self.tool_calls:
            size += len(json.dumps(tc.arguments, ensure_ascii=False)) + len(tc.name)
        return max(1, size // 4)


@dataclass
class SessionState:
    """Persisted state of one conversation window."""

    session_id: str
    user_id: str = "default"
    title: str = ""
    messages: List[Message] = field(default_factory=list)
    summary: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id", "default"),
            title=data.get("title", ""),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            summary=data.get("summary", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            metadata=data.get("metadata", {}) or {},
        )
