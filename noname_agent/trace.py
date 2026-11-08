"""Lightweight structured tracing / execution log.

Every meaningful step of the loop (llm call, tool call, compression, error,
final answer) is recorded as an :class:`TraceEvent`.  Events can be printed,
returned as part of the agent result, or dumped to a JSONL file for later
inspection -- which is what the "tool call trace / execution log" requirement
asks for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TraceEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "timestamp": self.timestamp, "data": self.data}

    def format(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        if len(payload) > 240:
            payload = payload[:237] + "..."
        return f"[{self.kind}] {payload}"


class Tracer:
    def __init__(
        self,
        echo: bool = False,
        sink: Optional[Callable[[TraceEvent], None]] = None,
        max_events: int = 500,
    ) -> None:
        self.events: List[TraceEvent] = []
        self.echo = echo
        self.sink = sink
        self.max_events = max_events

    def log(self, kind: str, **data: Any) -> TraceEvent:
        event = TraceEvent(kind=kind, data=data)
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[0]
        if self.echo:
            print("  " + event.format())
        if self.sink is not None:
            try:
                self.sink(event)
            except Exception:
                pass
        return event

    def child(self) -> "Tracer":
        """A tracer that shares this one's sink but keeps its own event list."""
        return Tracer(echo=self.echo, sink=self.sink, max_events=self.max_events)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
