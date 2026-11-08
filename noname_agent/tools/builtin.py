"""The five built-in tools.

* ``calculator``  -- safe arithmetic evaluation (no ``eval`` of raw input).
* ``search``      -- mock web search over a small deterministic corpus.
* ``weather``     -- mock weather lookup, deterministic per city.
* ``todo``        -- the mandatory tool: add / list / complete / clear.
* ``read_docs``   -- read text files from a sandboxed docs directory.

All tools are pure-ish and side-effect free except ``todo``, whose state lives
in an injected :class:`TodoStore` so it can be persisted per user.
"""

from __future__ import annotations

import ast
import hashlib
import operator
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError

# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("expression contains an unsupported operation")


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "Evaluate an arithmetic expression. Supports + - * / // % ** and "
                "parentheses. Use this for any maths instead of guessing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '(12 + 3) * 4 / 2'",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            func=self._calculate,
        )

    @staticmethod
    def _calculate(expression: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"invalid expression: {exc.msg}") from exc
        value = _eval_node(tree)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return {"expression": expression, "result": value}


# --------------------------------------------------------------------------
# search (mock)
# --------------------------------------------------------------------------
_SEARCH_CORPUS: List[Dict[str, str]] = [
    {
        "title": "Python (programming language)",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "snippet": "Python is a high-level, interpreted language emphasising readability.",
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting",
        "url": "https://arxiv.org/abs/2210.03629",
        "snippet": "ReAct interleaves reasoning traces with tool actions for LLM agents.",
    },
    {
        "title": "Guangzhou weather overview",
        "url": "https://example.com/guangzhou-weather",
        "snippet": "Guangzhou has a humid subtropical climate with warm, wet summers.",
    },
    {
        "title": "Xiamen travel guide",
        "url": "https://example.com/xiamen",
        "snippet": "Xiamen is a coastal city in Fujian known for Gulangyu island.",
    },
    {
        "title": "Todo list best practices",
        "url": "https://example.com/todo-best-practices",
        "snippet": "Keep todos small, concrete and time-boxed to stay actionable.",
    },
]


class SearchTool(Tool):
    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None) -> None:
        self.corpus = corpus if corpus is not None else _SEARCH_CORPUS
        super().__init__(
            name="search",
            description=(
                "Search the (mock) web for a query and return ranked snippets "
                "with URLs. Use when you need external facts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search keywords"},
                    "top_k": {
                        "type": "integer",
                        "description": "number of results, default 3",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            func=self._search,
        )

    def _search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        terms = [t for t in query.lower().split() if t]
        scored = []
        for doc in self.corpus:
            haystack = (doc["title"] + " " + doc["snippet"]).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
        hits = [doc for _, doc in scored[: max(1, int(top_k))]]
        if not hits:
            return {"query": query, "results": [], "note": "no matches in mock corpus"}
        return {"query": query, "results": hits}


# --------------------------------------------------------------------------
# weather (mock, deterministic)
# --------------------------------------------------------------------------
_CONDITIONS = ["sunny", "cloudy", "light rain", "overcast", "clear"]


class WeatherTool(Tool):
    def __init__(self, known_cities: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.known_cities = known_cities or {
            "guangzhou": {"temperature_c": 27, "condition": "light rain", "humidity": 78},
            "xiamen": {"temperature_c": 25, "condition": "cloudy", "humidity": 70},
            "beijing": {"temperature_c": 18, "condition": "clear", "humidity": 40},
            "shanghai": {"temperature_c": 22, "condition": "overcast", "humidity": 65},
        }
        super().__init__(
            name="weather",
            description="Get the (mock) current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "city name, e.g. Guangzhou"},
                    "unit": {
                        "type": "string",
                        "description": "temperature unit: c or f (default c)",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            func=self._weather,
        )

    def _weather(self, city: str, unit: str = "c") -> Dict[str, Any]:
        key = city.strip().lower()
        data = self.known_cities.get(key)
        if data is None:
            # Deterministic pseudo-weather so unknown cities still answer.
            digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
            data = {
                "temperature_c": 10 + digest % 25,
                "condition": _CONDITIONS[digest % len(_CONDITIONS)],
                "humidity": 40 + digest % 50,
            }
        temp = data["temperature_c"]
        if unit.lower().startswith("f"):
            temp = round(temp * 9 / 5 + 32)
            unit_label = "F"
        else:
            unit_label = "C"
        return {
            "city": city,
            "temperature": f"{temp}°{unit_label}",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
        }


# --------------------------------------------------------------------------
# todo (mandatory)
# --------------------------------------------------------------------------
class TodoStore:
    """In-memory todo state, keyed per owner (e.g. user or session)."""

    def __init__(self) -> None:
        self._items: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, owner: str, item: str) -> Dict[str, Any]:
        item = item.strip()
        if not item:
            raise ToolError("todo item must not be empty")
        bucket = self._items.setdefault(owner, [])
        entry = {"id": len(bucket) + 1, "item": item, "done": False}
        bucket.append(entry)
        return entry

    def list(self, owner: str) -> List[Dict[str, Any]]:
        return list(self._items.get(owner, []))

    def complete(self, owner: str, item_id: int) -> Dict[str, Any]:
        for entry in self._items.get(owner, []):
            if entry["id"] == int(item_id):
                entry["done"] = True
                return entry
        raise ToolError(f"no todo with id {item_id} for owner '{owner}'")

    def clear(self, owner: str) -> int:
        return len(self._items.pop(owner, []))

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {owner: list(items) for owner, items in self._items.items()}

    def load(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self._items = {owner: list(items) for owner, items in (data or {}).items()}


class TodoTool(Tool):
    def __init__(self, store: Optional[TodoStore] = None, owner: str = "default") -> None:
        self.store = store or TodoStore()
        self.owner = owner
        super().__init__(
            name="todo",
            description=(
                "Manage a todo list. actions: add (needs 'item'), list, "
                "complete (needs 'item_id'), clear. Always use this instead of "
                "claiming a task was recorded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "one of: add, list, complete, clear",
                    },
                    "item": {"type": "string", "description": "text for action=add"},
                    "item_id": {
                        "type": "integer",
                        "description": "todo id for action=complete",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            func=self._todo,
        )

    def _todo(self, action: str, item: Optional[str] = None, item_id: Optional[int] = None) -> Any:
        action = (action or "").strip().lower()
        if action == "add":
            if not item:
                raise ToolError("action 'add' requires 'item'")
            entry = self.store.add(self.owner, item)
            return {"status": "added", "todo": entry, "total": len(self.store.list(self.owner))}
        if action == "list":
            items = self.store.list(self.owner)
            return {"count": len(items), "todos": items}
        if action == "complete":
            if item_id is None:
                raise ToolError("action 'complete' requires 'item_id'")
            entry = self.store.complete(self.owner, item_id)
            return {"status": "completed", "todo": entry}
        if action == "clear":
            removed = self.store.clear(self.owner)
            return {"status": "cleared", "removed": removed}
        raise ToolError("action must be one of: add, list, complete, clear")


# --------------------------------------------------------------------------
# read_docs
# --------------------------------------------------------------------------
class ReadDocsTool(Tool):
    def __init__(self, root: Optional[str | os.PathLike] = None) -> None:
        self.root = Path(root or Path(__file__).resolve().parents[2] / "docs").resolve()
        super().__init__(
            name="read_docs",
            description=(
                "Read a text document from the local docs directory. Use for "
                "project documentation questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "relative file path inside the docs dir",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "truncate the file to this many chars",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            func=self._read,
        )

    def _read(self, path: str, max_chars: int = 4000) -> Dict[str, Any]:
        candidate = (self.root / path).resolve()
        # Prevent path traversal outside the docs sandbox.
        if self.root not in candidate.parents and candidate != self.root:
            raise ToolError("path escapes the docs directory")
        if not candidate.is_file():
            raise ToolError(f"document not found: {path}")
        text = candidate.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {
            "path": path,
            "content": text[:max_chars],
            "truncated": truncated,
        }

    def list_docs(self) -> List[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file()
        )
