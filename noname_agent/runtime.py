"""High-level application object wiring sessions, memory, tools and tracing.

``AgentApp`` is what a UI or CLI talks to.  It owns the long-lived pieces
(session store, shared todo store, memory config) and builds a short-lived
:class:`Agent` + tool registry for each session on demand.

Two windows for the same user therefore share only what the design intends
(nothing, by default) and keep their own message history, summary and trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import Agent, AgentResult
from .llm import LLMClient
from .memory import MemoryConfig, MemoryManager
from .session import InMemoryStore, JsonFileStore, SessionManager, SessionStore
from .tools import ToolRegistry, build_default_registry
from .tools.builtin import TodoStore
from .trace import Tracer


class AgentApp:
    def __init__(
        self,
        llm: LLMClient,
        store: Optional[SessionStore] = None,
        memory_config: Optional[MemoryConfig] = None,
        max_steps: int = 6,
        todo_scope: str = "user",
        docs_root: Optional[str] = None,
        echo_trace: bool = False,
    ) -> None:
        if todo_scope not in {"user", "session"}:
            raise ValueError("todo_scope must be 'user' or 'session'")
        self.llm = llm
        self.sessions = SessionManager(store=store or InMemoryStore())
        self.memory_config = memory_config or MemoryConfig()
        self.max_steps = max_steps
        self.todo_scope = todo_scope
        self.docs_root = docs_root
        self.echo_trace = echo_trace
        self.todos = TodoStore()
        self._registries: Dict[str, ToolRegistry] = {}
        self._todo_path = self._derive_todo_path(store)
        self._load_todos()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _derive_todo_path(store: Optional[SessionStore]) -> Optional[Path]:
        if isinstance(store, JsonFileStore):
            return store.root / "_todos.json"
        return None

    def _load_todos(self) -> None:
        if self._todo_path and self._todo_path.is_file():
            try:
                self.todos.load(json.loads(self._todo_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass

    def _save_todos(self) -> None:
        if not self._todo_path:
            return
        self._todo_path.parent.mkdir(parents=True, exist_ok=True)
        self._todo_path.write_text(
            json.dumps(self.todos.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def registry_for(self, state) -> ToolRegistry:
        """One registry per session, with the todo owner bound per scope."""
        if state.session_id in self._registries:
            return self._registries[state.session_id]
        owner = state.user_id if self.todo_scope == "user" else state.session_id
        registry = build_default_registry(todo_store=self.todos, docs_root=self.docs_root)
        # Re-bind the todo tool to this session's owner.
        from .tools.builtin import TodoTool

        registry.tools["todo"] = TodoTool(store=self.todos, owner=owner)
        self._registries[state.session_id] = registry
        return registry

    # ---- session API ---------------------------------------------------
    def new_session(self, user_id: str = "default", title: str = "") -> str:
        return self.sessions.create(user_id=user_id, title=title).session_id

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "title": s.title or "(untitled)",
                "messages": len(s.messages),
                "updated_at": s.updated_at,
            }
            for s in self.sessions.list(user_id=user_id)
        ]

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        state = self.sessions.get(session_id)
        return [m.to_dict() for m in state.messages]

    # ---- main entry point ----------------------------------------------
    def chat(
        self, user_input: str, session_id: Optional[str] = None, user_id: str = "default"
    ) -> AgentResult:
        state = self.sessions.get_or_create(session_id, user_id=user_id)
        tracer = Tracer(echo=self.echo_trace)
        agent = Agent(
            llm=self.llm,
            registry=self.registry_for(state),
            memory=MemoryManager(config=self.memory_config),
            tracer=tracer,
            max_steps=self.max_steps,
        )
        result = agent.run(state, user_input)
        self.sessions.save(state)
        self._save_todos()
        return result
