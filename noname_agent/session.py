"""Session management.

A *session* is one conversation window.  Two windows belonging to the same
user are completely independent: they have their own message list, their own
summary, their own todo owner and their own trace.  Switching windows is just
``manager.get(session_id)``; nothing is shared implicitly.

Persistence is a plain JSON file (one directory of ``<session_id>.json``), so
"resume window 1 tomorrow" works without a database.  A simple in-memory
backend is also provided for tests.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .messages import Message, SessionState


class SessionStore(Protocol):
    def save(self, state: SessionState) -> None: ...
    def load(self, session_id: str) -> Optional[SessionState]: ...
    def list_ids(self) -> List[str]: ...
    def delete(self, session_id: str) -> bool: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._data: Dict[str, SessionState] = {}

    def save(self, state: SessionState) -> None:
        self._data[state.session_id] = state

    def load(self, session_id: str) -> Optional[SessionState]:
        return self._data.get(session_id)

    def list_ids(self) -> List[str]:
        return sorted(self._data)

    def delete(self, session_id: str) -> bool:
        return self._data.pop(session_id, None) is not None


class JsonFileStore:
    """One JSON document per session under ``root``."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.root / f"{safe}.json"

    def save(self, state: SessionState) -> None:
        path = self._path(state.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)  # atomic-ish write

    def load(self, session_id: str) -> Optional[SessionState]:
        path = self._path(session_id)
        if not path.is_file():
            return None
        return SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_ids(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False


class SessionManager:
    """Creates, fetches and lists sessions; owns their persistence."""

    def __init__(self, store: Optional[SessionStore] = None) -> None:
        self.store = store or InMemoryStore()
        self._cache: Dict[str, SessionState] = {}

    def create(
        self, session_id: Optional[str] = None, user_id: str = "default", title: str = ""
    ) -> SessionState:
        sid = session_id or uuid.uuid4().hex[:12]
        if self.store.load(sid) is not None:
            raise ValueError(f"session '{sid}' already exists")
        now = time.time()
        state = SessionState(
            session_id=sid, user_id=user_id, title=title, created_at=now, updated_at=now
        )
        self._cache[sid] = state
        self.store.save(state)
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id in self._cache:
            return self._cache[session_id]
        state = self.store.load(session_id)
        if state is None:
            raise KeyError(f"no such session: {session_id}")
        self._cache[session_id] = state
        return state

    def get_or_create(self, session_id: Optional[str], user_id: str = "default") -> SessionState:
        if session_id:
            try:
                return self.get(session_id)
            except KeyError:
                pass
        return self.create(session_id=session_id, user_id=user_id)

    def list(self, user_id: Optional[str] = None) -> List[SessionState]:
        states = []
        for sid in self.store.list_ids():
            state = self.store.load(sid)
            if state is None:
                continue
            if user_id is None or state.user_id == user_id:
                states.append(state)
        return sorted(states, key=lambda s: s.updated_at, reverse=True)

    def append(self, state: SessionState, *messages: Message) -> None:
        state.messages.extend(messages)
        state.updated_at = time.time()
        self.save(state)

    def save(self, state: SessionState) -> None:
        state.updated_at = time.time()
        self._cache[state.session_id] = state
        self.store.save(state)

    def delete(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        return self.store.delete(session_id)
