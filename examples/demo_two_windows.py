"""Runnable demo of the two-window scenario from the brief.

    python examples/demo_two_windows.py

User A opens window 1 (weather + todo) and window 2 (weekly report + todo).
The script shows that the two sessions, their histories and their todos stay
independent, and that both can be resumed later.

Runs fully offline: no API key, no network.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noname_agent import AgentApp
from noname_agent.demo import HeuristicLLM
from noname_agent.session import JsonFileStore


def line(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    store = JsonFileStore("examples/.sessions")
    app = AgentApp(llm=HeuristicLLM(), store=store, todo_scope="session", echo_trace=False)

    line("User A opens window 1 and window 2")
    w1 = app.new_session(user_id="A", title="window-1")
    w2 = app.new_session(user_id="A", title="window-2")
    print(f"window 1 = {w1}\nwindow 2 = {w2}")

    line("Window 1: check weather + record a todo")
    r = app.chat("what is the weather in Guangzhou and remind me to buy milk",
                 session_id=w1, user_id="A")
    print("agent>", r.answer)

    line("Window 2: record a different todo")
    r = app.chat("remind me to write the weekly report", session_id=w2, user_id="A")
    print("agent>", r.answer)

    line("Todos are isolated per window (todo_scope='session')")
    print("window 1 todos:", [t["item"] for t in app.todos.list(w1)])
    print("window 2 todos:", [t["item"] for t in app.todos.list(w2)])

    line("Histories are isolated per window")
    print("window 1 history:")
    for m in app.history(w1):
        print(f"   [{m['role']}] {m['content'][:80]}")
    print("window 2 history:")
    for m in app.history(w2):
        print(f"   [{m['role']}] {m['content'][:80]}")

    line("Resume window 1 later (fresh app instance, same store)")
    app2 = AgentApp(llm=HeuristicLLM(), store=JsonFileStore("examples/.sessions"),
                    todo_scope="session")
    r = app2.chat("remind me to call the client", session_id=w1, user_id="A")
    print("agent>", r.answer)
    print("window 1 todos now:", [t["item"] for t in app2.todos.list(w1)])


if __name__ == "__main__":
    main()
