"""Interactive CLI.

Run it:

    python -m noname_agent.cli                 # uses MockLLM if no key set
    OPENAI_API_KEY=... python -m noname_agent.cli --live

Commands inside the REPL:

    /new [title]      start a new session (window)
    /sessions         list your sessions
    /use <id>         switch to another session
    /history          show the current session transcript
    /todos            show the current todo list
    /trace            show the trace of the last turn
    /help, /quit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from .demo import HeuristicLLM
from .llm import OpenAICompatibleLLM
from .memory import MemoryConfig
from .runtime import AgentApp
from .session import JsonFileStore


def build_app(args) -> AgentApp:
    if args.mock or not os.environ.get("OPENAI_API_KEY"):
        llm = HeuristicLLM()
        mode = "offline heuristic model (no OPENAI_API_KEY found)"
    else:
        llm = OpenAICompatibleLLM()
        mode = f"live model={os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"

    store = JsonFileStore(Path(args.data_dir).expanduser()) if args.data_dir else None
    app = AgentApp(
        llm=llm,
        store=store,
        memory_config=MemoryConfig(
            max_context_tokens=args.max_context_tokens,
            keep_recent_messages=args.keep_recent,
        ),
        max_steps=args.max_steps,
        todo_scope=args.todo_scope,
        echo_trace=args.trace,
    )
    print(f"[noname-agent] mode: {mode}; persistence: {args.data_dir or 'in-memory'}")
    return app


def repl(app: AgentApp, user_id: str = "default") -> None:
    session_id = app.new_session(user_id=user_id, title="window-1")
    last_trace = []
    print(f"[noname-agent] active session: {session_id}  (type /help for commands)")

    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            arg = arg.strip()
            if cmd in {"quit", "exit", "q"}:
                print("bye.")
                return
            if cmd == "help":
                print(__doc__)
            elif cmd == "new":
                session_id = app.new_session(user_id=user_id, title=arg or "untitled")
                print(f"switched to new session: {session_id}")
            elif cmd == "sessions":
                for s in app.list_sessions(user_id=user_id):
                    print(f"  {s['session_id']}  {s['title']!r}  msgs={s['messages']}")
            elif cmd == "use":
                try:
                    app.sessions.get(arg)
                    session_id = arg
                    print(f"switched to session: {session_id}")
                except KeyError:
                    print(f"no such session: {arg}")
            elif cmd == "history":
                for m in app.history(session_id):
                    print(f"  [{m['role']}] {m['content'][:120]}")
            elif cmd == "todos":
                items = app.todos.list(user_id)
                print(items or "(empty)")
            elif cmd == "trace":
                for ev in last_trace:
                    print("  " + str(ev))
            else:
                print(f"unknown command: /{cmd}")
            continue

        result = app.chat(text, session_id=session_id, user_id=user_id)
        last_trace = result.trace
        print(f"\nagent> {result.answer}")
        if result.tool_calls:
            names = ", ".join(tc["name"] for tc in result.tool_calls)
            print(f"  (steps={result.steps}, tools=[{names}])")
        if result.error:
            print(f"  (error: {result.error})")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal from-scratch ReAct agent")
    parser.add_argument("--live", action="store_true", help="force the real LLM client")
    parser.add_argument("--mock", action="store_true", help="force the mock LLM")
    parser.add_argument("--data-dir", default="", help="persist sessions here (JSON files)")
    parser.add_argument("--user", default="default", help="user id")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-context-tokens", type=int, default=3000)
    parser.add_argument("--keep-recent", type=int, default=8)
    parser.add_argument("--todo-scope", choices=["user", "session"], default="user")
    parser.add_argument("--trace", action="store_true", help="echo the execution trace")
    parser.add_argument("--once", default="", help="run a single prompt then exit")
    args = parser.parse_args(argv)

    app = build_app(args)
    if args.once:
        result = app.chat(args.once, user_id=args.user)
        print(result.answer)
        if args.trace:
            for ev in result.trace:
                print("  " + str(ev))
        return
    repl(app, user_id=args.user)


if __name__ == "__main__":
    main()
