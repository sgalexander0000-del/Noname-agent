# AGENTS.md

Repository: a from-scratch minimal ReAct agent (`noname_agent/`).

## Commands

- Run tests: `python -m pytest -q` (offline, deterministic, no key needed).
- Lint: `python -m pyflakes noname_agent tests examples`.
- Demo: `python examples/demo_two_windows.py` (no key, no network).
- CLI: `python -m noname_agent.cli` (falls back to the offline heuristic model).

## Conventions

- **No API keys in the repo, ever.** Credentials come from `OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, `OPENAI_MODEL` env vars, read at call time only. Never
  log request bodies or auth headers.
- **Zero third-party runtime dependencies.** Standard library only; `pytest`
  is a dev-only extra. Do not add a framework or SDK to the main flow.
- Provider-agnostic: the internal protocol is the textual ReAct form
  (`Thought` / `Action` / `Final Answer`). Native `tool_calls` are normalised
  into it at the LLM client boundary (`llm.py`). Keep a single code path.
- Tool failures are returned to the model as `ERROR: ...` strings
  (`ToolRegistry.dispatch`); they must never raise out of the loop.
- Tests use `MockLLM` (scripted) or `HeuristicLLM` (offline router); they must
  not hit the network.

## Layout

`messages.py` (model) · `llm.py` + `demo.py` (backends) · `parser.py` ·
`tools/` · `memory.py` (context/compression) · `session.py` · `trace.py` ·
`agent.py` (the loop) · `runtime.py` (wiring) · `cli.py`.
