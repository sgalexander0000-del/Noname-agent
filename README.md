# Noname Agent — a minimal ReAct agent, from scratch

A small, dependency-free agent runtime built for the "从零实现一个最小可用 Agent"
brief. The loop, the output parser, the tool registry, the memory manager and
the session store are all implemented here — **no agent framework**
(langgraph / openhands / etc.) is used for the main flow. The only delegated
piece is the raw HTTP call to an OpenAI-compatible chat endpoint.

> **No API key is ever stored in this repository.** Credentials are read from
> environment variables at call time only. If no key is present the CLI runs a
> deterministic offline model so you can still watch the real loop work.

---

## 1. Quick start

Requires Python 3.9+. The runtime has **zero third-party dependencies**.

```bash
# 1. run the offline demo (no key, no network) — two windows, isolation, resume
python examples/demo_two_windows.py

# 2. interactive CLI (offline heuristic model when no key is set)
python -m noname_agent.cli

# 3. use a real model (key stays in your environment, never in the repo)
export OPENAI_API_KEY=sk-...                       # your key, not committed
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export OPENAI_MODEL=gpt-4o-mini                    # optional
python -m noname_agent.cli --live --data-dir ./data

# 4. tests (offline, deterministic)
pip install pytest
python -m pytest -q
```

CLI commands inside the REPL:

| command | meaning |
| --- | --- |
| `/new [title]` | open a new session (window) |
| `/sessions` | list your sessions |
| `/use <id>` | switch to another session |
| `/history` | print the current transcript |
| `/todos` | show the todo list |
| `/trace` | show the trace of the last turn |
| `/quit` | exit |

---

## 2. System design

### 2.1 Module map

```
noname_agent/
├── messages.py    Message / Role / ToolCall / SessionState  (the data model)
├── llm.py         LLMClient protocol, OpenAI-compatible client, MockLLM
├── demo.py        HeuristicLLM — offline stand-in for a real model
├── parser.py      raw model text  ->  thought / tool calls / final answer
├── memory.py      MemoryManager — context assembly + compression + recall
├── session.py     SessionManager + InMemory/JsonFile stores
├── trace.py       Tracer — structured execution log
├── agent.py       Agent — the ReAct loop itself
├── runtime.py     AgentApp — wires sessions + memory + tools + tracing
├── cli.py         interactive REPL
└── tools/
    ├── base.py    Tool, ToolRegistry, schema validation
    └── builtin.py calculator / search / weather / todo / read_docs
```

### 2.2 The loop

```
run(state, user_input):
  append user message
  for step in 1..max_steps:
      compress context if over budget         # memory.py
      context = [system] + [summary?] + [recent window]
      raw     = llm.complete(context, tools=schemas)
      parsed  = parse_llm_output(raw)          # thought / actions / final
      if final answer and no action:  return
      if no action (malformed):       nudge the model, continue
      for each action:  observation = registry.dispatch(name, args)
                        append tool message
  return "step limit reached"
```

This is exactly the four required steps: **(1)** receive input, **(2)** decide
reply vs tool call, **(3)** run the tool, **(4)** loop or answer.

### 2.3 Tools and the registry

Each tool declares a `name`, a `description` and a JSON-Schema `parameters`
object. `ToolRegistry.schemas()` renders those into OpenAI function
definitions, so the model chooses tools from the schema rather than from
hard-coded rules. Arguments are validated before execution; a bad call returns
`ERROR: ...` to the model instead of raising.

| tool | purpose |
| --- | --- |
| `calculator` | safe arithmetic via `ast` (no `eval` of user input) |
| `search` | mock web search over a deterministic corpus |
| `weather` | mock weather, deterministic per city, `c`/`f` units |
| `todo` | **required** — `add` / `list` / `complete` / `clear` |
| `read_docs` | read text files from a sandboxed `docs/` dir (path-traversal safe) |

### 2.4 Output parsing

`parse_llm_output` is tolerant: it accepts the ReAct protocol
(`Thought:` / `Action:` / `Final Answer:`), a `name(args)` shorthand, several
actions in one turn, and it treats a marker-less reply as a final answer so a
model that ignores the format is still useful. Native provider `tool_calls`
are normalised into the same textual protocol, so there is a single code path.

### 2.5 Session management

A session is one conversation window. Two windows for the same user are
completely independent: separate messages, summary, todo owner and trace.
`SessionManager` persists each session as one JSON file (`JsonFileStore`) or in
memory (`InMemoryStore`). "Resume window 1 tomorrow" is just
`app.chat(..., session_id=w1)` — see `examples/demo_two_windows.py`.

`todo_scope` decides whether todos are shared across a user's windows
(`"user"`, default) or private to each window (`"session"`).

### 2.6 Exception handling

* LLM failures → retried, then surfaced as a readable answer (never a crash).
* Tool errors → returned to the model as `ERROR: ...` so it can recover.
* Malformed model output → one nudge, then continue.
* Step-limit reached → partial answer + `truncated=True`.
* Every turn is wrapped so one bad turn cannot take down the process.

### 2.7 Trace / execution log

`Tracer` records `user_input`, `llm_request`, `llm_response`, `tool_call`
(with name, args, ok flag, latency, observation), `context_compressed`,
`parse_failure`, `final_answer`. Use `--trace` to echo it, read
`result.trace`, or `tracer.dump("run.jsonl")` for a file.

---

## 3. Memory: when we recall, what we place, and where

This is the part the brief calls out, so it gets its own section.

### 3.1 Recall timing

Recall happens **once per agent turn**, immediately before the LLM call, in
`MemoryManager.build_context`. There is no background job and no hidden state:
the prompt for a turn is a pure function of `(session state, memory config)`.
Compression is checked first, on the same tick (`maybe_compress`).

```
turn:  [ compress? ] -> [ build_context ] -> [ llm ] -> [ tools ] -> [ llm ] -> ...
```

### 3.2 Recall placement

The prompt is assembled in a fixed order:

```
1. system prompt      role + tool list + output protocol
2. running summary    older turns folded into ONE system message (if any)
3. recent window      the last N messages, verbatim
```

Putting the summary *after* the system prompt and *before* the window keeps the
instruction block stable (good for prompt caching) while keeping the freshest
detail last (closest to the generation point).

### 3.3 What we keep, and why

| content | in the prompt? | reasoning |
| --- | --- | --- |
| user turns | yes | the task itself |
| assistant reasoning (`Thought:`) | yes, while recent | it is the assistant content that precedes a tool call, and a real model needs it to stay coherent across the loop |
| assistant final answers | yes | they are the conversation |
| tool results | yes, verbatim while recent | they are the evidence the model reasons over |
| old turns | no — folded into the summary | bounded context, facts preserved |

Every assistant message also stores a copy of its reasoning in
`metadata["thought"]`. The prompt is built from the message content, so the
reasoning is present while the turn is in the recent window; once the turn
leaves the window it is only present (compressed) in the summary. The
`metadata` copy exists so the trace and the persisted history keep the full
audit trail even after compression drops the turn from the prompt.

### 3.4 Compression

When the estimated prompt size exceeds `max_context_tokens - reserve_tokens`,
`maybe_compress` moves everything outside the recent window into the running
summary and repeats until the budget is met. Two important details:

* **The window never splits a tool call from its observation.**
  `_safe_split` walks the split point left until it does not land on a `tool`
  message, so the request stays valid for providers that require pairing.
* **The summarizer is pluggable.** The default is a local, deterministic,
  extractive summarizer (no extra model call). `make_llm_summarizer(llm)`
  upgrades it to an LLM-written summary with an automatic fallback, so
  compression can never break the loop.

Token counting uses a cheap `len(text) // 4` estimate — deterministic and
sufficient for window/compression decisions, and easy to assert on in tests.

---

## 4. Tests

52 tests, all offline and deterministic (`python -m pytest -q`).

| file | covers |
| --- | --- |
| `test_parser.py` | thought/action/final parsing, multi-action, malformed JSON, plain prose |
| `test_tools.py` | registry + schema, all five tools, injection & traversal safety, validation |
| `test_agent_loop.py` | direct reply, single/multi tool, error recovery, max-steps, trace |
| `test_session_memory.py` | two-window isolation, persistence, compression, tool-pair safety |
| `test_followups.py` | pure-conversation follow-up, tool-using follow-up, context reuse |
| `test_heuristic_demo.py` | the offline model driving the real loop end-to-end |

The two-window scenario from the brief is asserted directly in
`test_two_windows_are_isolated` (window 1: weather + "buy milk"; window 2:
weekly report) and demonstrated in `examples/demo_two_windows.py`.

---

## 5. AI prompt & problem-solving log

See [`docs/PROMPTS.md`](docs/PROMPTS.md) for the prompts used, the design
questions they were used for, and the problems hit along the way.

## 6. Repository layout

```
noname_agent/   runtime package
docs/           architecture.md, faq.md, PROMPTS.md  (also the read_docs corpus)
examples/       demo_two_windows.py
tests/          pytest suite
```
