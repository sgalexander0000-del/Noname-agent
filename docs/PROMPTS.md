# AI prompt & problem-solving record

This file documents how AI tooling was used to build this agent, the prompts
that drove the work, and the concrete problems that came up (and how they were
solved). The intent is to show *how* the model was used to think — not to hand
it the task.

---

## 1. How AI was used

Three modes, in order of how much thinking they saved:

1. **Design sparring.** Ask the model to argue *for and against* a design
   choice before writing code (e.g. textual ReAct protocol vs native function
   calling). The decision stayed with me; the model supplied the trade-offs.
2. **Adversarial review.** After each module, ask "what breaks this?" and
   "which test would catch it?". Several of the tests below came from this.
3. **Boilerplate.** Schema plumbing, dataclass serialization, CLI argument
   parsing — mechanical code where the model is fast and the risk is low.

The model was **not** asked to "write an agent" in one shot. Handing it the
whole task produces something that looks right and fails on the details the
brief is actually testing (session isolation, context placement, error
recovery). Those were designed and verified by hand.

---

## 2. Prompts that were actually used

### 2.1 Design: protocol choice

> I am implementing a ReAct agent runtime from scratch with no framework. The
> model backend may or may not support native function calling. Argue the
> trade-offs of (a) a textual `Thought/Action/Final Answer` protocol parsed by
> my own parser vs (b) relying on native `tool_calls`. Which is more robust
> for a "works with any OpenAI-compatible endpoint" requirement, and what
> failure modes does each have?

Outcome: textual protocol as the single internal representation, with native
`tool_calls` normalised into it at the client boundary. One code path,
provider-agnostic, and the parser is directly unit-testable.

### 2.2 Design: context assembly order

> For a chat agent with a running summary and a recent-message window, what is
> the best order to place: system prompt, summary, recent window? Consider
> prompt caching and the "recency" effect on generation. Give the reasoning,
> not just the answer.

Outcome: `system -> summary -> window`. Stable instruction prefix, freshest
detail nearest the generation point. Documented in README §3.2.

### 2.3 Design: what to keep in context

> Here is my message model (user / assistant / tool, plus assistant
> `tool_calls` and a `metadata` field). For each kind of message, should the
> assistant's private reasoning (the `Thought:` text) be included in the next
> prompt? Give the argument for dropping it and the argument for keeping it.

Outcome: reasoning is the assistant content that precedes a tool call, so it
must stay in the prompt while the turn is in the recent window (a real model
needs it to stay coherent across the loop). A copy is stored in
`metadata["thought"]` so the trace and persisted history keep the full audit
trail even after compression drops the turn from the prompt. Net effect: the
prompt stays bounded, the audit trail does not.

### 2.4 Review: what breaks this loop?

> Here is my agent loop and tool registry. List the failure modes that would
> make it hang, crash, or silently return a wrong answer, in priority order.
> For each, give me the smallest test that would catch it.

Outcome: produced the max-steps test, the tool-error-recovery test, the
malformed-output test, and the LLM-failure test. All four are in
`tests/test_agent_loop.py`.

### 2.5 Review: context compression edge case

> When I compress an old conversation window, what invariant must I preserve
> for OpenAI-style chat APIs that use tool calls? What happens if I violate it
> and how do I test for it?

Outcome: an assistant message with `tool_calls` must stay adjacent to its
`tool` replies. `_safe_split` never lands the window boundary on a `tool`
message; `test_compression_never_splits_tool_call_from_observation` asserts it.

---

## 3. Problems hit and how they were solved

### P1 — Calculator must not `eval` user input

**Problem.** The obvious `eval(expression)` is a remote-code-execution hole:
`__import__('os').system('...')` from a model-chosen argument.

**Solution.** Parse with `ast.parse(..., mode="eval")` and walk the tree,
allowing only numeric constants, `+ - * / // % **` and unary signs. Anything
else raises `ToolError`. Tested by
`test_calculator_rejects_code_injection`.

### P2 — `read_docs` path traversal

**Problem.** A model could ask for `../../etc/passwd`.

**Solution.** Resolve the candidate path and require it to stay inside the
docs root. Tested by `test_read_docs_blocks_traversal`.

### P3 — A tool error must not kill the loop

**Problem.** `1/0` raised inside the tool and propagated out of `run`.

**Solution.** `ToolRegistry.dispatch` catches `ToolError` and returns
`ERROR: ...` as the observation, which is fed back to the model so it can
recover. `test_tool_error_is_fed_back_and_recovered` and
`test_unknown_tool_recovery` cover it.

### P4 — Malformed model output

**Problem.** The model sometimes emits a half-written `Action:` line, or
ignores the protocol entirely.

**Solution.** The parser reports `error` instead of raising; the loop appends
a short protocol reminder and retries once. Separately, a *marker-less* reply
is treated as a final answer, so a chatty model still works. This is a
deliberate split: broken *action* syntax is recoverable, plain prose is a
legitimate answer. (`test_malformed_action_gets_nudged_then_succeeds`,
`test_plain_prose_is_accepted_as_final_answer`.)

### P5 — The loop could run forever

**Problem.** A model that keeps calling tools with no final answer.

**Solution.** `max_steps` bounds the loop; on exhaustion the agent returns a
partial answer with `truncated=True` and logs `max_steps_reached`.
`test_max_steps_truncation` asserts the agent stops at the limit rather than
hanging.

### P6 — Two windows leaking into each other

**Problem.** The brief's core scenario: window 1 (weather + todo) and window 2
(weekly report + todo) for the same user must not interfere.

**Solution.** State lives on the `SessionState`, not on the app. The app keeps
one registry per session with the todo tool bound to that session's owner.
`test_two_windows_are_isolated` asserts both histories *and* both todo lists
stay separate, under `todo_scope="session"`.

### P7 — Compressing away the tool evidence

**Problem.** Early compression attempts produced a window that began with an
orphan `tool` message, which real providers reject.

**Solution.** `_safe_split` walks the boundary left until it is not on a
`tool` message. Tested by
`test_compression_never_splits_tool_call_from_observation`.

### P8 — Re-running the demo reused stale observations

**Problem.** The offline heuristic model summarised tool output from a
*previous* turn, so "resume window 1" replayed an old weather result.

**Solution.** The heuristic only considers messages after the last real user
message. `test_observations_from_previous_turns_not_reused` guards it. (A good
example of a bug that only shows up when you actually run the demo end to end
rather than trusting the unit tests.)

### P9 — Never ship a key

**Problem.** The brief explicitly forbids API keys in the code.

**Solution.** No key literal exists anywhere. `OpenAICompatibleLLM` reads
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from the environment at
call time; if the key is absent the CLI uses the offline heuristic model. The
client also never logs request bodies or headers. `.gitignore` excludes
`.env`. (Grep the repo: the only occurrences of "OPENAI_API_KEY" are
`os.environ.get(...)` reads and documentation.)

---

## 4. What I would do next with more time

* Streaming responses and a token-accurate counter instead of `len/4`.
* Real vector recall over the summary for very long sessions.
* A `human-in-the-loop` confirmation gate for side-effecting tools
  (the `Tool.requires_confirmation` flag is already there).
* Parallel tool execution (the loop already supports multiple actions per
  turn; they are currently run sequentially).
