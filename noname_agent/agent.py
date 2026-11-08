"""The agent runtime: the ReAct loop, implemented from scratch.

Loop shape (matches the brief):

    step 1  receive user input            -> append to session
    step 2  decide reply vs tool call     -> ask the LLM, parse its output
    step 3  execute the tool(s)           -> via the registry, with tracing
    step 4  continue the loop or answer   -> until a Final Answer or max steps

Everything else (session isolation, memory, tracing) is injected, so the loop
itself stays small and readable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLMClient, LLMError
from .memory import MemoryManager
from .messages import Message, Role, SessionState, ToolCall
from .parser import parse_llm_output
from .tools import ToolRegistry
from .trace import Tracer

DEFAULT_SYSTEM_PROMPT = """You are a minimal tool-using assistant.

You work in a loop. On every turn you either call tools or give the final answer.

Available tools:
{tools}

Reply using EXACTLY this protocol:

Thought: <your private reasoning>
Action: {{"name": "<tool_name>", "arguments": {{...}}}}
   (you may emit several Action lines to call several tools at once)

When you are ready to answer the user, stop calling tools and write:

Thought: <reasoning>
Final Answer: <the answer for the user>

Rules:
- Use a tool whenever a fact, calculation or state change is needed. Never invent tool results.
- After an Action you will receive an Observation. Keep looping until you can answer.
- Ask a short clarifying question via Final Answer if the request is ambiguous.
- Answer in the same language the user used.
"""


@dataclass
class AgentResult:
    session_id: str
    answer: str
    steps: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    truncated: bool = False
    error: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: Optional[ToolRegistry] = None,
        memory: Optional[MemoryManager] = None,
        tracer: Optional[Tracer] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
    ) -> None:
        self.llm = llm
        self.registry = registry if registry is not None else ToolRegistry()
        self.memory = memory or MemoryManager()
        self.tracer = tracer or Tracer()
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    # ---- prompt --------------------------------------------------------
    def render_system_prompt(self) -> str:
        template = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        return template.format(tools=self.registry.describe() or "(none)")

    # ---- public API ----------------------------------------------------
    def run(self, state: SessionState, user_input: str) -> AgentResult:
        """Handle one user turn against ``state`` and mutate it in place."""
        tracer = self.tracer.child()
        result = AgentResult(session_id=state.session_id, answer="")

        user_msg = Message(role=Role.USER, content=user_input)
        state.messages.append(user_msg)
        if not state.title:
            state.title = user_input.strip()[:60]
        tracer.log("user_input", session_id=state.session_id, text=user_input)

        try:
            self._loop(state, result, tracer)
        except LLMError as exc:
            result.error = str(exc)
            result.answer = f"I could not reach the language model: {exc}"
            tracer.log("llm_error", error=str(exc))
        except Exception as exc:  # never let a turn crash the whole app
            result.error = f"{type(exc).__name__}: {exc}"
            result.answer = "Something went wrong while handling that request."
            tracer.log("unhandled_error", error=result.error)

        if not result.answer:
            result.answer = (
                "I reached the step limit without a final answer. "
                "Here is what I found so far; please refine your request."
            )
            result.truncated = True
            tracer.log("max_steps_reached", steps=result.steps)

        state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=result.answer,
                metadata={"kind": "final", "truncated": result.truncated},
            )
        )
        state.updated_at = time.time()
        tracer.log("final_answer", session_id=state.session_id, answer=result.answer)
        result.trace = tracer.to_list()
        return result

    # ---- the loop ------------------------------------------------------
    def _loop(self, state: SessionState, result: AgentResult, tracer: Tracer) -> None:
        for step in range(1, self.max_steps + 1):
            result.steps = step

            before = self.memory.estimate_tokens(state.messages)
            if self.memory.maybe_compress(state):
                tracer.log(
                    "context_compressed",
                    before_tokens=before,
                    after_tokens=self.memory.estimate_tokens(state.messages),
                    summary_chars=len(state.summary),
                )

            context = self.memory.build_context(state, self.render_system_prompt())
            tracer.log(
                "llm_request",
                step=step,
                messages=len(context),
                approx_tokens=self.memory.estimate_tokens(context),
            )

            raw = self.llm.complete(context, tools=self.registry.schemas())
            parsed = parse_llm_output(raw)
            tracer.log(
                "llm_response",
                step=step,
                thought=parsed.thought,
                actions=[tc.to_dict() for tc in parsed.tool_calls],
                final=parsed.final_answer,
                error=parsed.error,
            )

            # (2) decide: final answer vs tool call ---------------------
            if parsed.is_final and not parsed.has_action:
                result.answer = parsed.final_answer or ""
                if parsed.thought:
                    result.thoughts.append(parsed.thought)
                return

            if not parsed.has_action:
                # Malformed output: nudge the model once, then bail out.
                tracer.log("parse_failure", raw=raw[:300], error=parsed.error)
                state.messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=raw,
                        metadata={"kind": "malformed", "thought": parsed.thought},
                    )
                )
                state.messages.append(
                    Message(
                        role=Role.USER,
                        content=(
                            "Your previous reply did not follow the protocol. "
                            "Reply with either 'Action: {...}' or 'Final Answer: ...'."
                        ),
                        metadata={"kind": "protocol_reminder"},
                    )
                )
                continue

            # (3) execute the tools --------------------------------------
            if parsed.thought:
                result.thoughts.append(parsed.thought)
            state.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=parsed.thought or "(calling tools)",
                    tool_calls=parsed.tool_calls,
                    metadata={"kind": "tool_call", "thought": parsed.thought},
                )
            )

            for call in parsed.tool_calls:
                observation = self._execute(call, tracer, result)
                state.messages.append(
                    Message(
                        role=Role.TOOL,
                        name=call.name,
                        tool_call_id=call.call_id,
                        content=observation,
                        metadata={"kind": "observation"},
                    )
                )
            # (4) loop again

    # ---- tool execution ------------------------------------------------
    def _execute(self, call: ToolCall, tracer: Tracer, result: AgentResult) -> str:
        start = time.time()
        observation = self.registry.dispatch(call.name, call.arguments)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        ok = not observation.startswith("ERROR:")
        record = {
            "name": call.name,
            "arguments": call.arguments,
            "ok": ok,
            "elapsed_ms": elapsed_ms,
            "observation": observation,
        }
        result.tool_calls.append(record)
        tracer.log("tool_call", **record)
        return observation
