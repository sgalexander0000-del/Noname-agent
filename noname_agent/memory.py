"""Context / memory management.

This is the module that answers "what actually goes into the prompt, and
when?".  The policy is intentionally simple and explainable:

Recall timing
    Recall happens once per agent turn, right before the LLM call
    (:meth:`MemoryManager.build_context`).  There is no background process and
    no hidden state: the context for a turn is a pure function of the session
    state plus the memory config.

Recall placement
    The prompt is assembled in a fixed order:

        1. system prompt      -- role, tool list, output protocol
        2. running summary    -- older turns folded into one system message
        3. recent window      -- the last ``keep_recent_messages`` messages

What is kept
    * user turns              -- always kept (the task itself)
    * assistant final answers -- kept (they are the conversation)
    * tool results            -- kept verbatim while recent, because they are
                                 the evidence the model reasons over
    * assistant thoughts      -- kept in ``metadata`` but stripped from the
                                 *prompt* once a turn leaves the recent window
                                 (they are useful for debugging, cheap to drop
                                 for the model)

Compression
    When the estimated prompt size exceeds the budget we fold everything
    outside the recent window into the running summary.  The summarizer is
    pluggable; by default it is a local, deterministic, extractive summarizer
    so the whole system works with no extra model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .messages import Message, Role, SessionState


@dataclass
class MemoryConfig:
    max_context_tokens: int = 3000
    reserve_tokens: int = 400        # headroom for the model's own reply
    keep_recent_messages: int = 8    # size of the verbatim window
    summary_max_chars: int = 1200
    min_keep_messages: int = 2


Summarizer = Callable[[str, Sequence[Message]], str]


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.summarizer = summarizer or self._local_summarize
        self.compression_count = 0

    # ---- context assembly ---------------------------------------------
    def build_context(self, state: SessionState, system_prompt: str) -> List[Message]:
        """Assemble the exact message list handed to the LLM for this turn."""
        self.maybe_compress(state)
        context: List[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
        if state.summary:
            context.append(
                Message(
                    role=Role.SYSTEM,
                    content="Conversation summary so far:\n" + state.summary,
                    metadata={"kind": "summary"},
                )
            )
        context.extend(state.messages)
        return context

    # ---- budget --------------------------------------------------------
    def estimate_tokens(self, messages: Sequence[Message]) -> int:
        return sum(m.approx_tokens() for m in messages)

    def budget(self) -> int:
        return max(1, self.config.max_context_tokens - self.config.reserve_tokens)

    def maybe_compress(self, state: SessionState) -> bool:
        """Fold old turns into the summary until the prompt fits the budget."""
        compressed = False
        while (
            self.estimate_tokens(state.messages) > self.budget()
            and len(state.messages) > self.config.min_keep_messages
        ):
            keep = min(self.config.keep_recent_messages, len(state.messages) - 1)
            keep = max(self.config.min_keep_messages, keep)
            split = self._safe_split(state.messages, len(state.messages) - keep)
            if split <= 0:
                break
            old, state.messages = state.messages[:split], state.messages[split:]
            state.summary = self.summarizer(state.summary, old)
            self.compression_count += 1
            compressed = True
            if keep <= self.config.min_keep_messages:
                break
        return compressed

    @staticmethod
    def _safe_split(messages: List[Message], split: int) -> int:
        """Move ``split`` left so the recent window never starts mid tool-call.

        An assistant message with ``tool_calls`` must stay together with the
        ``tool`` messages that answer it, otherwise providers reject the
        request.
        """
        split = max(0, min(split, len(messages)))
        while split > 0 and split < len(messages) and messages[split].role is Role.TOOL:
            split -= 1
        return split

    # ---- summarizers ---------------------------------------------------
    def _local_summarize(self, previous: str, old: Sequence[Message]) -> str:
        """Deterministic extractive summary -- no extra LLM call required."""
        lines: List[str] = []
        for msg in old:
            text = (msg.content or "").strip().replace("\n", " ")
            if msg.role is Role.USER:
                lines.append(f"- user: {_clip(text, 160)}")
            elif msg.role is Role.ASSISTANT and not msg.tool_calls:
                lines.append(f"- agent: {_clip(text, 160)}")
            elif msg.role is Role.TOOL:
                lines.append(f"- {msg.name or 'tool'} -> {_clip(text, 100)}")
        body = "\n".join(lines)
        combined = (previous + "\n" + body).strip() if previous else body
        if len(combined) > self.config.summary_max_chars:
            combined = combined[-self.config.summary_max_chars :]
        return combined

    def make_llm_summarizer(self, llm) -> Summarizer:
        """Optional higher-quality summarizer backed by the LLM itself.

        Falls back to the local summarizer if the call fails, so compression
        can never break the loop.
        """

        def _summarize(previous: str, old: Sequence[Message]) -> str:
            transcript = "\n".join(
                f"{m.role.value}: {_clip(m.content, 400)}" for m in old
            )
            prompt = [
                Message(role=Role.SYSTEM, content="You compress conversation history."),
                Message(
                    role=Role.USER,
                    content=(
                        "Update the running summary. Keep concrete facts, names, "
                        "numbers, decisions and unfinished tasks.\n\n"
                        f"Previous summary:\n{previous or '(none)'}\n\n"
                        f"New turns:\n{transcript}"
                    ),
                ),
            ]
            try:
                text = llm.complete(prompt).strip()
                return text[: self.config.summary_max_chars] if text else self._local_summarize(previous, old)
            except Exception:
                return self._local_summarize(previous, old)

        return _summarize


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
