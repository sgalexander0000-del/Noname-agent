"""A tiny *heuristic* model so the agent can be demonstrated with no API key.

This is deliberately NOT an LLM: it is a keyword router that emits the same
ReAct protocol a real model would.  Its only purpose is to let a reviewer run
``python -m noname_agent.cli`` offline and watch the real loop, registry,
memory manager and session layer do their job.  With a key set, the CLI uses
the real model instead and this file is unused.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .messages import Message, Role


class HeuristicLLM:
    def __init__(self) -> None:
        self.calls: List[List[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))

        # Only look at the current turn: everything after the last real user
        # message.  Observations from earlier turns must not be reused.
        start = 0
        user_text = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role is Role.USER and msg.metadata.get("kind") != "protocol_reminder":
                user_text = msg.content
                start = i + 1
                break
        observations = [
            m.content for m in messages[start:] if m.role is Role.TOOL
        ]
        return self._decide(user_text, "\n".join(observations))

    # ---- routing -------------------------------------------------------
    def _decide(self, text: str, last_observation: str) -> str:
        lowered = text.lower()

        # Second pass: we already have observations -> summarise them.
        if last_observation:
            return f"Thought: I have the results.\nFinal Answer: {_summarise(last_observation)}"

        actions: List[str] = []

        # Weather?
        if "weather" in lowered or "天气" in lowered:
            m = re.search(r"(?:weather|天气).*?(?:in|for|在)?\s*([A-Za-z\u4e00-\u9fff]+)", lowered)
            city = _known_city(lowered) or (m.group(1) if m else "Guangzhou")
            actions.append(
                f'Action: {{"name": "weather", "arguments": {{"city": "{city.title()}"}}}}'
            )

        # Todo?
        if any(k in lowered for k in ["todo", "remind", "待办", "提醒", "记一下", "记待办"]):
            item = _todo_item(text)
            actions.append(
                f'Action: {{"name": "todo", "arguments": {{"action": "add", "item": "{item}"}}}}'
            )

        # Maths?
        if not actions and (re.search(r"\d\s*[-+*/x×]\s*\d", text) or "计算" in text):
            expr = _extract_expression(text)
            if expr:
                actions.append(
                    f'Action: {{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}'
                )

        # Docs?
        if not actions and ("docs" in lowered or "文档" in lowered):
            actions.append(
                'Action: {"name": "read_docs", "arguments": {"path": "architecture.md"}}'
            )

        # Search?
        if not actions and any(
            k in lowered for k in ["search", "find", "搜索", "查一下", "什么是", "what is"]
        ):
            actions.append(
                f'Action: {{"name": "search", "arguments": {{"query": "{_clip(text, 40)}"}}}}'
            )

        if actions:
            return "Thought: I will use the tools I need.\n" + "\n".join(actions)

        return (
            "Thought: No tool needed.\n"
            f"Final Answer: I received: {_clip(text, 200)}. Ask me to check the "
            "weather, do maths, search, read docs, or record a todo."
        )


def _known_city(text: str) -> Optional[str]:
    for city in ["guangzhou", "xiamen", "beijing", "shanghai", "广州", "厦门", "北京", "上海"]:
        if city in text:
            return {"广州": "Guangzhou", "厦门": "Xiamen", "北京": "Beijing", "上海": "Shanghai"}.get(
                city, city
            )
    return None


def _todo_item(text: str) -> str:
    # Prefer whatever follows an explicit reminder phrase.
    m = re.search(
        r"(?:remind me to|add a todo to|todo:?|提醒我|记一下|记待办[:：]?)\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    candidate = m.group(1) if m else text
    candidate = re.sub(r"^(please|and|then)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*(and|and then|并且|然后)\s.*$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(" ,.。，")
    return _clip(candidate or text, 60).replace('"', "'")


def _extract_expression(text: str) -> Optional[str]:
    expr = re.sub(r"[^0-9+\-*/().\s]", "", text.replace("x", "*").replace("×", "*"))
    expr = expr.strip()
    return expr or None


def _summarise(observation: str) -> str:
    return _clip(observation.replace("\n", " "), 300)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
