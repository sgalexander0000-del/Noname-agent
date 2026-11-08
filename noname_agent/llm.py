"""LLM access layer.

Two implementations:

* ``OpenAICompatibleLLM`` talks to any OpenAI-compatible ``/chat/completions``
  endpoint.  Credentials are read **only** from environment variables at call
  time -- the repo never contains an API key, and nothing is logged.
* ``MockLLM`` is a deterministic, scripted model used for the offline demo and
  the test-suite.  It lets us exercise the whole agent loop with zero network.

The runtime itself only depends on the small ``LLMClient`` protocol, so an
agent built on this code is provider-agnostic.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .messages import Message


class LLMClient(Protocol):
    """Minimal contract every model backend must satisfy."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """Return the raw assistant text for the given context."""


class LLMError(RuntimeError):
    """Raised when the provider call fails after retries."""


class OpenAICompatibleLLM:
    """Provider-agnostic client for OpenAI-style chat completion APIs.

    Reads configuration from the environment:

        OPENAI_API_KEY   (required)  -- never hard-coded
        OPENAI_BASE_URL  (optional)  -- defaults to api.openai.com/v1
        OPENAI_MODEL     (optional)  -- defaults to gpt-4o-mini
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _payload(
        self, messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_chat() for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise LLMError(
                "No API key configured. Set OPENAI_API_KEY in the environment "
                "(the key must never be committed to the repository)."
            )
        payload = self._payload(messages, tools)
        payload.update({k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens"}})

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._extract_text(body)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_err = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code < 500:  # 4xx will not fix itself
                    break
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last_err}")

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> str:
        """Normalise a chat-completions response into plain text.

        Native ``tool_calls`` are converted back into the textual ReAct
        protocol the parser understands, so the runtime has a single code path
        regardless of whether the provider speaks functions natively.
        """
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"Malformed response, no choices: {str(body)[:200]}")
        message = choices[0].get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            rendered = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                rendered.append(
                    json.dumps(
                        {"name": fn.get("name"), "arguments": _maybe_json(fn.get("arguments"))},
                        ensure_ascii=False,
                    )
                )
            text = (text + "\nAction: " + "\nAction: ".join(rendered)).strip()
        return text


def _maybe_json(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return raw or {}


class MockLLM:
    """Deterministic model for tests and offline demos.

    ``script`` is an ordered list of raw assistant strings.  When the script is
    exhausted the ``fallback`` callable (or a plain default) is used, so the
    agent can never hang waiting for a reply.
    """

    def __init__(
        self,
        script: Optional[Sequence[str]] = None,
        fallback: Optional[Callable[[Sequence[Message]], str]] = None,
    ) -> None:
        self.script: List[str] = list(script or [])
        self.fallback = fallback
        self.calls: List[List[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        if self.fallback is not None:
            return self.fallback(messages)
        return "Thought: No scripted response left.\nFinal Answer: (mock) done."


def build_llm_from_env(mock: bool = False, **kwargs: Any) -> LLMClient:
    """Factory used by the CLI: real client when a key exists, else mock."""
    if mock or not os.environ.get("OPENAI_API_KEY"):
        return MockLLM()
    return OpenAICompatibleLLM(**kwargs)
