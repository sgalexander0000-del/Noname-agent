"""End-to-end tests of the ReAct loop using the deterministic MockLLM."""

from noname_agent import AgentApp, MockLLM
from noname_agent.session import InMemoryStore


def make_app(script, **kwargs):
    return AgentApp(llm=MockLLM(script=script), store=InMemoryStore(), **kwargs)


def test_direct_reply_without_tools():
    app = make_app(["Thought: easy.\nFinal Answer: Hello! How can I help?"])
    result = app.chat("hi")
    assert result.answer == "Hello! How can I help?"
    assert result.steps == 1
    assert result.tool_calls == []


def test_single_tool_then_answer():
    script = [
        'Thought: compute.\nAction: {"name": "calculator", "arguments": {"expression": "6*7"}}',
        "Thought: done.\nFinal Answer: The answer is 42.",
    ]
    app = make_app(script)
    result = app.chat("what is 6 times 7?")
    assert result.answer == "The answer is 42."
    assert [t["name"] for t in result.tool_calls] == ["calculator"]
    assert result.steps == 2


def test_multi_step_multi_tool_loop():
    script = [
        'Thought: weather first.\nAction: {"name": "weather", "arguments": {"city": "Guangzhou"}}',
        'Thought: now remember.\nAction: {"name": "todo", "arguments": {"action": "add", "item": "buy umbrella"}}',
        "Thought: done.\nFinal Answer: It is rainy in Guangzhou; added a todo to buy an umbrella.",
    ]
    app = make_app(script)
    result = app.chat("weather in Guangzhou and remind me to take an umbrella")
    assert [t["name"] for t in result.tool_calls] == ["weather", "todo"]
    assert result.steps == 3
    assert app.todos.list("default")[0]["item"] == "buy umbrella"


def test_tool_error_is_fed_back_and_recovered():
    script = [
        'Thought: try.\nAction: {"name": "calculator", "arguments": {"expression": "1/0"}}',
        "Thought: that failed, answer honestly.\nFinal Answer: I cannot divide by zero.",
    ]
    app = make_app(script)
    result = app.chat("what is 1 divided by 0?")
    assert result.answer == "I cannot divide by zero."
    assert result.tool_calls[0]["ok"] is False
    assert "ERROR" in result.tool_calls[0]["observation"]


def test_unknown_tool_recovery():
    script = [
        'Thought: wrong tool.\nAction: {"name": "teleport", "arguments": {}}',
        "Thought: recover.\nFinal Answer: I do not have that tool.",
    ]
    app = make_app(script)
    result = app.chat("teleport me")
    assert result.answer == "I do not have that tool."
    assert "unknown tool" in result.tool_calls[0]["observation"]


def test_max_steps_truncation():
    # Never emits a Final Answer -> must stop at max_steps, not hang.
    loop_script = [
        'Action: {"name": "calculator", "arguments": {"expression": "1+1"}}'
        for _ in range(10)
    ]
    app = make_app(loop_script, max_steps=3)
    result = app.chat("loop forever")
    assert result.truncated is True
    assert result.steps == 3
    assert "step limit" in result.answer


def test_malformed_action_gets_nudged_then_succeeds():
    script = [
        'Action: {"name": "calculator", ',   # broken JSON -> nudge + retry
        "Thought: ok.\nFinal Answer: Fixed.",
    ]
    app = make_app(script)
    result = app.chat("say hi")
    assert result.answer == "Fixed."
    assert result.steps == 2
    assert any(e["kind"] == "parse_failure" for e in result.trace)


def test_plain_prose_is_accepted_as_final_answer():
    # A model that ignores the protocol but just answers should still be useful.
    app = make_app(["Sure, the answer is 4."])
    result = app.chat("what is 2+2?")
    assert result.answer == "Sure, the answer is 4."
    assert result.steps == 1


def test_llm_failure_is_handled_gracefully():
    class BoomLLM:
        def complete(self, messages, tools=None, **kwargs):
            raise RuntimeError("network down")

    app = AgentApp(llm=BoomLLM(), store=InMemoryStore())
    result = app.chat("hello")
    assert result.error is not None
    assert "went wrong" in result.answer


def test_trace_records_tool_calls():
    script = [
        'Action: {"name": "calculator", "arguments": {"expression": "2+2"}}',
        "Final Answer: 4",
    ]
    app = make_app(script)
    result = app.chat("2+2")
    kinds = [e["kind"] for e in result.trace]
    assert "tool_call" in kinds
    assert "llm_request" in kinds
    assert "final_answer" in kinds
    tool_event = next(e for e in result.trace if e["kind"] == "tool_call")
    assert tool_event["data"]["name"] == "calculator"
    assert tool_event["data"]["ok"] is True
    assert "elapsed_ms" in tool_event["data"]
