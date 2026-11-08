"""Follow-up questions: pure-conversation and tool-using."""

from noname_agent import AgentApp, MockLLM
from noname_agent.session import InMemoryStore


def test_pure_conversation_follow_up_keeps_context():
    """Turn 2 refers to turn 1; the context handed to the LLM must include it."""
    llm = MockLLM(
        script=[
            "Final Answer: Great, the project is called Falcon.",
            "Final Answer: The project you mentioned is Falcon.",
        ]
    )
    app = AgentApp(llm=llm, store=InMemoryStore())
    sid = app.new_session(user_id="A")
    app.chat("My project is called Falcon.", session_id=sid, user_id="A")
    result = app.chat("What is my project called?", session_id=sid, user_id="A")

    assert result.answer == "The project you mentioned is Falcon."
    second_context = " ".join(m.content for m in llm.calls[-1])
    assert "Falcon" in second_context
    assert "My project is called Falcon." in second_context


def test_tool_using_follow_up():
    """Turn 1 records a todo; turn 2 asks to list it via the todo tool."""
    llm = MockLLM(
        script=[
            'Action: {"name": "todo", "arguments": {"action": "add", "item": "book flights"}}',
            "Final Answer: Saved.",
            'Action: {"name": "todo", "arguments": {"action": "list"}}',
            "Final Answer: You have one todo: book flights.",
        ]
    )
    app = AgentApp(llm=llm, store=InMemoryStore())
    sid = app.new_session(user_id="A")
    app.chat("remind me to book flights", session_id=sid, user_id="A")
    result = app.chat("what is on my list?", session_id=sid, user_id="A")

    assert result.answer == "You have one todo: book flights."
    assert result.tool_calls[-1]["name"] == "todo"
    assert "book flights" in result.tool_calls[-1]["observation"]


def test_follow_up_reuses_previous_tool_result_without_recomputing():
    """The weather observation from turn 1 is still in context on turn 2."""
    llm = MockLLM(
        script=[
            'Action: {"name": "weather", "arguments": {"city": "Xiamen"}}',
            "Final Answer: It is cloudy in Xiamen.",
            "Final Answer: You asked about Xiamen; it was cloudy.",
        ]
    )
    app = AgentApp(llm=llm, store=InMemoryStore())
    sid = app.new_session(user_id="A")
    app.chat("weather in Xiamen?", session_id=sid, user_id="A")
    result = app.chat("which city did I ask about?", session_id=sid, user_id="A")

    assert "Xiamen" in result.answer
    # The tool observation from turn 1 is visible in the turn-2 context.
    turn2_context = " ".join(m.content for m in llm.calls[-1])
    assert "Xiamen" in turn2_context
    assert "cloudy" in turn2_context


def test_history_shape_after_two_turns():
    llm = MockLLM(script=["Final Answer: a", "Final Answer: b"])
    app = AgentApp(llm=llm, store=InMemoryStore())
    sid = app.new_session(user_id="A")
    app.chat("first", session_id=sid, user_id="A")
    app.chat("second", session_id=sid, user_id="A")
    roles = [m["role"] for m in app.history(sid)]
    assert roles == ["user", "assistant", "user", "assistant"]
