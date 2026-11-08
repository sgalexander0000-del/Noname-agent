"""The offline heuristic model must drive the real loop end-to-end."""

from noname_agent import AgentApp
from noname_agent.demo import HeuristicLLM
from noname_agent.session import InMemoryStore


def make_app():
    return AgentApp(llm=HeuristicLLM(), store=InMemoryStore(), todo_scope="session")


def test_weather_and_todo_in_one_turn():
    a = make_app()
    sid = a.new_session(user_id="A")
    result = a.chat("what is the weather in Xiamen and remind me to write the report", session_id=sid)
    names = [t["name"] for t in result.tool_calls]
    assert "weather" in names and "todo" in names
    assert [t["item"] for t in a.todos.list(sid)] == ["write the report"]


def test_calculator_route():
    a = make_app()
    result = a.chat("what is 12 * 8")
    assert result.tool_calls[-1]["name"] == "calculator"
    assert "96" in result.tool_calls[-1]["observation"]


def test_search_route():
    a = make_app()
    result = a.chat("search for ReAct agent")
    assert result.tool_calls[-1]["name"] == "search"


def test_docs_route():
    a = make_app()
    result = a.chat("read the docs")
    assert result.tool_calls[-1]["name"] == "read_docs"


def test_plain_chat_route():
    a = make_app()
    result = a.chat("hello there")
    assert result.tool_calls == []
    assert result.answer


def test_observations_from_previous_turns_not_reused():
    a = make_app()
    sid = a.new_session(user_id="A")
    a.chat("what is 2 + 2", session_id=sid)
    result = a.chat("hello", session_id=sid)
    # The second turn must not re-emit a calculator call from stale state.
    assert result.tool_calls == []
