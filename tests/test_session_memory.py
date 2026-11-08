"""Session isolation, persistence, follow-ups and context compression."""

import json

from noname_agent import AgentApp, MockLLM
from noname_agent.memory import MemoryConfig, MemoryManager
from noname_agent.messages import Message, Role, SessionState
from noname_agent.session import InMemoryStore, JsonFileStore, SessionManager


# --------------------------------------------------------------------------
# session isolation
# --------------------------------------------------------------------------
def test_two_windows_are_isolated():
    """User A: window 1 (weather + todo) and window 2 (report + todo)."""
    llm = MockLLM(
        script=[
            # window 1
            'Action: {"name": "weather", "arguments": {"city": "Guangzhou"}}',
            'Action: {"name": "todo", "arguments": {"action": "add", "item": "buy milk"}}',
            "Final Answer: Guangzhou is 27C; todo saved.",
            # window 2
            'Action: {"name": "todo", "arguments": {"action": "add", "item": "write weekly report"}}',
            "Final Answer: Weekly report todo saved.",
        ]
    )
    app = AgentApp(llm=llm, store=InMemoryStore(), todo_scope="session")
    w1 = app.new_session(user_id="A", title="window-1")
    w2 = app.new_session(user_id="A", title="window-2")

    r1 = app.chat("check Guangzhou weather and remind me to buy milk", session_id=w1, user_id="A")
    r2 = app.chat("remind me to write the weekly report", session_id=w2, user_id="A")

    assert r1.session_id == w1 and r2.session_id == w2

    # Histories do not leak into each other.
    h1 = app.history(w1)
    h2 = app.history(w2)
    assert any("milk" in m["content"] for m in h1)
    assert not any("milk" in m["content"] for m in h2)
    assert any("report" in m["content"] for m in h2)

    # Todos are scoped per session, so the two windows stay independent.
    assert [t["item"] for t in app.todos.list(w1)] == ["buy milk"]
    assert [t["item"] for t in app.todos.list(w2)] == ["write weekly report"]


def test_resume_same_window_keeps_state():
    llm = MockLLM(
        script=[
            "Final Answer: Noted, you are preparing the weekly report.",
            "Final Answer: Yes, you mentioned the weekly report earlier.",
        ]
    )
    app = AgentApp(llm=llm, store=InMemoryStore())
    sid = app.new_session(user_id="A")
    app.chat("I am preparing the weekly report", session_id=sid, user_id="A")
    app.chat("what was I doing?", session_id=sid, user_id="A")
    history = app.history(sid)
    # 2 user turns + 2 assistant turns
    assert len(history) == 4
    # The second LLM call must have seen the first turn in context.
    second_call = llm.calls[-1]
    joined = " ".join(m.content for m in second_call)
    assert "weekly report" in joined


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def test_json_persistence_round_trip(tmp_path):
    store = JsonFileStore(tmp_path)
    manager = SessionManager(store=store)
    state = manager.create(user_id="A", title="persisted")
    manager.append(state, Message(role=Role.USER, content="hello"))

    fresh = SessionManager(store=JsonFileStore(tmp_path))
    reloaded = fresh.get(state.session_id)
    assert reloaded.title == "persisted"
    assert reloaded.messages[0].content == "hello"

    raw = json.loads((tmp_path / f"{state.session_id}.json").read_text())
    assert raw["messages"][0]["role"] == "user"


def test_session_manager_lists_and_deletes(tmp_path):
    manager = SessionManager(store=JsonFileStore(tmp_path))
    a = manager.create(user_id="A")
    b = manager.create(user_id="B")
    assert {s.session_id for s in manager.list()} == {a.session_id, b.session_id}
    assert {s.session_id for s in manager.list(user_id="A")} == {a.session_id}
    assert manager.delete(a.session_id) is True
    assert manager.delete(a.session_id) is False


# --------------------------------------------------------------------------
# memory / compression
# --------------------------------------------------------------------------
def test_compression_folds_old_turns_into_summary():
    config = MemoryConfig(max_context_tokens=120, reserve_tokens=20, keep_recent_messages=4)
    manager = MemoryManager(config=config)
    state = SessionState(session_id="s")
    for i in range(40):
        state.messages.append(Message(role=Role.USER, content=f"user message number {i} " * 5))
        state.messages.append(Message(role=Role.ASSISTANT, content=f"answer {i} " * 5))

    assert manager.estimate_tokens(state.messages) > manager.budget()
    assert manager.maybe_compress(state) is True
    assert state.summary
    assert manager.estimate_tokens(state.messages) <= manager.budget()
    assert len(state.messages) <= config.keep_recent_messages


def test_compression_never_splits_tool_call_from_observation():
    config = MemoryConfig(max_context_tokens=80, reserve_tokens=10, keep_recent_messages=3)
    manager = MemoryManager(config=config)
    state = SessionState(session_id="s")
    for i in range(20):
        state.messages.append(Message(role=Role.USER, content=f"q{i} " * 10))
        state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content="thinking",
                tool_calls=[],
                metadata={"kind": "tool_call"},
            )
        )
        state.messages.append(
            Message(role=Role.TOOL, name="calculator", tool_call_id="c", content="42 " * 10)
        )
    manager.maybe_compress(state)
    # The window must never begin with an orphan tool observation.
    assert not state.messages or state.messages[0].role is not Role.TOOL


def test_build_context_order_system_summary_window():
    config = MemoryConfig(max_context_tokens=100, reserve_tokens=20, keep_recent_messages=2)
    manager = MemoryManager(config=config)
    state = SessionState(session_id="s", summary="earlier we discussed X")
    state.messages = [
        Message(role=Role.USER, content="last question"),
        Message(role=Role.ASSISTANT, content="last answer"),
    ]
    context = manager.build_context(state, "SYSTEM PROMPT")
    assert context[0].role is Role.SYSTEM
    assert context[0].content == "SYSTEM PROMPT"
    assert context[1].role is Role.SYSTEM
    assert "earlier we discussed X" in context[1].content
    assert context[2].content == "last question"


def test_follow_up_with_tool_after_compression():
    """A long conversation still answers a tool-using follow-up."""
    config = MemoryConfig(max_context_tokens=60, reserve_tokens=20, keep_recent_messages=4)
    script = []
    for i in range(10):
        script.append(f"Final Answer: reply {i} with some extra words to add tokens")
    script.append('Action: {"name": "calculator", "arguments": {"expression": "3*3"}}')
    script.append("Final Answer: 9")
    app = AgentApp(llm=MockLLM(script=script), store=InMemoryStore(), memory_config=config)
    sid = app.new_session(user_id="A")
    for i in range(10):
        app.chat(f"filler message {i} " * 3, session_id=sid, user_id="A")
    result = app.chat("now compute 3*3", session_id=sid, user_id="A")
    assert result.answer == "9"
    assert result.tool_calls[-1]["name"] == "calculator"
    # Compression happened at some point during the long conversation.
    assert any(e["kind"] == "context_compressed" for e in result.trace) or app.sessions.get(sid).summary
