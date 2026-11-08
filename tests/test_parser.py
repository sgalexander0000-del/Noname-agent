from noname_agent.parser import parse_llm_output


def test_thought_action_final():
    raw = (
        "Thought: I need the weather.\n"
        'Action: {"name": "weather", "arguments": {"city": "Guangzhou"}}\n'
    )
    parsed = parse_llm_output(raw)
    assert parsed.thought == "I need the weather."
    assert parsed.has_action
    assert parsed.tool_calls[0].name == "weather"
    assert parsed.tool_calls[0].arguments == {"city": "Guangzhou"}
    assert parsed.final_answer is None


def test_final_answer_only():
    parsed = parse_llm_output("Thought: trivial.\nFinal Answer: 42")
    assert parsed.final_answer == "42"
    assert not parsed.has_action


def test_multiple_actions_in_one_turn():
    raw = (
        "Thought: do both.\n"
        'Action: {"name": "weather", "arguments": {"city": "Xiamen"}}\n'
        'Action: {"name": "todo", "arguments": {"action": "add", "item": "call mom"}}\n'
    )
    parsed = parse_llm_output(raw)
    assert [tc.name for tc in parsed.tool_calls] == ["weather", "todo"]


def test_function_call_syntax():
    parsed = parse_llm_output("Action: calculator(expression=\"2 + 3 * 4\")")
    assert parsed.tool_calls[0].name == "calculator"
    assert parsed.tool_calls[0].arguments == {"expression": "2 + 3 * 4"}


def test_malformed_json_reports_error():
    parsed = parse_llm_output('Action: {"name": "calculator", ')
    assert parsed.error is not None
    assert not parsed.tool_calls


def test_plain_text_becomes_final_answer():
    parsed = parse_llm_output("hello there")
    assert parsed.final_answer == "hello there"


def test_multiline_final_answer():
    parsed = parse_llm_output("Final Answer: line one\nline two")
    assert parsed.final_answer == "line one\nline two"


def test_empty_output():
    parsed = parse_llm_output("   ")
    assert parsed.error == "empty model output"
    assert parsed.final_answer == ""
