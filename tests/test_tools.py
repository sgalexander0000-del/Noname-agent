import pytest

from noname_agent.tools import Tool, ToolRegistry, build_default_registry
from noname_agent.tools.builtin import TodoStore, TodoTool


def test_default_registry_has_required_tools():
    registry = build_default_registry()
    for name in ["calculator", "search", "weather", "todo", "read_docs"]:
        assert name in registry.names()


def test_schemas_are_well_formed():
    registry = build_default_registry()
    for schema in registry.schemas():
        fn = schema["function"]
        assert fn["name"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_calculator_precedence_and_parens():
    registry = build_default_registry()
    out = registry.dispatch("calculator", {"expression": "(2 + 3) * 4"})
    assert '"result": 20' in out


def test_calculator_rejects_code_injection():
    registry = build_default_registry()
    out = registry.dispatch("calculator", {"expression": "__import__('os').system('ls')"})
    assert out.startswith("ERROR:")


def test_calculator_division_by_zero_is_error_not_crash():
    registry = build_default_registry()
    out = registry.dispatch("calculator", {"expression": "1 / 0"})
    assert out.startswith("ERROR:")


def test_weather_deterministic_for_unknown_city():
    registry = build_default_registry()
    a = registry.dispatch("weather", {"city": "Atlantis"})
    b = registry.dispatch("weather", {"city": "Atlantis"})
    assert a == b


def test_weather_unit_conversion():
    registry = build_default_registry()
    out = registry.dispatch("weather", {"city": "Guangzhou", "unit": "f"})
    assert "°F" in out


def test_search_returns_results():
    registry = build_default_registry()
    out = registry.dispatch("search", {"query": "ReAct agent"})
    assert "arxiv.org/abs/2210.03629" in out


def test_todo_lifecycle():
    store = TodoStore()
    tool = TodoTool(store=store, owner="u1")
    tool.run({"action": "add", "item": "write report"})
    tool.run({"action": "add", "item": "buy milk"})
    listed = tool.run({"action": "list"})
    assert '"count": 2' in listed
    tool.run({"action": "complete", "item_id": 1})
    after = tool.run({"action": "list"})
    assert '"done": true' in after


def test_todo_isolated_per_owner():
    store = TodoStore()
    a = TodoTool(store=store, owner="A")
    b = TodoTool(store=store, owner="B")
    a.run({"action": "add", "item": "A task"})
    assert '"count": 0' in b.run({"action": "list"})


def test_unknown_tool_returns_error_string():
    registry = build_default_registry()
    out = registry.dispatch("does_not_exist", {})
    assert out.startswith("ERROR: unknown tool")


def test_validation_missing_required_argument():
    registry = build_default_registry()
    out = registry.dispatch("calculator", {})
    assert "missing required argument 'expression'" in out


def test_validation_wrong_type():
    registry = build_default_registry()
    out = registry.dispatch("calculator", {"expression": 123})
    assert "must be string" in out


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    tool = Tool(name="t", description="d", parameters={}, func=lambda: "ok")
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)


def test_read_docs_reads_file(tmp_path):
    (tmp_path / "note.txt").write_text("hello docs", encoding="utf-8")
    registry = build_default_registry(docs_root=tmp_path)
    out = registry.dispatch("read_docs", {"path": "note.txt"})
    assert "hello docs" in out


def test_read_docs_blocks_traversal(tmp_path):
    registry = build_default_registry(docs_root=tmp_path)
    out = registry.dispatch("read_docs", {"path": "../../etc/passwd"})
    assert out.startswith("ERROR:")
