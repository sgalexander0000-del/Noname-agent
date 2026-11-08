"""Built-in tools and the default registry factory."""

from __future__ import annotations

from .base import Tool, ToolError, ToolRegistry
from .builtin import (
    CalculatorTool,
    ReadDocsTool,
    SearchTool,
    TodoTool,
    WeatherTool,
)


def build_default_registry(todo_store=None, docs_root=None) -> ToolRegistry:
    """Registry pre-loaded with the five required tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    registry.register(TodoTool(store=todo_store))
    registry.register(ReadDocsTool(root=docs_root))
    return registry


__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
    "WeatherTool",
    "TodoTool",
    "ReadDocsTool",
    "build_default_registry",
]
