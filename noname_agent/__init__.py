"""A minimal, from-scratch ReAct agent runtime.

Public surface:

    from noname_agent import Agent, SessionManager, ToolRegistry

No third-party agent framework is used anywhere in the runtime; the
LLM provider is the only pluggable external dependency.
"""

from .agent import Agent, AgentResult
from .llm import LLMClient, MockLLM, OpenAICompatibleLLM, build_llm_from_env
from .memory import MemoryConfig, MemoryManager
from .messages import Message, Role, SessionState, ToolCall
from .parser import ParsedOutput, parse_llm_output
from .runtime import AgentApp
from .session import InMemoryStore, JsonFileStore, SessionManager
from .tools import Tool, ToolError, ToolRegistry, build_default_registry
from .trace import Tracer

__all__ = [
    "Agent",
    "AgentResult",
    "AgentApp",
    "MemoryConfig",
    "InMemoryStore",
    "JsonFileStore",
    "ToolError",
    "LLMClient",
    "MockLLM",
    "OpenAICompatibleLLM",
    "build_llm_from_env",
    "MemoryManager",
    "Message",
    "Role",
    "SessionState",
    "ToolCall",
    "ParsedOutput",
    "parse_llm_output",
    "SessionManager",
    "Tool",
    "ToolRegistry",
    "build_default_registry",
    "Tracer",
]

__version__ = "0.1.0"
