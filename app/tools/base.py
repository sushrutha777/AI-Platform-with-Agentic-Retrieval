"""Base tool definition and registry."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Structured tool execution output."""
    tool_name: str
    source_type: str  # 'document', 'wikipedia', 'web'
    output: str
    sources: List[Dict[str, Any]] = []


class BaseAgentTool(ABC):
    """Abstract base class for all agentic tools."""

    name: str
    description: str

    @abstractmethod
    async def run(self, query: str) -> ToolResult:
        """Execute the tool with given query asynchronously."""
        pass


class ToolRegistry:
    """Registry managing available agentic tools."""

    def __init__(self):
        self._tools: Dict[str, BaseAgentTool] = {}

    def register(self, tool: BaseAgentTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseAgentTool]:
        """Get tool by name."""
        return self._tools.get(name)


