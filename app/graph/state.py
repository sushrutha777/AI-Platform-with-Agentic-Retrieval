"""LangGraph state schema."""

from typing import TypedDict, List, Dict, Any, Optional
from app.tools.base import ToolResult

class AgentState(TypedDict):
    """The state of the agent graph during request execution."""
    
    # Input
    original_query: str
    session_id: str
    
    # Context Processing
    rewritten_query: str
    
    # Routing
    intent: str
    tools_to_run: List[str]
    source_type: str
    
    # Retrieval
    retrieved_docs: List[ToolResult]
    context_text: str
    sources: List[Dict[str, Any]]
    
    # Output
    answer: str
    error: Optional[str]
    ttft_seconds: Optional[float]
    generation_latency_seconds: Optional[float]
