from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class SourceCitation(BaseModel):
    """Retrieved context source citation."""
    title: str
    source_type: str  # 'document', 'wikipedia', 'web'
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    url: Optional[str] = None
    score: Optional[float] = None
    content: str


class ChatMessageSchema(BaseModel):
    """Message payload."""
    id: Optional[str] = None
    role: str
    content: str
    tool_used: Optional[str] = None
    source_type: Optional[str] = None
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None
    sources: Optional[List[SourceCitation]] = None
    latency_seconds: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    """Chat message request."""
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


class RegenerateRequest(BaseModel):
    """Regenerate request."""
    conversation_id: str
    message_id: Optional[str] = None
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


class StreamStepEvent(BaseModel):
    """Step progression update in SSE stream."""
    type: str = "step"
    node: str
    label: str
    tool_used: Optional[str] = None
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None


class StreamMetadataEvent(BaseModel):
    """Metadata event emitted before token streaming."""
    type: str = "metadata"
    tool_used: Optional[str] = None
    source_type: Optional[str] = None
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None
    sources: List[SourceCitation] = []


class StreamDoneEvent(BaseModel):
    """Final stream event."""
    type: str = "done"
    full_answer: str
    tool_used: Optional[str] = None
    source_type: Optional[str] = None
    latency_seconds: float
    total_latency_seconds: Optional[float] = None
    ttft_seconds: Optional[float] = None
    generation_latency_seconds: Optional[float] = None
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None
