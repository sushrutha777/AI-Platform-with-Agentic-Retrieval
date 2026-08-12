"""Chat orchestration and streaming service."""

import time
import json
from typing import AsyncGenerator
from app.schemas.chat import ChatRequest
from app.agents.orchestrator import AgentOrchestrator
from app.tools.base import ToolRegistry
from app.tools.retriever_tool import DocumentRetrieverTool
from app.tools.wikipedia_tool import WikipediaSearchTool
from app.tools.web_search_tool import WebSearchTool
from app.retriever.base import BaseRetriever
from app.retriever.sparse import SparseBM25Retriever
from app.retriever.hybrid import HybridRetriever
from app.reranker.flashrank_reranker import FlashRankReranker
from app.reranker.null_reranker import NullReranker
from app.core.config import settings
from app.core.logging import logger


class ChatService:
    """Orchestrates the new AgentOrchestrator pipeline."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: SparseBM25Retriever,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.sparse_retriever)

        # Reranker
        if settings.USE_RERANKER:
            self.reranker = FlashRankReranker()
        else:
            self.reranker = NullReranker()

        # Tool Registry
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(DocumentRetrieverTool(self.hybrid_retriever, self.reranker))
        self.tool_registry.register(WikipediaSearchTool())
        self.tool_registry.register(WebSearchTool())

        # Initialize new orchestrator
        self.orchestrator = AgentOrchestrator(self.tool_registry)

    async def stream_chat(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream SSE formatted events."""
        session_id = request.conversation_id or "default_session"
        
        if request.chat_history:
            from app.context.service import context_service
            context_service.restore_session(session_id, request.chat_history)

        try:
            async for event in self.orchestrator.stream_chat(
                question=request.question,
                session_id=session_id,
            ):
                event_type = event.get("type", "step")
                
                # Yield SSE chunk
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"Error in stream_chat: {e}", exc_info=True)
            err_event = {"type": "error", "error": str(e)}
            yield f"event: error\ndata: {json.dumps(err_event)}\n\n"
