"""Chat orchestration and streaming service."""

import time
import json
import asyncio
from contextlib import suppress
from typing import AsyncGenerator
from app.schemas.chat import ChatRequest
from app.graph import build_graph
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
from app.guardrails.model_armor import GuardrailResult, guardrail


INPUT_BLOCKED_MESSAGE = (
    "Your request could not be processed because it did not pass the application's security policy. "
    "Please try to rephrase your request."
)
INPUT_SCAN_ERROR_MESSAGE = (
    "Your request could not be processed because the application's security service is unavailable."
)


class ChatService:
    """Orchestrates the LangGraph pipeline."""

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

        # Initialize LangGraph
        self.graph = build_graph(self.tool_registry)

    async def stream_chat(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream SSE formatted events."""
        start_time = time.time()
        session_id = request.conversation_id or "default_session"

        # Model Armor must see the raw latest user message before session
        # restoration, contextual rewriting, routing, or retrieval begins.
        try:
            prompt_check = await asyncio.to_thread(
                guardrail.sanitize_user_prompt,
                request.question,
            )
        except Exception as exc:
            logger.error(
                "MODEL_ARMOR_ERROR stage=input reason=unexpected_guardrail_exception error_type=%s",
                type(exc).__name__,
            )
            prompt_check = GuardrailResult(
                allowed=False,
                reason="api_unavailable",
                error=True,
            )

        if not prompt_check.allowed:
            blocked_message = (
                INPUT_SCAN_ERROR_MESSAGE if prompt_check.error else INPUT_BLOCKED_MESSAGE
            )
            yield f"event: metadata\ndata: {json.dumps({'type': 'metadata', 'intent': 'blocked', 'tool_used': 'none', 'source_type': 'guardrail', 'sources': []})}\n\n"
            yield f"event: token\ndata: {json.dumps({'type': 'token', 'token': blocked_message})}\n\n"
            yield f"event: done\ndata: {json.dumps({'type': 'done', 'full_answer': blocked_message, 'tool_used': 'none', 'source_type': 'guardrail', 'latency_seconds': round(time.time() - start_time, 2)})}\n\n"
            return
        
        if request.chat_history:
            from app.context.service import context_service
            context_service.restore_session(session_id, request.chat_history)

        try:
            stream_queue = asyncio.Queue()
            
            state = {
                "original_query": request.question,
                "session_id": session_id
            }

            async def run_graph():
                try:
                    final_state = await self.graph.ainvoke(
                        state, 
                        config={
                            "run_name": "agentic_rag_request",
                            "tags": ["agentic-rag", "langgraph", "sse"],
                            "metadata": {
                                "conversation_id": session_id,
                                "transport": "sse",
                            },
                            "configurable": {"stream_queue": stream_queue},
                        }
                    )
                    
                    # Compute latency
                    latency_seconds = round(time.time() - start_time, 2)
                    
                    # Compute metadata for done event
                    used_tools = "none"
                    if final_state.get("retrieved_docs"):
                        used_tools = ", ".join([r.tool_name for r in final_state.get("retrieved_docs", [])])
                        
                    source_type = final_state.get("source_type", "direct")
                    
                    await stream_queue.put({
                        "type": "done",
                        "full_answer": final_state.get("answer", ""),
                        "tool_used": used_tools,
                        "source_type": source_type,
                        "latency_seconds": latency_seconds,
                        "total_latency_seconds": latency_seconds,
                        "ttft_seconds": final_state.get("ttft_seconds"),
                        "generation_latency_seconds": final_state.get("generation_latency_seconds"),
                    })
                except asyncio.CancelledError:
                    logger.info("Chat graph task cancelled after client disconnect.")
                    raise
                except Exception as e:
                    logger.error(f"Graph execution failed: {e}", exc_info=True)
                    await stream_queue.put({"type": "error", "error": str(e)})
                finally:
                    await stream_queue.put(None) # EOF sentinel

            # Start graph in background
            graph_task = asyncio.create_task(run_graph())

            # Consume the queue and yield SSE events
            try:
                while True:
                    event = await stream_queue.get()
                    if event is None:
                        break

                    event_type = event.get("type", "step")
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            finally:
                if not graph_task.done():
                    graph_task.cancel()
                with suppress(asyncio.CancelledError):
                    await graph_task

        except Exception as e:
            logger.error(f"Error in stream_chat: {e}", exc_info=True)
            err_event = {"type": "error", "error": str(e)}
            yield f"event: error\ndata: {json.dumps(err_event)}\n\n"
