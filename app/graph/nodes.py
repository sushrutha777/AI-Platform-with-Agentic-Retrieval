"""LangGraph node definitions."""

import asyncio
import time
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig

from app.graph.state import AgentState
from app.agents.router import AgentRouter
from app.context.service import context_service
from app.prompts.templates import DIRECT_RESPONSE_PROMPT, SYNTHESIS_PROMPT
from app.llm.gateway import gateway
from app.core.logging import logger
from app.tools.base import ToolRegistry, ToolResult


class GraphNodes:
    """Methods acting as nodes for the LangGraph."""

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.router = AgentRouter()

    async def context_resolver(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Node: Resolves context and rewrites query if needed."""
        queue = config.get("configurable", {}).get("stream_queue")
        if queue:
            await queue.put({"type": "step", "label": "Analyzing context..."})
            
        rewritten_query = await context_service.rewrite_query(state["session_id"], state["original_query"])
        return {"rewritten_query": rewritten_query}

    async def router_node(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Node: Routes the query based on heuristics."""
        queue = config.get("configurable", {}).get("stream_queue")
        if queue:
            await queue.put({"type": "step", "label": "Routing intent..."})
            
        decision = self.router.route(state["rewritten_query"])
        return {
            "intent": decision.intent,
            "tools_to_run": decision.tools,
            "source_type": "direct" if not decision.tools else "pending"
        }

    async def parallel_retrieval(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Node: Executes selected tools in parallel."""
        queue = config.get("configurable", {}).get("stream_queue")
        tools_to_run = state.get("tools_to_run", [])
        
        if queue and tools_to_run:
            await queue.put({"type": "step", "label": f"Searching ({', '.join(tools_to_run)})..."})

        tasks = []
        for tool_name in tools_to_run:
            tool = self.tool_registry.get(tool_name)
            if tool:
                tasks.append(tool.run(state["rewritten_query"]))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for res in results:
            if isinstance(res, BaseException):
                logger.error(f"Tool execution failed: {res}")
            elif isinstance(res, ToolResult) and res.sources:
                valid_results.append(res)
                
        return {"retrieved_docs": valid_results}

    async def context_aggregator(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Node: Aggregates tool results into context."""
        valid_results = state.get("retrieved_docs", [])
        
        context_text = ""
        sources = []
        
        if valid_results:
            for r in valid_results:
                sources.extend(r.sources)
                context_text += f"\n--- From {r.tool_name} ---\n{r.output}\n"
                
        source_type = "hybrid" if len(valid_results) > 1 else (valid_results[0].source_type if valid_results else "direct")
        
        return {
            "context_text": context_text,
            "sources": sources,
            "source_type": source_type
        }

    async def generate(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Node: Synthesizes final answer and streams tokens."""
        queue = config.get("configurable", {}).get("stream_queue")
        
        intent = state.get("intent", "knowledge")
        rewritten_query = state.get("rewritten_query", "")
        tools_to_run = state.get("tools_to_run", [])
        context_text = state.get("context_text", "")
        session_id = state.get("session_id", "")
        
        # Emit metadata early
        if queue:
            used_tools = ", ".join([r.tool_name for r in state.get("retrieved_docs", [])]) if state.get("retrieved_docs") else "none"
            await queue.put({
                "type": "metadata",
                "intent": intent,
                "tool_used": used_tools,
                "source_type": state.get("source_type", "direct"),
                "sources": state.get("sources", []),
                "rewritten_query": rewritten_query
            })

        # Fast-path for greetings and farewells (0ms latency, zero LLM overhead)
        if intent == "greeting":
            full_answer = "Hello! How can I help you today?"
            if queue:
                for word in full_answer.split(" "):
                    await queue.put({"type": "token", "token": word + " "})
            return {"answer": full_answer, "ttft_seconds": 0.0, "generation_latency_seconds": 0.0}
            
        elif intent == "farewell":
            full_answer = "Goodbye! Feel free to reach out if you have any more questions."
            if queue:
                for word in full_answer.split(" "):
                    await queue.put({"type": "token", "token": word + " "})
            return {"answer": full_answer, "ttft_seconds": 0.0, "generation_latency_seconds": 0.0}

        else:
            if queue:
                await queue.put({"type": "step", "label": "Synthesizing answer..."})
                
            history = context_service.format_history_text(session_id)
            
            if intent == "casual" or not tools_to_run:
                prompt = DIRECT_RESPONSE_PROMPT.format(history=history, question=rewritten_query)
            else:
                prompt = SYNTHESIS_PROMPT.format(context=context_text, history=history, question=rewritten_query)
                
            messages = [{"role": "user", "content": prompt}]
            
            full_answer = ""
            llm_start_time = time.time()
            first_token_time = None
            
            async for token in gateway.stream(messages):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = round(first_token_time - llm_start_time, 3)
                    logger.info("LLM_STREAMING ttft_seconds=%.3f", ttft)
                    if queue:
                        await queue.put({"type": "metrics", "ttft_seconds": ttft})
                full_answer += token
                if queue:
                    await queue.put({"type": "token", "token": token})

            llm_total_time = round(time.time() - llm_start_time, 3)
            logger.info("LLM_STREAMING total_generation_seconds=%.3f", llm_total_time)
            
            # Update context
            context_service.add_turn(session_id, "user", state["original_query"])
            context_service.add_turn(session_id, "assistant", full_answer)

            return {
                "answer": full_answer,
                "ttft_seconds": round(first_token_time - llm_start_time, 3) if first_token_time else None,
                "generation_latency_seconds": llm_total_time,
            }
