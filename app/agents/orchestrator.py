"""Agent Orchestrator executing the streamlined one-LLM-call pipeline."""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any
from app.agents.router import AgentRouter
from app.context.service import context_service
from app.prompts.templates import DIRECT_RESPONSE_PROMPT, SYNTHESIS_PROMPT
from app.llm.gateway import gateway
from app.core.logging import logger
from app.tools.base import ToolResult

class AgentOrchestrator:
    """Manages the end-to-end execution of a chat request."""

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        self.router = AgentRouter()

    async def stream_chat(self, question: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the execution pipeline."""
        start_time = time.time()
        
        # 1. Rewrite Query (using ContextService heuristics)
        yield {"type": "step", "label": "Analyzing context..."}
        rewritten_query = await context_service.rewrite_query(session_id, question)
        
        # 2. Route
        yield {"type": "step", "label": "Routing intent..."}
        decision = self.router.route(rewritten_query)
        
        # 3. Parallel Retrieval
        context_text = ""
        used_tools = "none"
        sources = []
        valid_results = []
        
        if decision.tools:
            yield {"type": "step", "label": f"Searching ({', '.join(decision.tools)})..."}
            
            # Execute tools in parallel
            tasks = []
            for tool_name in decision.tools:
                tool = self.tool_registry.get(tool_name)
                if tool:
                    tasks.append(tool.run(rewritten_query))
                    
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for res in results:
                if isinstance(res, BaseException):
                    logger.error(f"Tool execution failed: {res}")
                elif isinstance(res, ToolResult) and res.sources:  # Only keep results that actually found something
                    valid_results.append(res)
                    
            if valid_results:
                used_tools = ", ".join([r.tool_name for r in valid_results])
                
                for r in valid_results:
                    sources.extend(r.sources)
                    context_text += f"\n--- From {r.tool_name} ---\n{r.output}\n"
        source_type = "hybrid" if len(valid_results) > 1 else (valid_results[0].source_type if valid_results else "direct")
        
        # Emit metadata early
        yield {
            "type": "metadata",
            "intent": decision.intent,
            "tool_used": used_tools,
            "source_type": source_type,
            "sources": sources,
            "rewritten_query": rewritten_query
        }

        # Fast-path for greetings and farewells (0ms latency, zero LLM overhead)
        if decision.intent == "greeting":
            fast_replies = "Hello! How can I help you today?"
            full_answer = fast_replies
            for word in fast_replies.split(" "):
                yield {"type": "token", "token": word + " "}
                await asyncio.sleep(0.02)
        elif decision.intent == "farewell":
            fast_replies = "Goodbye! Feel free to reach out if you have any more questions."
            full_answer = fast_replies
            for word in fast_replies.split(" "):
                yield {"type": "token", "token": word + " "}
                await asyncio.sleep(0.02)
        else:
            # 4. Synthesize with LLM
            yield {"type": "step", "label": "Synthesizing answer..."}
            
            history = context_service.format_history_text(session_id)
            
            if decision.intent == "casual" or not decision.tools:
                prompt = DIRECT_RESPONSE_PROMPT.format(history=history, question=rewritten_query)
            else:
                prompt = SYNTHESIS_PROMPT.format(context=context_text, history=history, question=rewritten_query)
                
            messages = [{"role": "user", "content": prompt}]
            
            # Stream LLM tokens
            full_answer = ""
            async for token in gateway.stream(messages):
                full_answer += token
                yield {"type": "token", "token": token}
            
        # Update context
        context_service.add_turn(session_id, "user", question)
        context_service.add_turn(session_id, "assistant", full_answer)
        
        latency_seconds = round(time.time() - start_time, 2)
        
        # Done
        yield {
            "type": "done",
            "full_answer": full_answer,
            "tool_used": used_tools,
            "source_type": source_type,
            "latency_seconds": latency_seconds,
        }
