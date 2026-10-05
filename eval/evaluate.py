import os
import sys
import asyncio
import time
from typing import List, Dict

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.graph import build_graph
from app.retriever import get_default_dense_retriever, SparseBM25Retriever, HybridRetriever
from app.reranker.null_reranker import NullReranker
from app.tools.base import ToolRegistry
from app.tools.retriever_tool import DocumentRetrieverTool

# Example dataset
TEST_CASES = [
    {"query": "Hello there", "expected_intent": "greeting"},
    {"query": "What is the capital of France?", "expected_intent": "knowledge"},
]

async def run_eval():
    print("Starting evaluation...")
    
    dense = get_default_dense_retriever()
    sparse = SparseBM25Retriever()
    hybrid = HybridRetriever(dense, sparse)
    reranker = NullReranker()
    
    registry = ToolRegistry()
    registry.register(DocumentRetrieverTool(hybrid, reranker))
    
    graph = build_graph(registry)
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"\n--- Test Case {idx+1} ---")
        print(f"Query: {tc['query']}")
        
        start_time = time.time()
        
        final_answer = ""
        intent = "unknown"
        
        queue = asyncio.Queue()
        
        async def invoke():
            result = await graph.ainvoke(
                {"original_query": tc["query"], "session_id": f"eval_sess_{idx}"},
                config={"configurable": {"stream_queue": queue}}
            )
            await queue.put(None)
            return result
        
        task = asyncio.create_task(invoke())
        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "metadata":
                intent = event.get("intent", "unknown")
            elif event.get("type") == "token":
                final_answer += event.get("token", "")
        await task
                
        latency = round(time.time() - start_time, 2)
        
        print(f"Intent detected: {intent} (Expected: {tc['expected_intent']})")
        print(f"Latency: {latency}s")
        print(f"Answer snippet: {final_answer[:100]}...")
        
if __name__ == "__main__":
    asyncio.run(run_eval())
