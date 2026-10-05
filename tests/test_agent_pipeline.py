"""Automated Unit and Integration Tests for LangGraph Pipeline and Evaluation."""

import pytest
import asyncio
from langchain_core.documents import Document
from app.graph import build_graph
from app.agents.router import AgentRouter
from app.tools.base import ToolRegistry
from app.tools.retriever_tool import DocumentRetrieverTool
from app.retriever.sparse import SparseBM25Retriever
from app.retriever.hybrid import HybridRetriever
from app.retriever.base import BaseRetriever
from app.reranker.null_reranker import NullReranker
from eval.evaluate_ragas import compute_empirical_metrics, EVAL_DATASET


class DummyDenseRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5):
        return [
            Document(page_content="Electronics can be returned within 30 days of purchase with original receipt."),
            Document(page_content="Standard shipping takes 3-5 business days across the country."),
        ]


@pytest.fixture
def compiled_graph():
    sparse_docs = [
        Document(page_content="Electronics can be returned within 30 days of purchase with original receipt."),
        Document(page_content="Standard shipping takes 3-5 business days across the country."),
    ]
    sparse = SparseBM25Retriever(sparse_docs)
    dense = DummyDenseRetriever()
    hybrid = HybridRetriever(dense, sparse)
    reranker = NullReranker()
    
    registry = ToolRegistry()
    retriever_tool = DocumentRetrieverTool(hybrid, reranker)
    registry.register(retriever_tool)
    return build_graph(registry)


def test_golden_dataset_structure():
    """Verify that the evaluation benchmark contains comprehensive, valid samples."""
    assert len(EVAL_DATASET) >= 20, f"Expected at least 20 evaluation samples, got {len(EVAL_DATASET)}"
    for item in EVAL_DATASET:
        assert "question" in item and len(item["question"]) > 5
        assert "ground_truth" in item and len(item["ground_truth"]) > 5
        assert "id" in item
        assert "category" in item


def test_empirical_metrics_calculation():
    """Verify that the mathematical benchmark metric calculations return valid bounded numbers."""
    mock_samples = [
        {
            "id": "test_1",
            "category": "returns",
            "user_input": "What is the return policy?",
            "reference": "Returns are accepted within 30 days.",
            "response": "You can return items within 30 days of purchase.",
            "retrieved_contexts": ["Returns are accepted within 30 days in original condition."],
            "latency_total_sec": 1.2,
            "latency_retrieval_sec": 0.3,
            "latency_generation_sec": 0.9,
        },
        {
            "id": "test_2",
            "category": "shipping",
            "user_input": "How long is shipping?",
            "reference": "Shipping takes 3 to 5 business days.",
            "response": "Standard shipping takes 3-5 business days.",
            "retrieved_contexts": ["Shipping delivery takes 3 to 5 business days."],
            "latency_total_sec": 1.5,
            "latency_retrieval_sec": 0.4,
            "latency_generation_sec": 1.1,
        },
    ]

    metrics = compute_empirical_metrics(mock_samples)
    assert "faithfulness" in metrics
    assert "context_recall" in metrics
    assert "context_precision" in metrics
    assert "answer_relevancy" in metrics
    assert "average_total_latency_sec" in metrics

    # Assert bounded between 0.0 and 1.0
    assert 0.0 <= metrics["faithfulness"] <= 1.0
    assert 0.0 <= metrics["context_recall"] <= 1.0
    assert 0.0 <= metrics["context_precision"] <= 1.0
    assert 0.0 <= metrics["answer_relevancy"] <= 1.0
    assert metrics["sample_count"] == 2
    assert metrics["evaluation_mode"] == "empirical_lexical_benchmark"


def test_agent_routing_categories():
    """Verify intent router correctly routes diverse user intents."""
    # Knowledge / Document
    r1 = AgentRouter.route("What is the company return policy?")
    assert r1.intent == "knowledge"

    # Conversational
    r2 = AgentRouter.route("Good morning, how are you?")
    assert r2.intent in ["greeting", "casual"]
    assert len(r2.tools) == 0

    # Explicit web search
    r3 = AgentRouter.route("What is the stock price of Apple today in 2026?")
    assert r3.intent == "knowledge"
    assert "web_search" in r3.tools


def test_graph_retrieval_flow(compiled_graph):
    """Verify LangGraph retrieves documents and streams tokens."""
    async def _run():
        queue = asyncio.Queue()
        events = []

        async def invoke():
            result = await compiled_graph.ainvoke(
                {"original_query": "What is the return policy?", "session_id": "pytest_sess_1"},
                config={"configurable": {"stream_queue": queue}}
            )
            await queue.put(None)
            return result

        task = asyncio.create_task(invoke())
        while True:
            event = await queue.get()
            if event is None:
                break
            events.append(event)
        await task

        event_types = [e["type"] for e in events]
        assert "step" in event_types or "token" in event_types or "metadata" in event_types

    asyncio.run(_run())
