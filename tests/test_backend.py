"""Unit and integration tests for FastAPI backend, Qdrant/BM25 Hybrid retrieval, and Reranker."""

import unittest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from app.main import app
from app.retriever.sparse import SparseBM25Retriever
from app.retriever.hybrid import HybridRetriever
from app.retriever.base import BaseRetriever
from app.reranker.flashrank_reranker import FlashRankReranker
from typing import List


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_checks(self):
        """Verify health and probe endpoints return status 200."""
        resp_health = self.client.get("/api/v1/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertEqual(resp_health.json()["status"], "healthy")

        resp_live = self.client.get("/api/v1/health/live")
        self.assertEqual(resp_live.status_code, 200)
        self.assertEqual(resp_live.json()["status"], "alive")

        resp_ready = self.client.get("/api/v1/health/ready")
        self.assertEqual(resp_ready.status_code, 200)

        resp_metrics = self.client.get("/api/v1/health/metrics")
        self.assertEqual(resp_metrics.status_code, 200)
        self.assertIn("memory_rss_mb", resp_metrics.json())

    def test_agent_router_intent_classification(self):
        """Test fast zero-LLM heuristic router across diverse query types."""
        from app.agents.router import AgentRouter

        # 1. Greetings
        dec = AgentRouter.route("Hello there!")
        self.assertEqual(dec.intent, "greeting")
        self.assertEqual(len(dec.tools), 0)

        # 2. Casual
        dec = AgentRouter.route("Thank you for your help")
        self.assertEqual(dec.intent, "casual")
        self.assertEqual(len(dec.tools), 0)

        # 3. Wikipedia / Encyclopedic
        dec = AgentRouter.route("Who is Albert Einstein?")
        self.assertEqual(dec.intent, "knowledge")
        self.assertIn("wikipedia", dec.tools)

        # 4. Web search for real-time / temporal
        dec = AgentRouter.route("What is the latest news in 2026?")
        self.assertEqual(dec.intent, "knowledge")
        self.assertIn("web_search", dec.tools)

        # 5. Document search
        dec = AgentRouter.route("According to the uploaded policy document, what is the refund rule?")
        self.assertEqual(dec.intent, "knowledge")
        self.assertIn("document_search", dec.tools)

    def test_context_service_and_query_rewriting(self):
        """Test in-memory session tracking and pronoun resolution."""
        from app.context.service import ContextService
        cs = ContextService()
        session_id = "test_sess_001"

        cs.add_turn(session_id, "user", "What is quantum computing?")
        cs.add_turn(session_id, "assistant", "Quantum computing leverages qubits and superposition.")

        # Test pronoun rewriting
        import asyncio
        rewritten = asyncio.run(cs.rewrite_query(session_id, "How does it work?"))
        self.assertIsInstance(rewritten, str)

        # Test context window
        window = cs.get_context_window(session_id)
        self.assertEqual(len(window), 2)
        self.assertEqual(window[0]["role"], "user")

    def test_sparse_bm25_retriever(self):
        """Test BM25 keyword indexing and retrieval."""
        docs = [
            Document(page_content="LangGraph is a library for building stateful multi-actor applications with LLMs."),
            Document(page_content="Qdrant is a vector similarity search engine and database."),
            Document(page_content="FastAPI is a modern web framework for building APIs with Python."),
        ]
        bm25 = SparseBM25Retriever(docs)
        results = bm25.retrieve("vector search Qdrant", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Qdrant", results[0].page_content)

    def test_hybrid_rrf_fusion(self):
        """Test Reciprocal Rank Fusion of dense and sparse candidates."""
        class MockDenseRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
                return [
                    Document(page_content="Doc A: High semantic match"),
                    Document(page_content="Doc B: Moderate match"),
                ]

        sparse_docs = [
            Document(page_content="Doc B: Moderate match with keyword"),
            Document(page_content="Doc C: Keyword only match"),
        ]
        sparse_retriever = SparseBM25Retriever(sparse_docs)
        hybrid = HybridRetriever(MockDenseRetriever(), sparse_retriever)

        fused = hybrid.retrieve("keyword", top_k=5)
        self.assertGreater(len(fused), 0)
        contents = [d.page_content for d in fused]
        self.assertTrue(any("Doc B" in c for c in contents))

    def test_flashrank_reranker(self):
        """Test FlashRank neural reranking."""
        reranker = FlashRankReranker()
        docs = [
            Document(page_content="Cats are domestic animals that like to nap in the sun."),
            Document(page_content="Qdrant enables low-latency vector similarity search in production."),
            Document(page_content="Python is a popular programming language for AI."),
        ]
        reranked = reranker.rerank(query="vector search database", documents=docs, top_k=2)
        self.assertLessEqual(len(reranked), 2)
        self.assertIn("Qdrant", reranked[0].page_content)


if __name__ == "__main__":
    unittest.main()
