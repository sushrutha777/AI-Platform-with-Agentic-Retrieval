"""Focused behavioral checks for the LangGraph migration and streaming runtime."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.context.service import ContextService
from app.agents.router import AgentRouter
from app.graph.nodes import GraphNodes
from app.graph import build_graph
from app.llm.gateway import LLMGateway
from app.services.chat_service import ChatService
from app.tools.base import BaseAgentTool, ToolRegistry, ToolResult
from app.schemas.chat import ChatRequest


class _Tool(BaseAgentTool):
    def __init__(self, name, delay=0.0, error=False):
        self.name = name
        self.description = name
        self.delay = delay
        self.error = error
        self.started = None
        self.finished = None

    async def run(self, query):
        self.started = time.perf_counter()
        await asyncio.sleep(self.delay)
        self.finished = time.perf_counter()
        if self.error:
            raise RuntimeError(f"{self.name} failed")
        return ToolResult(
            tool_name=self.name,
            source_type=self.name,
            output=f"{self.name} output",
            sources=[{"title": self.name, "content": "evidence"}],
        )


def test_independent_tools_execute_concurrently_and_failures_are_isolated():
    async def run():
        good = _Tool("good", delay=0.08)
        failed = _Tool("failed", delay=0.08, error=True)
        registry = ToolRegistry()
        registry.register(good)
        registry.register(failed)
        nodes = GraphNodes(registry)
        result = await nodes.parallel_retrieval(
            {"rewritten_query": "q", "tools_to_run": ["good", "failed"]},
            {"configurable": {}},
        )
        return good, failed, result

    good, failed, result = asyncio.run(run())
    assert good.started is not None and failed.started is not None
    assert abs(good.started - failed.started) < 0.04
    assert [item.tool_name for item in result["retrieved_docs"]] == ["good"]


def test_self_contained_query_skips_rewrite_llm_and_follow_up_uses_it():
    async def run():
        service = ContextService()
        service.add_turn("s", "user", "What is Qdrant?")
        service.add_turn("s", "assistant", "A vector database.")
        with patch("app.llm.gateway.gateway.complete", new=AsyncMock(return_value="rewritten")) as complete:
            standalone = await service.rewrite_query("s", "What is the return policy?")
            follow_up = await service.rewrite_query("s", "How does it work?")
        return standalone, follow_up, complete

    standalone, follow_up, complete = asyncio.run(run())
    assert standalone == "What is the return policy?"
    assert follow_up == "rewritten"
    assert complete.await_count == 1


def test_router_keeps_substantive_query_after_greeting_or_courtesy():
    greeting_query = AgentRouter.route("Hi, what is the return policy?")
    courtesy_query = AgentRouter.route("Thanks, how long does shipping take?")
    greeting_only = AgentRouter.route("Hi there")

    assert greeting_query.intent == "knowledge"
    assert "document_search" in greeting_query.tools
    assert courtesy_query.intent == "knowledge"
    assert "document_search" in courtesy_query.tools
    assert greeting_only.intent == "greeting"


def test_common_casual_questions_work_without_llm_generation():
    async def run(question):
        graph = build_graph(ToolRegistry())
        queue = asyncio.Queue()

        async def unexpected_generation(*args, **kwargs):
            raise AssertionError("known casual response should not call the LLM")
            yield "unreachable"

        with patch("app.graph.nodes.gateway.stream", new=unexpected_generation):
            result = await graph.ainvoke(
                {"original_query": question, "session_id": f"casual-{question}"},
                config={"configurable": {"stream_queue": queue}},
            )
        return result

    how_are_you = asyncio.run(run("how are you"))
    who_are_you = asyncio.run(run("who are you"))
    assert "doing well" in how_are_you["answer"]
    assert "AI assistant" in who_are_you["answer"]


def test_normal_query_has_one_final_generation_call():
    async def fake_stream(messages, **kwargs):
        yield "answer"

    async def run():
        registry = ToolRegistry()
        registry.register(_Tool("document_search"))
        graph = build_graph(registry)
        stream = MagicMock(side_effect=fake_stream)
        with patch("app.graph.nodes.gateway.stream", new=stream):
            await graph.ainvoke(
                {"original_query": "According to the policy, what is shipping?", "session_id": "one-call"},
                config={"configurable": {}},
            )
            return stream

    stream = asyncio.run(run())
    # The patched async generator is invoked once by the final generate node.
    assert stream.call_count == 1


def test_stream_reports_ttft_and_generation_latency_separately():
    async def fake_stream(messages, **kwargs):
        await asyncio.sleep(0.01)
        yield "first"
        await asyncio.sleep(0.01)
        yield "second"

    async def run():
        registry = ToolRegistry()
        registry.register(_Tool("document_search"))
        graph = build_graph(registry)
        queue = asyncio.Queue()
        with patch("app.graph.nodes.gateway.stream", new=fake_stream):
            result = await graph.ainvoke(
                {"original_query": "According to the policy, what is shipping?", "session_id": "metrics"},
                config={"configurable": {"stream_queue": queue}},
            )
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return result, events

    result, events = asyncio.run(run())
    metrics = [event for event in events if event.get("type") == "metrics"]
    assert len(metrics) == 1
    assert result["ttft_seconds"] is not None
    assert result["generation_latency_seconds"] >= result["ttft_seconds"]


def test_sse_tokens_arrive_before_graph_finishes_and_disconnect_cancels_graph():
    async def run():
        release = asyncio.Event()
        cancelled = asyncio.Event()
        service = ChatService.__new__(ChatService)

        async def fake_ainvoke(state, config):
            await config["configurable"]["stream_queue"].put({"type": "token", "token": "early"})
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"answer": "early", "retrieved_docs": [], "source_type": "direct"}

        service.graph = SimpleNamespace(ainvoke=fake_ainvoke)
        stream = service.stream_chat(ChatRequest(question="hello"))
        first = await stream.__anext__()
        assert '"token": "early"' in first
        assert not cancelled.is_set()
        await stream.aclose()
        await asyncio.sleep(0)
        return cancelled

    assert asyncio.run(run()).is_set()


def test_chat_graph_invocation_has_langsmith_trace_context():
    async def run():
        service = ChatService.__new__(ChatService)
        captured = {}

        async def fake_ainvoke(state, config):
            captured.update(config)
            return {"answer": "ok", "retrieved_docs": [], "source_type": "direct"}

        service.graph = SimpleNamespace(ainvoke=fake_ainvoke)
        [event async for event in service.stream_chat(
            ChatRequest(question="hello", conversation_id="trace-test")
        )]
        return captured

    config = asyncio.run(run())
    assert config["run_name"] == "agentic_rag_request"
    assert "langgraph" in config["tags"]
    assert config["metadata"]["conversation_id"] == "trace-test"


def test_queue_gets_terminal_event_when_graph_fails():
    async def run():
        service = ChatService.__new__(ChatService)

        async def failing_ainvoke(state, config):
            raise RuntimeError("generation failed")

        service.graph = SimpleNamespace(ainvoke=failing_ainvoke)
        events = [event async for event in service.stream_chat(ChatRequest(question="hello"))]
        return events

    events = asyncio.run(run())
    assert any("event: error" in event for event in events)


def test_gemini_failure_uses_groq_fallback():
    async def fallback_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="groq answer"))])

    async def run():
        gateway = LLMGateway()
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == gateway.primary_model:
                raise RuntimeError("Gemini unavailable")
            return fallback_stream()

        with patch("app.llm.gateway.litellm.acompletion", new=fake_acompletion):
            tokens = [token async for token in gateway.stream([{"role": "user", "content": "q"}])]
        return calls, tokens

    calls, tokens = asyncio.run(run())
    assert calls[:2] == ["gemini/gemini-3.1-flash-lite", "groq/llama-3.3-70b-versatile"]
    assert tokens == ["groq answer"]
