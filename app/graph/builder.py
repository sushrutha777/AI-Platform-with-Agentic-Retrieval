"""LangGraph construction and compilation."""

from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import GraphNodes
from app.tools.base import ToolRegistry


def route_after_router(state: AgentState) -> str:
    """Conditional edge after the router node."""
    intent = state.get("intent", "")
    if intent in ["greeting", "farewell", "casual"]:
        return "generate"
    return "parallel_retrieval"


def build_graph(tool_registry: ToolRegistry):
    """Build and compile the execution graph."""
    nodes = GraphNodes(tool_registry)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("context_resolver", nodes.context_resolver)
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("parallel_retrieval", nodes.parallel_retrieval)
    workflow.add_node("context_aggregator", nodes.context_aggregator)
    workflow.add_node("generate", nodes.generate)
    
    # Add edges
    workflow.set_entry_point("context_resolver")
    workflow.add_edge("context_resolver", "router")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "generate": "generate",
            "parallel_retrieval": "parallel_retrieval"
        }
    )
    
    workflow.add_edge("parallel_retrieval", "context_aggregator")
    workflow.add_edge("context_aggregator", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
