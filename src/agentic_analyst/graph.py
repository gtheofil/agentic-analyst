"""Graph assembly: which node runs, and in what order.

Phase 1 is deliberately linear (planner → writer). The researcher, critic and
revision loop slot in here in later phases without any node changing.
"""

from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.planner import planner
from .agents.writer import writer
from .state import AgentState


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("writer", writer)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()
