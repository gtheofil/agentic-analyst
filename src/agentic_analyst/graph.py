from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.planner import planner
from .agents.writer import writer  # <-- import the writer node
from .state import AgentState


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("writer", writer)  # <-- add it

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "writer")  # <-- planner -> writer
    workflow.add_edge("writer", END)  # <-- writer -> END

    return workflow.compile()
