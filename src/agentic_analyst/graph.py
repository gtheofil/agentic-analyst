from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.planner import planner
from .agents.researcher import researcher
from .agents.writer import writer
from .state import AgentState, Finding


def researcher_node(state: AgentState) -> dict[str, Any]:
    """Run the per-task researcher over every task in the plan, sequentially."""
    all_findings: list[Finding] = []
    for task in state["plan"]:
        all_findings.extend(researcher(task))
    return {"findings": all_findings}


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_node)   # the wrapper, not researcher() itself
    workflow.add_node("writer", writer)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()