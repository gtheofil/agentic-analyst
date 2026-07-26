import logging
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.planner import planner
from .agents.researcher import researcher
from .agents.writer import writer
from .llm import RUN_METER
from .state import AgentState, Finding

log = logging.getLogger(__name__)


def researcher_node(state: AgentState) -> dict[str, Any]:
    """Run the per-task researcher over every task in the plan, sequentially.

    Sequential on purpose: tasks are independent, so this fans out cleanly
    later, but the cost meter is process-global and not concurrency-safe yet.
    That trade is recorded in the README roadmap rather than half-built here.
    """
    all_findings: list[Finding] = []
    with RUN_METER.track() as spend:
        for task in state["plan"]:
            all_findings.extend(researcher(task))

    log.info(
        "researcher produced %d findings across %d tasks (cost $%.6f)",
        len(all_findings),
        len(state["plan"]),
        spend.usd,
    )
    return {"findings": all_findings, "cost_usd": spend.usd}


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_node)  # the wrapper, not researcher() itself
    workflow.add_node("writer", writer)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()
