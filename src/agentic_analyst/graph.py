import logging
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.planner import planner
from .agents.researcher import researcher
from .agents.writer import writer
from .agents.critic import critic
from .llm import RUN_METER
from .state import AgentState, Finding

log = logging.getLogger(__name__)

MAX_REVISIONS = 2


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


def finalize_with_caveats(state: AgentState) -> dict[str, Any]:
    """Terminal node for a draft that never passed the critic.

    We don't silently ship a failing draft, and we don't loop forever. Instead
    we append a limitations section built from the critic's unresolved fixes, so
    the reader is told exactly where the draft is weak. No model call — this is
    a deterministic string append, so it adds no cost.
    """
    critique = state["critique"]
    assert critique is not None  # the router only sends us here after a critique

    lines = [
        state["draft"],
        "",
        "---",
        "## Limitations",
        (
            f"This draft did not pass automated review after {state['revision_count']} "
            "revisions. The following issues remain unresolved:"
        ),
        "",
    ]
    lines.extend(f"- {fix}" for fix in critique.required_fixes)
    lines.append("")
    lines.append(f"_Weakest claim flagged by review: {critique.weakest_claim}_")

    log.info(
        "finalizing with caveats after %d revisions (%d unresolved fixes)",
        state["revision_count"],
        len(critique.required_fixes),
    )
    return {"draft": "\n".join(lines)}


def route_after_critic(state: AgentState) -> str:
    """Decide where to go once the critic has produced a verdict.

    Three outcomes, matching the playbook mermaid:
      - pass                      -> END
      - fail, revisions remaining -> writer (revision mode, consumes fixes)
      - fail, revisions exhausted -> finalize_with_caveats
    """
    critique = state["critique"]
    assert critique is not None  # critic always writes one before this runs

    if critique.passed:
        log.info("critic passed the draft; finishing")
        return END

    if state["revision_count"] < MAX_REVISIONS:
        log.info(
            "critic failed (revision %d/%d); sending back to writer",
            state["revision_count"],
            MAX_REVISIONS,
        )
        return "writer"

    log.info(
        "critic failed and revisions exhausted (%d); finalizing with caveats",
        state["revision_count"],
    )
    return "finalize_with_caveats"


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_node)  # the wrapper, not researcher() itself
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)
    workflow.add_node("finalize_with_caveats", finalize_with_caveats)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")

    # Writer always hands off to the critic for judgement.
    workflow.add_edge("writer", "critic")

    # The critic's verdict decides the next hop. The dict maps the router's
    # return values to real node names; END is a valid target too.
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "finalize_with_caveats": "finalize_with_caveats",
            END: END,
        },
    )

    # The caveats path is terminal.
    workflow.add_edge("finalize_with_caveats", END)

    return workflow.compile()