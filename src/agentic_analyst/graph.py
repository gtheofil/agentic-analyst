import logging
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.critic import critic
from .agents.planner import planner
from .agents.researcher import researcher
from .agents.writer import writer
from .llm import RUN_METER
from .memory.episodic import load_memory, write_memory
from .state import AgentState, Critique, Finding

log = logging.getLogger(__name__)

# Revisions actually performed by the writer, not critiques issued. Two means
# the writer gets at most three attempts: the original plus two rewrites.
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

    revisions = state["revision_count"]
    scores = critique.scores

    # Deliberately NOT "## Limitations": the writer's own prompt already ends
    # every report with a section by that name. A second one would read as a
    # continuation of the writer's own caveats rather than as a machine-appended
    # review failure, which is exactly the distinction the reader needs.
    lines = [
        state["draft"],
        "",
        "---",
        "## Unresolved review issues",
        (
            f"This report did not pass automated review after {revisions} "
            f"revision{'' if revisions == 1 else 's'}. It scored "
            f"groundedness {scores.groundedness}/10, structure {scores.structure}/10, "
            f"actionability {scores.actionability}/10 (mean {scores.mean:.1f}; "
            f"{Critique.PASS_MEAN:.0f} required)."
        ),
        "",
    ]

    if critique.required_fixes:
        lines.append("The following issues remain unresolved:")
        lines.append("")
        # Critical first — a reader skimming this section should hit the
        # blocking problems before the cosmetic ones.
        order = {"critical": 0, "major": 1, "minor": 2}
        for fix in sorted(critique.required_fixes, key=lambda f: order[f.severity]):
            lines.append(f"- **{fix.severity}** — {fix.location}: {fix.issue}")

    lines.append("")
    lines.append(f"_Weakest claim flagged by review: {critique.weakest_claim}_")

    log.info(
        "finalizing with caveats after %d revisions (%d unresolved fixes, %d critical)",
        revisions,
        len(critique.required_fixes),
        len(critique.critical_fixes),
    )
    return {"draft": "\n".join(lines)}


def route_after_critic(state: AgentState) -> str:
    """Decide where to go once the critic has produced a verdict.

    Three outcomes, matching the playbook mermaid:
      - pass                      -> write_memory (then END)
      - fail, revisions remaining -> writer (revision mode, consumes fixes)
      - fail, revisions exhausted -> finalize_with_caveats

    `critique.passed` is computed from the scores and fix severities, never read
    off the model's own response — the gate would be worthless if the thing
    being gated got to open it.

    Termination is guaranteed by `revision_count`, which only the writer
    increments and only when it revises: every trip round this loop passes
    through the writer, so the counter cannot stall.
    """
    critique = state["critique"]
    assert critique is not None  # critic always writes one before this runs

    if critique.passed:
        log.info("critic passed the draft (mean %.1f); finishing", critique.scores.mean)
        return "write_memory"

    reason = (
        f"{len(critique.critical_fixes)} critical fix(es)"
        if critique.critical_fixes
        else f"mean {critique.scores.mean:.1f} < {Critique.PASS_MEAN:.0f}"
    )

    if state["revision_count"] < MAX_REVISIONS:
        log.info(
            "critic failed (%s); revision %d/%d, sending back to writer",
            reason,
            state["revision_count"] + 1,
            MAX_REVISIONS,
        )
        return "writer"

    log.info(
        "critic failed (%s) and revisions exhausted (%d); finalizing with caveats",
        reason,
        state["revision_count"],
    )
    return "finalize_with_caveats"


def build_graph() -> Runnable[AgentState, dict[str, Any]]:
    """Assemble the agent graph and return a compiled runnable."""
    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", load_memory)
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_node)  # the wrapper, not researcher() itself
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)
    workflow.add_node("finalize_with_caveats", finalize_with_caveats)
    workflow.add_node("write_memory", write_memory)

    # Memory is read before planning, so the plan itself benefits from what
    # earlier runs learned — not just the prose at the end.
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")

    # Writer always hands off to the critic for judgement.
    workflow.add_edge("writer", "critic")

    # The critic's verdict decides the next hop. The dict maps the router's
    # return values to real node names.
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "finalize_with_caveats": "finalize_with_caveats",
            "write_memory": "write_memory",
        },
    )

    # Both terminal paths write memory before ending. A run that failed review
    # is the one most worth remembering — "this kind of brief goes badly" is a
    # lesson, and routing only the successes to memory would leave the agent
    # remembering an unrepresentatively easy history.
    workflow.add_edge("finalize_with_caveats", "write_memory")
    workflow.add_edge("write_memory", END)

    return workflow.compile()
