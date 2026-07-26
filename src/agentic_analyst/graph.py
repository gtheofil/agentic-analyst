import logging
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from .agents.critic import critic
from .agents.planner import planner
from .agents.researcher import researcher
from .agents.writer import writer
from .llm import RUN_METER, over_budget
from .memory.episodic import load_memory, write_memory
from .settings import SETTINGS
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


def finalize_over_budget(state: AgentState) -> dict[str, Any]:
    """Terminal node for a run that hit its cost cap mid-flight.

    Makes no model call — it cannot, that is the whole point — so it works with
    whatever the run had already produced. Two shapes, depending on how far it
    got: an unreviewed draft, or (if the researcher exhausted the budget before
    the writer ever ran) the raw findings, which are still evidence a human can
    read even though nobody wrote them up.

    The alternative, letting `BudgetExceeded` propagate out of `invoke`, would
    throw away the state along with the money already spent on it.
    """
    draft = state["draft"]
    findings = state["findings"]
    spent = state["cost_usd"]

    header = [
        "---",
        "## Incomplete — run budget exhausted",
        (
            f"This run stopped early after spending ${spent:.4f} of its "
            f"${SETTINGS.max_run_cost_usd:.2f} cap. What follows was produced "
            "before the cap was reached and has not been through review."
        ),
        "",
    ]

    if draft:
        log.warning("over budget after the draft; shipping it unreviewed")
        return {"draft": "\n".join([draft, "", *header])}

    log.warning("over budget before a draft existed; shipping %d raw findings", len(findings))
    lines = [
        "# Report unavailable",
        "",
        *header,
        (
            f"No report was written. The {len(findings)} finding(s) gathered before the "
            "budget ran out are listed below, unsynthesised."
            if findings
            else "No report was written and no evidence was gathered."
        ),
        "",
    ]
    for i, finding in enumerate(findings, 1):
        lines.extend(
            [
                f"**[F{i}]** ({finding.confidence} confidence) {finding.claim}",
                f"> {finding.quote}",
                f"— {finding.source_url}",
                "",
            ]
        )
    return {"draft": "\n".join(lines)}


def route_on_budget(next_node: str) -> Callable[[AgentState], str]:
    """Build a router that goes to `next_node` unless the run is out of money.

    A factory because the same question is asked at two different edges, and
    LangGraph identifies a conditional edge by its function. Asked *between*
    nodes, on the cheap side of the expensive ones: the writer and the critic
    each cost about a third of a run, so the useful moment to check is before
    paying for one, not after.
    """

    def router(state: AgentState) -> str:
        if over_budget():
            log.warning(
                "budget cap $%.2f reached ($%.6f spent); skipping %s",
                SETTINGS.max_run_cost_usd,
                state["cost_usd"],
                next_node,
            )
            return "finalize_over_budget"
        return next_node

    return router


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
    workflow.add_node("finalize_over_budget", finalize_over_budget)
    workflow.add_node("write_memory", write_memory)

    # Memory is read before planning, so the plan itself benefits from what
    # earlier runs learned — not just the prose at the end.
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "planner")
    workflow.add_edge("planner", "researcher")

    # The two expensive hand-offs are guarded by the cost cap. Both are the
    # same question asked before paying for a strong-tier call over the whole
    # draft; the researcher is not guarded here because it is guarded call by
    # call inside its own loop, by the same cap in `llm._check_budget`.
    workflow.add_conditional_edges(
        "researcher",
        route_on_budget("writer"),
        {"writer": "writer", "finalize_over_budget": "finalize_over_budget"},
    )
    workflow.add_conditional_edges(
        "writer",
        route_on_budget("critic"),
        {"critic": "critic", "finalize_over_budget": "finalize_over_budget"},
    )

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

    # Both *completed* paths write memory before ending. A run that failed
    # review is the one most worth remembering — "this kind of brief goes
    # badly" is a lesson, and routing only the successes to memory would leave
    # the agent remembering an unrepresentatively easy history.
    workflow.add_edge("finalize_with_caveats", "write_memory")
    workflow.add_edge("write_memory", END)

    # The over-budget path is the exception, and goes straight to END. Writing
    # memory costs a model call, which is precisely what this run has run out
    # of permission to make — `_check_budget` would refuse it and the graceful
    # exit would end in the exception it exists to avoid.
    workflow.add_edge("finalize_over_budget", END)

    return workflow.compile()
