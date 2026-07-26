"""End-to-end run of the compiled graph with the LLM stubbed out."""

import re

import pytest

import agentic_analyst.llm as llm
from agentic_analyst.graph import build_graph
from agentic_analyst.state import AgentState


def test_planner_to_researcher_to_writer_pipeline(stub_llm: None, seed_state: AgentState) -> None:
    result = build_graph().invoke(seed_state)

    assert result["plan"], "planner wrote no plan into state"
    assert result["findings"], "researcher wrote no findings into state"
    assert result["draft"], "writer wrote no draft into state"


def test_findings_are_attributed_to_every_task(stub_llm: None, seed_state: AgentState) -> None:
    """The node runs one researcher per task and concatenates the results.

    Without the `extend`, only the last task's findings would survive — the
    same class of bug as the cost reducer below, one layer down.
    """
    result = build_graph().invoke(seed_state)

    task_ids = {task.id for task in result["plan"]}
    covered = {finding.task_id for finding in result["findings"]}
    assert covered == task_ids


def test_citation_format_matches_hardcheck(stub_llm: None, seed_state: AgentState) -> None:
    """The Phase 4 hard-check will regex for these tags — lock the format now."""
    result = build_graph().invoke(seed_state)

    tags = re.findall(r"\[(?:F\d+|unverified)\]", result["draft"])
    assert tags, "writer produced no citation tags"


def test_cost_is_summed_not_overwritten(stub_llm_with_cost: float, seed_state: AgentState) -> None:
    """Every node reports a cost, so the total must be their sum.

    Without the `Annotated[float, operator.add]` reducer on AgentState, each
    node's cost would overwrite the previous one and this would read a single
    call's charge.
    """
    result = build_graph().invoke(seed_state)

    assert llm.RUN_METER.calls > 3, "expected planner + a researcher loop + writer"
    assert result["cost_usd"] == pytest.approx(llm.RUN_METER.calls * stub_llm_with_cost)
