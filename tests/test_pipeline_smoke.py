# tests/test_pipeline_smoke.py
import re
from unittest.mock import patch

from agentic_analyst.graph import build_graph
from agentic_analyst.state import AgentState, Task


def fake_call(tier, system, user, schema=None):
    """Stand-in for llm.call — no network, no API key."""
    if schema is not None:
        return schema(
            tasks=[
                Task(id=1, goal="Assess baseline energy use"),
                Task(id=2, goal="Identify efficiency options", depends_on=[1]),
            ]
        )
    return "Energy use is high.[unverified] Savings are possible.[unverified]"


def _seed_state() -> AgentState:
    return AgentState(
        brief="test brief",
        memory_context="",
        plan=[],
        findings=[],
        draft="",
        critique=None,
        revision_count=0,
        cost_usd=0.0,
    )


def test_planner_to_writer_pipeline():
    with (
        patch("agentic_analyst.agents.planner.call", side_effect=fake_call),
        patch("agentic_analyst.agents.writer.call", side_effect=fake_call),
    ):
        result = build_graph().invoke(_seed_state())

    # print("planner.call fired:", p_plan.called)
    # print("writer.call fired:", p_write.called)
    # print("draft repr:", repr(result["draft"]))
    assert result["draft"]


def test_citation_format_matches_hardcheck():
    with (
        patch("agentic_analyst.agents.planner.call", side_effect=fake_call),
        patch("agentic_analyst.agents.writer.call", side_effect=fake_call),
    ):
        result = build_graph().invoke(_seed_state())

    tags = re.findall(r"\[(?:F\d+|unverified)\]", result["draft"])
    assert tags, "writer produced no citation tags"
