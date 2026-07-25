"""End-to-end run of the compiled graph with the LLM stubbed out."""

import re

from agentic_analyst.graph import build_graph
from agentic_analyst.state import AgentState


def test_planner_to_writer_pipeline(stub_llm: None, seed_state: AgentState) -> None:
    result = build_graph().invoke(seed_state)

    assert result["plan"], "planner wrote no plan into state"
    assert result["draft"], "writer wrote no draft into state"


def test_citation_format_matches_hardcheck(stub_llm: None, seed_state: AgentState) -> None:
    """The Phase 4 hard-check will regex for these tags — lock the format now."""
    result = build_graph().invoke(seed_state)

    tags = re.findall(r"\[(?:F\d+|unverified)\]", result["draft"])
    assert tags, "writer produced no citation tags"


def test_cost_is_summed_not_overwritten(stub_llm_with_cost: float, seed_state: AgentState) -> None:
    """Both nodes report a cost, so the total must be their sum.

    Without the `Annotated[float, operator.add]` reducer on AgentState, the
    writer's cost would overwrite the planner's and this would read 0.01.
    """
    result = build_graph().invoke(seed_state)

    assert result["cost_usd"] == 2 * stub_llm_with_cost
