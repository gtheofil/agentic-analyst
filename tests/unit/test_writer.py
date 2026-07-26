"""Unit tests for the writer node.

The writer's job is to be given evidence and cite it. What can be tested
without a model is what it is *handed*: the findings, numbered and quoted, in
a form the Phase 4 hard-check can match report tags back to.
"""

from typing import Any
from unittest.mock import patch

from agentic_analyst.agents.writer import _format_findings, writer
from agentic_analyst.state import AgentState, Finding, Task
from tests.conftest import FakeLLM

F1 = Finding(
    task_id=1,
    claim="Installations rose 43% in 2024",
    quote="a 43% increase on 2023",
    source_url="https://mcs.example/2024",
    confidence="high",
)
F2 = Finding(
    task_id=2,
    claim="A 100kW system costs $150,000 before incentives",
    quote="$1.50 to $3.00 per watt",
    source_url="https://vendor.example/costs",
    confidence="low",
)


def _state(findings: list[Finding]) -> AgentState:
    return AgentState(
        brief="Assess energy options",
        memory_context="",
        plan=[Task(id=1, goal="Quantify growth"), Task(id=2, goal="Compare costs")],
        findings=findings,
        draft="",
        critique=None,
        revision_count=0,
        cost_usd=0.0,
    )


def _prompt_for(findings: list[Finding]) -> str:
    """Run the writer with the model stubbed, return the prompt it was sent."""
    with patch("agentic_analyst.agents.writer.call", return_value="draft") as model:
        writer(_state(findings))
    user: str = model.call_args.kwargs["user"]
    return user


def test_findings_are_numbered_from_one() -> None:
    """`F<n>` is a label for this report, which is why it is assigned here and
    not stored on the Finding."""
    rendered = _format_findings([F1, F2])

    assert "[F1]" in rendered
    assert "[F2]" in rendered
    assert "[F0]" not in rendered


def test_each_finding_carries_its_evidence() -> None:
    """The writer has to see the quote to judge what it does and does not
    support — a claim alone would invite over-citation."""
    rendered = _format_findings([F2])

    assert F2.claim in rendered
    assert F2.quote in rendered
    assert F2.source_url in rendered
    assert "confidence: low" in rendered


def test_no_findings_is_said_out_loud() -> None:
    """An empty list must not render as blank: the prompt tells the writer to
    report the gap, and it can only do that if it is told there is one."""
    assert "none" in _format_findings([]).lower()


def test_writer_is_handed_the_findings(stub_llm: None) -> None:
    prompt = _prompt_for([F1, F2])

    assert "Findings:" in prompt
    assert "[F1]" in prompt and "[F2]" in prompt
    assert F1.quote in prompt


def test_writer_still_runs_with_no_findings(stub_llm: None) -> None:
    """A run where every task came back empty must still produce a report."""
    result = writer(_state([]))

    assert result["draft"]


def test_writer_prompt_bans_the_old_unverified_tag() -> None:
    """Phase 1's `[unverified]` was a placeholder for having no findings.

    Leaving it available would give the model a legal way to write an
    unsupported claim, which is exactly what citing findings is meant to stop.
    """
    from agentic_analyst.agents.writer import _PROMPT_PATH

    text = _PROMPT_PATH.read_text(encoding="utf-8")

    assert "[F3]" in text, "the prompt should show the citation format by example"
    assert "no `[unverified]`" in text


def test_writer_reports_its_own_cost(stub_llm_with_cost: FakeLLM) -> None:
    result: dict[str, Any] = writer(_state([F1]))

    assert result["cost_usd"] == stub_llm_with_cost.charge
