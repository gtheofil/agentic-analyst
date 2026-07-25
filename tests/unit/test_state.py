"""Unit tests for the typed contracts in state.py."""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_analyst.state import AgentState, Critique, Finding, Task


# ---------- Finding ----------
def test_finding_rejects_missing_source_url() -> None:
    incorrect: dict[str, Any] = {
        "task_id": 1,
        "claim": "The sky is blue.",
        "quote": "The sky appears blue.",
        "confidence": "high",
        # source_url deliberately omitted
    }
    with pytest.raises(ValidationError):
        Finding(**incorrect)


# ---------- Critique ----------
@pytest.mark.parametrize("value", [1, 5, 10])
def test_critique_score_accepts_valid_range(value: int) -> None:
    correct: dict[str, Any] = {
        "score": value,
        "passed": True,
        "weakest_claim": "The sky is blue",
        "required_fixes": [],
    }
    Critique(**correct)


@pytest.mark.parametrize("value", [0, 11, -1, 100])
def test_critique_score_rejects_out_of_range(value: int) -> None:
    wrong: dict[str, Any] = {
        "score": value,
        "passed": True,
        "weakest_claim": "The sky is blue",
        "required_fixes": [],
    }
    with pytest.raises(ValidationError):
        Critique(**wrong)


# ---------- AgentState ----------
def test_agent_state_accepts_well_formed_dict() -> None:
    task = Task(id=1, goal="define meal-kit market size", depends_on=[])

    finding = Finding(
        task_id=1,
        claim="UK meal-kit market grew 12% 2025.",
        quote="The UK meal-kit market grew 12% year-on-year in 2025.",
        source_url="https://example.com/report",
        confidence="high",
    )

    state = AgentState(
        brief="Should a UK cafe chain launch meal kits?",
        memory_context="prior work: none",
        plan=[task],
        findings=[finding],
        draft="",
        critique=None,
        revision_count=0,
        cost_usd=0.0,
    )

    assert state["brief"].startswith("Should")
    assert state["plan"][0].goal == "define meal-kit market size"
    assert state["findings"][0].source_url == "https://example.com/report"
    assert state["critique"] is None
    assert state["revision_count"] == 0
