"""Unit tests for the typed contracts in state.py."""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_analyst.state import AgentState, Critique, Finding, Fix, Scores, Task


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
def _critique(
    groundedness: int = 8,
    structure: int = 8,
    actionability: int = 8,
    fixes: list[Fix] | None = None,
) -> Critique:
    return Critique(
        scores=Scores(
            groundedness=groundedness,
            structure=structure,
            actionability=actionability,
        ),
        weakest_claim="The sky is blue",
        required_fixes=fixes or [],
    )


def _fix(severity: str) -> Fix:
    return Fix(
        severity=severity,  # type: ignore[arg-type]
        issue="unsupported claim",
        location="## Outlook",
        suggestion="cite a finding or delete it",
    )


@pytest.mark.parametrize("value", [0, 5, 10])
def test_scores_accept_valid_range(value: int) -> None:
    Scores(groundedness=value, structure=value, actionability=value)


@pytest.mark.parametrize("value", [11, -1, 100])
def test_scores_reject_out_of_range(value: int) -> None:
    wrong: dict[str, Any] = {"groundedness": value, "structure": 5, "actionability": 5}
    with pytest.raises(ValidationError):
        Scores(**wrong)


def test_critique_has_no_passed_field() -> None:
    """`passed` is the graph's decision, not the model's.

    If it ever becomes a real field it lands in the JSON schema, the model
    starts filling it in, and the quality gate is judged by the thing it exists
    to judge. This test is the tripwire for that regression.
    """
    assert "passed" not in Critique.model_fields
    assert "passed" not in Critique.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((8, 8, 8), True),  # comfortably over
        ((7, 7, 7), True),  # exactly on the line
        ((7, 7, 6), False),  # mean 6.67, just under
        ((10, 10, 0), False),  # mean 6.67 — one collapsed dimension still fails
        ((0, 0, 0), False),
    ],
)
def test_pass_rule_is_the_mean_of_three_dimensions(
    scores: tuple[int, int, int], expected: bool
) -> None:
    assert _critique(*scores).passed is expected


def test_critical_fix_vetoes_an_otherwise_passing_score() -> None:
    """A perfect score does not survive one critical fix — that is the point."""
    critique = _critique(10, 10, 10, fixes=[_fix("critical")])

    assert critique.scores.mean == 10
    assert critique.passed is False


@pytest.mark.parametrize("severity", ["major", "minor"])
def test_non_critical_fixes_do_not_block_a_passing_draft(severity: str) -> None:
    """Only `critical` is a veto. Otherwise no draft with a nitpick could ship."""
    assert _critique(8, 8, 8, fixes=[_fix(severity)]).passed is True


def test_critical_fixes_filters_by_severity() -> None:
    critique = _critique(fixes=[_fix("minor"), _fix("critical"), _fix("major")])

    assert [f.severity for f in critique.critical_fixes] == ["critical"]


def test_fix_rejects_an_unknown_severity() -> None:
    with pytest.raises(ValidationError):
        Fix(
            severity="blocker",  # type: ignore[arg-type]
            issue="x",
            location="y",
            suggestion="z",
        )


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
