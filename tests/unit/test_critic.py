"""Unit tests for the critic node.

The pass rule itself is tested in `test_state.py`, where it lives. These cover
what the *node* is responsible for: what it shows the model, and what it hands
back to the graph.
"""

from typing import Any
from unittest.mock import patch

from agentic_analyst.agents.critic import _render_sources, critic
from agentic_analyst.state import AgentState, Critique, Finding, Fix, Scores

F1 = Finding(
    task_id=1,
    claim="UK solar installations rose 43% in 2024",
    quote="installations rose 43% year on year",
    source_url="https://gov.example/solar",
    confidence="high",
)
F2 = Finding(
    task_id=2,
    claim="A 100kW system costs $150,000 before incentives",
    quote="$1.50 to $3.00 per watt",
    source_url="https://vendor.example/costs",
    confidence="low",
)

VERDICT = Critique(
    scores=Scores(groundedness=6, structure=7, actionability=5),
    weakest_claim="Costs will fall",
    required_fixes=[
        Fix(
            severity="major",
            issue="unsupported",
            location="## Costs",
            suggestion="cite F2",
        )
    ],
)


def _state(findings: list[Finding], draft: str = "A draft [F1].") -> AgentState:
    return AgentState(
        brief="Assess energy options",
        memory_context="",
        plan=[],
        findings=findings,
        draft=draft,
        critique=None,
        revision_count=0,
        cost_usd=0.0,
    )


def _prompt_for(state: AgentState) -> str:
    """Run the critic with the model stubbed, return the user prompt it sent."""
    with patch("agentic_analyst.agents.critic.call", return_value=VERDICT) as model:
        critic(state)
    user: str = model.call_args.kwargs["user"]
    return user


# ---------- what the critic is shown ----------
def test_sources_use_the_same_f_labels_the_draft_cites() -> None:
    """The critic checks whether `[F3]` in the draft is backed by F3.

    It cannot do that if it sees the evidence under different labels than the
    writer used. This is the one alignment the groundedness score depends on.
    """
    rendered = _render_sources([F1, F2])

    assert "[F1]" in rendered
    assert "[F2]" in rendered
    assert "[F0]" not in rendered


def test_sources_expose_quote_url_and_confidence() -> None:
    """Groundedness is unjudgeable without the verbatim quote and its source."""
    rendered = _render_sources([F2])

    assert "$1.50 to $3.00 per watt" in rendered
    assert "https://vendor.example/costs" in rendered
    assert "low" in rendered


def test_empty_findings_are_flagged_not_silently_blank() -> None:
    """A blank sources block reads as "nothing to check" rather than "nothing
    is supported" — the opposite of the truth, and a free pass on groundedness."""
    rendered = _render_sources([])

    assert "no sources" in rendered.lower()
    assert "unsupported" in rendered.lower()


def test_prompt_contains_both_sources_and_draft() -> None:
    prompt = _prompt_for(_state([F1], draft="Solar rose sharply [F1]."))

    assert "Solar rose sharply [F1]." in prompt
    assert "installations rose 43% year on year" in prompt


# ---------- what the critic hands back ----------
def test_critic_writes_the_verdict_and_its_cost() -> None:
    with patch("agentic_analyst.agents.critic.call", return_value=VERDICT):
        result = critic(_state([F1]))

    assert result["critique"] is VERDICT
    assert "cost_usd" in result


def test_critic_does_not_touch_revision_count() -> None:
    """The writer owns that counter — it counts revisions performed, and the
    critic performs none. Two nodes incrementing it is how it drifts."""
    with patch("agentic_analyst.agents.critic.call", return_value=VERDICT):
        result: dict[str, Any] = critic(_state([F1]))

    assert "revision_count" not in result


def test_critic_requests_the_critique_schema() -> None:
    """Structured output is what makes the verdict machine-readable at all."""
    with patch("agentic_analyst.agents.critic.call", return_value=VERDICT) as model:
        critic(_state([F1]))

    assert model.call_args.kwargs["schema"] is Critique
