"""The writer<->critic loop: routing, termination, and the revision itself.

Taskboard 3.2's "done when" is two claims — a forced-fail run shows
draft → critique → revision → pass, and the loop provably terminates. Both are
asserted here.
"""

import pytest
from langgraph.graph import END

from agentic_analyst.graph import (
    MAX_REVISIONS,
    build_graph,
    finalize_with_caveats,
    route_after_critic,
)
from agentic_analyst.state import AgentState, Critique, Fix, Scores
from tests.conftest import PASSING_CRITIQUE, REVISED_DRAFT, FakeLLM, failing_critique


def _state(critique: Critique | None, revision_count: int = 0) -> AgentState:
    return AgentState(
        brief="Assess energy options",
        memory_context="",
        plan=[],
        findings=[],
        draft="A draft [F1].",
        critique=critique,
        revision_count=revision_count,
        cost_usd=0.0,
    )


# ---------- routing ----------
def test_a_passing_draft_goes_to_memory_not_straight_to_end() -> None:
    """Successes must be remembered too, or the agent's history is all failure."""
    assert route_after_critic(_state(PASSING_CRITIQUE)) == "write_memory"


def test_a_failing_draft_with_budget_left_goes_back_to_the_writer() -> None:
    assert route_after_critic(_state(failing_critique(), revision_count=0)) == "writer"


def test_a_failing_draft_with_no_budget_left_is_finalized() -> None:
    state = _state(failing_critique(), revision_count=MAX_REVISIONS)

    assert route_after_critic(state) == "finalize_with_caveats"


def test_a_critical_fix_fails_a_draft_the_model_scored_perfectly() -> None:
    """The veto is the whole reason severity is a field and not prose.

    A model that scores its own work 10/10 while flagging a fabricated citation
    must not ship. `passed` being computed is what makes this true.
    """
    critique = Critique(
        scores=Scores(groundedness=10, structure=10, actionability=10),
        weakest_claim="none",
        required_fixes=[
            Fix(
                severity="critical",
                issue="[F4] does not exist",
                location="## Costs",
                suggestion="remove the fabricated citation",
            )
        ],
    )

    assert route_after_critic(_state(critique)) == "writer"


def test_the_router_never_returns_end_directly() -> None:
    """Every terminal path runs through write_memory first.

    A router that could return END would let a run finish without recording
    anything — silently, and only on whichever branch was overlooked.
    """
    outcomes = {
        route_after_critic(_state(PASSING_CRITIQUE)),
        route_after_critic(_state(failing_critique(), revision_count=0)),
        route_after_critic(_state(failing_critique(), revision_count=MAX_REVISIONS)),
    }

    assert END not in outcomes


# ---------- the revision actually revises ----------
def test_writer_is_re_run_with_the_fixes_and_produces_a_different_draft(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """Fail once, then pass: the second draft must differ from the first.

    Before the writer was shown the critique it was re-run on byte-identical
    input, so a "revision" could only re-roll the dice. `REVISED_DRAFT` is
    returned by the fake *only* when the prompt contains a required-fixes
    block, so this asserts the fixes really reached the writer.
    """
    fake_llm.critiques = [failing_critique(), PASSING_CRITIQUE]

    result = build_graph().invoke(seed_state)

    assert result["draft"] == REVISED_DRAFT
    assert result["revision_count"] == 1
    assert fake_llm.calls.count("Critique") == 2


def test_writer_prompt_carries_the_previous_draft_and_every_fix(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    from unittest.mock import patch

    fake_llm.critiques = [failing_critique(), PASSING_CRITIQUE]

    with patch("agentic_analyst.agents.writer.call", side_effect=fake_llm) as writer_call:
        build_graph().invoke(seed_state)

    revision_prompt = writer_call.call_args_list[-1].kwargs["user"]
    assert "## Your previous draft" in revision_prompt
    assert "## Required fixes" in revision_prompt
    assert "The savings figure is not supported by F1" in revision_prompt
    assert "## Efficiency options, paragraph 2" in revision_prompt


def test_first_pass_prompt_has_no_revision_block(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    """With no critique yet there is nothing to revise, and saying otherwise
    would invite the model to invent a draft it never wrote."""
    from unittest.mock import patch

    with patch("agentic_analyst.agents.writer.call", side_effect=fake_llm) as writer_call:
        build_graph().invoke(seed_state)

    first_prompt = writer_call.call_args_list[0].kwargs["user"]
    assert "## Your previous draft" not in first_prompt


def test_critical_fixes_are_listed_first_in_the_revision_brief() -> None:
    from agentic_analyst.agents.writer import _format_revision

    critique = Critique(
        scores=Scores(groundedness=3, structure=3, actionability=3),
        weakest_claim="x",
        required_fixes=[
            Fix(severity="minor", issue="typo", location="A", suggestion="fix it"),
            Fix(severity="critical", issue="fabricated", location="B", suggestion="remove"),
            Fix(severity="major", issue="unsupported", location="C", suggestion="cite"),
        ],
    )

    rendered = _format_revision("old draft", critique)
    order = [rendered.index(f"[{s}]") for s in ("critical", "major", "minor")]

    assert order == sorted(order)


# ---------- termination ----------
def test_a_permanently_failing_draft_terminates_with_caveats(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """The guarantee: a critic that never passes anything still ends the run.

    Without a counter the writer and critic would volley indefinitely, at a
    strong-tier call each way. Termination is enforced in the graph, not asked
    for in a prompt.
    """
    fake_llm.critiques = [failing_critique()]  # never passes

    result = build_graph().invoke(seed_state)

    assert result["revision_count"] == MAX_REVISIONS
    assert "## Unresolved review issues" in result["draft"]
    # original + MAX_REVISIONS rewrites, each judged once
    assert fake_llm.calls.count("Critique") == MAX_REVISIONS + 1


def test_revision_count_means_revisions_performed_not_critiques_issued(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """A draft that passes first time was revised zero times.

    When the critic owned this counter, the first critique — which follows the
    original draft and revises nothing — still incremented it, so the run
    reported one revision it never did and stopped a revision early.
    """
    fake_llm.critiques = [PASSING_CRITIQUE]

    result = build_graph().invoke(seed_state)

    assert result["revision_count"] == 0


# ---------- the caveats node ----------
def test_caveats_section_lists_unresolved_fixes_and_the_scores() -> None:
    state = _state(failing_critique("critical"), revision_count=2)

    draft = finalize_with_caveats(state)["draft"]

    assert "A draft [F1]." in draft, "the original draft must survive"
    assert "critical" in draft
    assert "The savings figure is not supported by F1" in draft
    assert "2 revisions" in draft
    assert "Savings are possible" in draft  # the weakest claim


def test_caveats_heading_does_not_collide_with_the_writers_own() -> None:
    """The writer's prompt already ends every report with `## Limitations`.

    A second one would read as a continuation of the author's own caveats
    rather than as a machine-appended review failure.
    """
    state = _state(failing_critique(), revision_count=2)

    draft = finalize_with_caveats(state)["draft"]

    assert "## Limitations" not in draft
    assert "## Unresolved review issues" in draft


def test_caveats_node_is_free() -> None:
    """It is a deterministic string append. If it ever starts costing money,
    something has quietly started calling a model on the failure path."""
    import agentic_analyst.llm as llm

    before = llm.RUN_METER.calls
    finalize_with_caveats(_state(failing_critique(), revision_count=2))

    assert llm.RUN_METER.calls == before


@pytest.mark.parametrize("count", [1, 2])
def test_caveats_pluralises_the_revision_count(count: int) -> None:
    draft = finalize_with_caveats(_state(failing_critique(), revision_count=count))["draft"]

    expected = "1 revision." if count == 1 else f"{count} revisions."
    assert expected in draft
