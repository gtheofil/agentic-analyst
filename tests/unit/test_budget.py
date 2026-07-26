"""The run cost cap.

Two layers, and they are testing different promises:

* `llm._check_budget` is the **guarantee** — no call goes out once the cap is
  reached, wherever in the system it was made from.
* the graph's budget routing is the **grace** — the run ends with a usable
  artefact instead of an exception, and without spending anything more.
"""

from pathlib import Path
from typing import Any

import pytest

from agentic_analyst import llm, runner
from agentic_analyst.graph import build_graph, finalize_over_budget
from agentic_analyst.llm import RUN_METER, BudgetExceededError
from agentic_analyst.settings import SETTINGS
from agentic_analyst.state import AgentState, Finding
from tests.conftest import FakeLLM


@pytest.fixture
def tiny_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap one stubbed call's charge will cross."""
    monkeypatch.setattr(SETTINGS, "max_run_cost_usd", 0.005)


def test_a_call_past_the_cap_is_refused_before_it_is_sent(tiny_cap: None, mock_client: Any) -> None:
    RUN_METER.total_cost_usd = 1.0

    with pytest.raises(BudgetExceededError, match="budget"):
        llm.call(tier="fast", system="s", user="u")

    # Refused *before* the request: the mock was never asked to generate.
    mock_client.models.generate_content.assert_not_called()


def test_the_guard_is_a_runtime_error_so_agents_already_handle_it() -> None:
    """The researcher catches `RuntimeError` per task and ends that task.

    Making `BudgetExceededError` anything else would mean every loop in the
    system growing a second except-clause it could forget.
    """
    assert issubclass(BudgetExceededError, RuntimeError)


def test_the_cap_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SETTINGS, "max_run_cost_usd", 0.0)
    RUN_METER.total_cost_usd = 99.0

    assert llm.over_budget() is False


def test_an_over_budget_run_finalizes_instead_of_crashing(
    tiny_cap: None, stub_llm_with_cost: Any, isolate_runs: Path
) -> None:
    """The whole point: a degraded-but-complete report, not a traceback."""
    record = runner.execute_run("Assess office energy options")

    assert record.status == "succeeded"
    report = runner.load_report(record.run_id) or ""
    assert "run budget exhausted" in report.lower()


def test_the_over_budget_run_skips_the_expensive_nodes(
    tiny_cap: None, stub_llm_with_cost: FakeLLM, isolate_runs: Path
) -> None:
    """A guard that fires and then keeps paying is not a guard.

    Note what this can and cannot prove. The stub replaces `call` in every agent
    module, so `_check_budget` — which lives *inside* `call` — is bypassed here;
    the researcher therefore burns its whole loop. What is under test is the
    other layer: once the run is over budget, the graph must not enter the
    writer, the critic or the memory summariser, which between them are two
    thirds of a real run's cost.
    """
    runner.execute_run("Assess office energy options")

    assert "text" not in stub_llm_with_cost.calls, "writer ran while over budget"
    assert "Critique" not in stub_llm_with_cost.calls, "critic ran while over budget"
    assert "RunSummary" not in stub_llm_with_cost.calls, "summariser ran while over budget"


def test_an_over_budget_run_before_a_draft_ships_the_raw_evidence() -> None:
    """The researcher can exhaust the budget before the writer ever runs.

    The findings were paid for; emitting them unsynthesised is worth more than
    an empty file, and is the difference between a degraded run and a lost one.
    """
    state = AgentState(
        brief="b",
        memory_context="",
        plan=[],
        findings=[
            Finding(
                task_id=1,
                claim="Energy use is high",
                quote="Energy use is high",
                source_url="https://example.com/a",
                confidence="high",
            )
        ],
        draft="",
        critique=None,
        revision_count=0,
        cost_usd=0.6,
    )

    report = finalize_over_budget(state)["draft"]

    assert "[F1]" in report
    assert "https://example.com/a" in report


def test_the_over_budget_path_never_reaches_write_memory(
    tiny_cap: None, stub_llm_with_cost: Any, isolate_runs: Path
) -> None:
    """Writing memory costs a model call the guard would refuse.

    Routing the degraded path through it would end the graceful exit in exactly
    the exception it exists to avoid.
    """
    graph = build_graph()
    edges = graph.get_graph().edges

    targets = {edge.target for edge in edges if edge.source == "finalize_over_budget"}
    assert targets == {"__end__"}
