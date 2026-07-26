"""Tests for the test harness itself.

NOTES.md #6: the `stub_llm` fixture patched agent modules by name, so adding a
node that called the model left a hole the fixture knew nothing about — and the
"fully mocked" suite silently started hitting the real API, at real cost, with
no failure to point at it.

The fixture now discovers its targets by walking the package. These tests guard
that discovery, because a walk that quietly finds nothing fails exactly the same
way the hand-written list did.
"""

from unittest.mock import patch

import pytest

import agentic_analyst.llm as llm
from agentic_analyst.graph import build_graph
from agentic_analyst.state import AgentState
from tests.conftest import FakeLLM, modules_importing_call


def test_discovery_finds_every_module_that_calls_the_model() -> None:
    """The known callers, spelled out.

    Listing them here rather than in the fixture is the point: if this drifts
    from reality the *test* fails loudly, instead of the fixture silently
    letting a node through to the network.
    """
    found = set(modules_importing_call())

    assert found >= {
        "agentic_analyst.agents.critic",
        "agentic_analyst.agents.planner",
        "agentic_analyst.agents.researcher",
        "agentic_analyst.agents.writer",
        "agentic_analyst.memory.episodic",
    }


def test_discovery_is_not_silently_empty() -> None:
    """A walk that matches nothing patches nothing and mocks nothing."""
    assert len(modules_importing_call()) >= 5


def test_no_agent_module_defines_its_own_call() -> None:
    """Every module's `call` must be the same object as `llm.call`.

    A node that wrapped or re-implemented it would slip past both the discovery
    walk and the cost meter, which is collected at that one choke point.
    """
    for name in modules_importing_call():
        module = __import__(name, fromlist=["call"])
        assert module.call is llm.call, f"{name}.call is not llm.call"


def test_a_full_graph_run_never_builds_a_real_client(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """The load-bearing assertion of the whole mocked suite.

    `_get_client` is the only path to the network and it needs an API key, so
    a run that reaches it is a run that would have made a real, billed request.
    """
    with patch.object(llm, "_get_client", side_effect=AssertionError("real client built")):
        build_graph().invoke(seed_state)


def test_the_meter_is_untouched_by_a_fully_stubbed_run(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """`FakeLLM` books no charge unless asked to, so a stubbed run costs zero.

    A non-zero total here means something got through to `llm.call` — the same
    symptom as NOTES #6, caught before it reaches an invoice.
    """
    build_graph().invoke(seed_state)

    assert llm.RUN_METER.calls == 0
    assert llm.RUN_METER.total_cost_usd == 0.0


@pytest.mark.parametrize(
    "node",
    ["planner", "researcher", "writer", "critic", "load_memory", "write_memory"],
)
def test_every_node_is_reachable_in_the_compiled_graph(node: str) -> None:
    """A node added but never wired is dead code that still type-checks."""
    assert node in build_graph().get_graph().nodes
