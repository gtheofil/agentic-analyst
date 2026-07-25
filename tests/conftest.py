"""Shared fixtures.

pytest imports `conftest.py` automatically — test files never import it. Any
fixture defined here is available by name to every test under `tests/`.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import agentic_analyst.llm as llm
from agentic_analyst.state import AgentState, Task


@pytest.fixture(autouse=True)
def reset_meter() -> Iterator[None]:
    """Zero the shared RunMeter around every test.

    `autouse=True` means this runs for every test without being requested.
    RUN_METER is process-global, so without this, costs recorded by one test
    would leak into the next and make assertions order-dependent.
    """
    llm.RUN_METER.reset()
    yield
    llm.RUN_METER.reset()


@pytest.fixture
def mock_client() -> Iterator[MagicMock]:
    """Swap the real Gemini client for a mock. No network, no API key.

    Patches the `_get_client` *function* rather than a module-level client
    object, because the client is now built lazily on first use.
    """
    client = MagicMock()
    # cache_clear() stops a client built by an earlier test being reused here.
    llm._get_client.cache_clear()
    with patch.object(llm, "_get_client", return_value=client):
        yield client
    llm._get_client.cache_clear()


def fake_call(
    tier: str,
    system: str,
    user: str,
    schema: type | None = None,
) -> Any:
    """Stand-in for `llm.call` with the same signature.

    Returns a valid Plan when a schema is requested, and citation-tagged prose
    otherwise — enough for the graph to run end to end deterministically.
    """
    if schema is not None:
        return schema(
            tasks=[
                Task(id=1, goal="Assess baseline energy use"),
                Task(id=2, goal="Identify efficiency options", depends_on=[1]),
            ]
        )
    return "Energy use is high [unverified]. Savings are possible [unverified]."


@pytest.fixture
def stub_llm() -> Iterator[None]:
    """Replace `call` in every agent module with `fake_call`.

    Patching happens where the name is *used* (`agents.planner.call`), not
    where it is defined (`llm.call`) — `from ..llm import call` already copied
    the reference into each agent module, so patching `llm.call` would not be
    seen by them.
    """
    with (
        patch("agentic_analyst.agents.planner.call", side_effect=fake_call),
        patch("agentic_analyst.agents.writer.call", side_effect=fake_call),
    ):
        yield


@pytest.fixture
def stub_llm_with_cost() -> Iterator[float]:
    """Like `stub_llm`, but each call also charges a fixed amount to the meter.

    Yields the per-call charge so a test can assert on the expected total.
    """
    charge = 0.01

    def charging_call(tier: str, system: str, user: str, schema: type | None = None) -> Any:
        llm.RUN_METER.total_cost_usd += charge
        return fake_call(tier, system, user, schema)

    with (
        patch("agentic_analyst.agents.planner.call", side_effect=charging_call),
        patch("agentic_analyst.agents.writer.call", side_effect=charging_call),
    ):
        yield charge


@pytest.fixture
def seed_state() -> AgentState:
    """A well-formed empty state to invoke the graph with."""
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
