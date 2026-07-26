"""Shared fixtures.

pytest imports `conftest.py` automatically — test files never import it. Any
fixture defined here is available by name to every test under `tests/`.
"""

import importlib
import pkgutil
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import agentic_analyst
import agentic_analyst.llm as llm
import evals
from agentic_analyst.agents.planner import Plan
from agentic_analyst.agents.researcher import NextStep, Record, Stop
from agentic_analyst.memory.episodic import RunSummary
from agentic_analyst.settings import SETTINGS
from agentic_analyst.state import AgentState, Critique, Fix, Scores, Task
from evals.judge import Judgement


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


@pytest.fixture(autouse=True)
def no_throttle() -> Iterator[None]:
    """Disable the client-side rate limiter for every test.

    `llm.THROTTLE` spaces real API calls ~5s apart to stay under the free-tier
    quota. Left on, the mocked suite would sleep through it for no reason —
    the tests never touch the network. `test_llm.py` re-enables it explicitly
    where the throttle itself is what is under test.
    """
    original = llm.THROTTLE.min_interval
    llm.THROTTLE.min_interval = 0.0
    yield
    llm.THROTTLE.min_interval = original


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never emit telemetry from the test suite.

    `test_runner_smoke` calls `run.main()`, which opens a root span and flushes
    it. Left alone, a developer with Langfuse keys in `.env` would file two
    fake runs to their real dashboard on every `pytest`, from a suite whose
    entire premise is that it touches no network.

    Dropping the keys routes `get_tracer()` down its null-object path, so the
    tracing code still *runs* in tests — it is just wired to a client that
    discards everything. Skipping the instrumentation instead would leave it
    untested.
    """
    import agentic_analyst.observability as obs

    monkeypatch.setattr(SETTINGS, "langfuse_public_key", None)
    monkeypatch.setattr(SETTINGS, "langfuse_secret_key", None)
    obs.get_tracer.cache_clear()
    yield
    obs.get_tracer.cache_clear()


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path: Any) -> Iterator[None]:
    """Point episodic memory at a throwaway Chroma store for every test.

    Without this the suite reads and writes the developer's real `data/chroma`,
    so tests would see each other's memories, pollute a live store, and pass or
    fail depending on what the last real run happened to remember.
    """
    import agentic_analyst.memory.episodic as episodic

    episodic._get_collection.cache_clear()
    with patch.object(episodic, "_CHROMA_PATH", tmp_path / "chroma"):
        yield
    episodic._get_collection.cache_clear()


@pytest.fixture(autouse=True)
def isolate_runs(tmp_path: Any) -> Iterator[Path]:
    """Point the run registry at a throwaway directory for every test.

    Same reasoning as `isolate_memory`: `runs/` used to be resolved relative to
    the working directory, so a test could redirect it by chdir'ing. It is now
    an absolute path under the repo root — which is what an API server needs,
    since a server's working directory is nobody's business but its own — and
    that makes a stray `execute_run` in a test write a real run into the
    developer's own history. Autouse so no test has to remember.
    """
    import agentic_analyst.runner as runner

    runs = tmp_path / "runs"
    with patch.object(runner, "RUNS_DIR", runs):
        yield runs


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


# ════════════════════════════════════════════════════════════════════
# Stubbing the model
# ════════════════════════════════════════════════════════════════════


def modules_importing_call() -> list[str]:
    """Every module in the codebase that pulled `call` into its own namespace.

    Agents do `from ..llm import call`, which *copies* the reference. Patching
    `llm.call` therefore does nothing to them — the patch has to be applied at
    each module that holds a copy.

    Discovering those modules by walking the packages, rather than listing them
    by hand, is the whole point. The hand-written list is exactly the bug in
    NOTES.md #6: adding a node that calls the model left a hole the fixture knew
    nothing about, and the "mocked" suite quietly started hitting the real API.
    A list has to be maintained; this cannot fall out of date, because a new
    module that imports `call` is found by the same walk that finds the old ones.

    `evals` is walked alongside the package for exactly that reason. The judge
    calls the model too, and it lives outside `src/` — which is precisely the
    kind of "it is not an agent, the fixture does not apply to it" reasoning
    that put a live API call in a mocked suite twice already.
    """
    found = []
    for package in (agentic_analyst, evals):
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            module = importlib.import_module(info.name)
            if getattr(module, "call", None) is llm.call:
                found.append(info.name)
    return sorted(found)


# Drafts are distinguishable so a test can tell an original from a revision.
FIRST_DRAFT = "Energy use is high [F1]. Savings are possible [F1]."
REVISED_DRAFT = "Energy use is high [F1]. Revised: savings are possible [F1]."

PASSING_CRITIQUE = Critique(
    scores=Scores(groundedness=8, structure=8, actionability=8),
    weakest_claim="Savings are possible",
    required_fixes=[],
)


def failing_critique(severity: str = "major") -> Critique:
    """A critique that fails the pass rule, by score and optionally by veto."""
    return Critique(
        scores=Scores(groundedness=4, structure=5, actionability=4),
        weakest_claim="Savings are possible",
        required_fixes=[
            Fix(
                severity=severity,  # type: ignore[arg-type]
                issue="The savings figure is not supported by F1",
                location="## Efficiency options, paragraph 2",
                suggestion="Cite a finding that states a savings figure, or drop the claim.",
            )
        ],
    )


class FakeLLM:
    """A scripted stand-in for `llm.call`, shared by every agent module.

    Dispatches on the requested schema so one stub serves the whole graph, and
    counts what it was asked for so tests can assert on the shape of a run
    without hardcoding how many turns the researcher loop happens to take.
    """

    def __init__(self, charge: float = 0.0) -> None:
        self.charge = charge
        self.calls: list[str] = []  # schema name per call, in order
        # Popped left to right; the last entry repeats once exhausted, so a
        # test can script "fail, fail, pass" or just "always fail".
        self.critiques: list[Critique] = [PASSING_CRITIQUE]

    def __call__(
        self,
        tier: str,
        system: str,
        user: str,
        schema: type | None = None,
    ) -> Any:
        self.calls.append(schema.__name__ if schema is not None else "text")

        if self.charge:
            # Book the charge the way a real call would, so a test can assert
            # against RUN_METER.calls rather than a magic number.
            llm.RUN_METER.total_cost_usd += self.charge
            llm.RUN_METER.calls += 1

        if schema is None:
            # The writer. "Required fixes" only appears in a revision prompt.
            return REVISED_DRAFT if "## Required fixes" in user else FIRST_DRAFT

        if schema is Plan:
            return Plan(
                tasks=[
                    Task(id=1, goal="Assess baseline energy use"),
                    Task(id=2, goal="Identify efficiency options", depends_on=[1]),
                ]
            )

        if schema is NextStep:
            # Record one finding, then stop — the shortest path through the loop.
            if "[did record]" in user:
                return NextStep(step=Stop(action="stop", reason="evidence gathered"))
            return NextStep(
                step=Record(
                    action="record",
                    claim="Baseline energy use is high",
                    quote="Energy use is high",
                    source_url="https://example.com/a",
                    confidence="medium",
                )
            )

        if schema is Critique:
            return self.critiques.pop(0) if len(self.critiques) > 1 else self.critiques[0]

        if schema is Judgement:
            # The eval judge. Scores high enough to pass so a test asserting on
            # the *harness* is not also asserting on a scripted verdict.
            return Judgement(score=8, justification="Well evidenced throughout.")

        if schema is RunSummary:
            return RunSummary(
                brief="Assessed energy options.",
                surprises="Evidence was thin.",
                what_worked="Searching for the regulator's own figures.",
            )

        raise AssertionError(f"FakeLLM has no scripted response for schema {schema!r}")


@pytest.fixture
def fake_llm() -> Iterator[FakeLLM]:
    """Replace `call` in *every* module that imported it, with one fake.

    Yields the fake so a test can script its critiques or inspect what was
    asked of it.
    """
    fake = FakeLLM()
    with ExitStack() as stack:
        for name in modules_importing_call():
            stack.enter_context(patch(f"{name}.call", side_effect=fake))
        yield fake


@pytest.fixture
def stub_llm(fake_llm: FakeLLM) -> FakeLLM:
    """Alias kept for tests that only need the graph to run, not to script it."""
    return fake_llm


@pytest.fixture
def stub_llm_with_cost() -> Iterator[FakeLLM]:
    """Like `fake_llm`, but each call also charges a fixed amount to the meter."""
    fake = FakeLLM(charge=0.01)
    with ExitStack() as stack:
        for name in modules_importing_call():
            stack.enter_context(patch(f"{name}.call", side_effect=fake))
        yield fake


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
