"""Unit tests for the researcher loop.

Nothing here touches the network: the model, the search client and the fetcher
are all replaced. What is being tested is the *loop's* behaviour — that it
terminates, that a failing tool becomes an observation rather than an
exception, and that `task_id` is injected rather than trusted.
"""

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_analyst.agents import researcher as researcher_module
from agentic_analyst.agents.researcher import (
    MAX_FINDINGS,
    MAX_OBSERVATION_CHARS,
    MAX_STEPS,
    MAX_TOOL_CALLS,
    MIN_FINDINGS,
    Calc,
    Fetch,
    NextStep,
    Record,
    Search,
    Step,
    Stop,
    researcher,
    run_step,
)
from agentic_analyst.state import Finding, Task
from agentic_analyst.tools.fetch import FetchError, FetchResult
from agentic_analyst.tools.search import SearchResult

TASK = Task(id=7, goal="Quantify UK heat-pump growth in 2024")

_RECORD = Record(
    action="record",
    claim="Installations rose 43%",
    quote="a 43% increase on the 41,000 recorded in 2023",
    source_url="https://mcs.example/2024",
    confidence="high",
)


# ── Helpers ─────────────────────────────────────────────────


def scripted(*steps: Step) -> Callable[..., NextStep]:
    """A fake `call` that returns each step in turn, then repeats the last one.

    Repeating the tail is deliberate: it lets a test say "the model keeps
    choosing to search" without scripting the exact number of turns, which is
    the whole point of the loop guards.
    """
    calls = list(steps)

    def _fake(tier: str, system: str, user: str, schema: type | None = None) -> NextStep:
        step = calls.pop(0) if len(calls) > 1 else calls[0]
        return NextStep(step=step)

    return _fake


@pytest.fixture
def stub_model() -> Iterator[Callable[[Callable[..., NextStep]], MagicMock]]:
    """Patch the researcher's `call` with a scripted stand-in.

    Yields a function that installs the script and hands back the mock, so a
    test can assert on how many turns the model was given and what it saw.
    """
    patcher = patch.object(researcher_module, "call")
    mock = patcher.start()

    def install(script: Callable[..., NextStep]) -> MagicMock:
        mock.side_effect = script
        return mock

    yield install
    patcher.stop()


# ── The schema the model must fill in ───────────────────────


def test_next_step_schema_is_accepted_by_gemini() -> None:
    """Regression test for the bug that made the loop unrunnable.

    `NextStep.step` was a Pydantic *discriminated* union, which serialises to
    `oneOf` + `discriminator`. Gemini's structured-output converter rejects
    both keys, so every researcher call died before reaching the network — and
    no mocked test could catch it, because the conversion happens inside the
    SDK call we mock out. This asserts the conversion itself still works.

    It reaches into a private SDK module on purpose: that conversion is the
    contract being tested, and there is no public entry point to it that does
    not also make a request.
    """
    from google.genai import _transformers

    _transformers.t_schema(None, NextStep)  # must not raise


# ── Termination: the three guards ───────────────────────────


def test_stops_at_the_tool_call_budget(stub_model: Any) -> None:
    """A model that only ever searches is cut off at MAX_TOOL_CALLS."""
    model = stub_model(scripted(Search(action="search", query="anything")))

    with patch.object(researcher_module, "search", return_value=[]) as fake_search:
        findings = researcher(TASK)

    assert fake_search.call_count == MAX_TOOL_CALLS
    assert model.call_count == MAX_TOOL_CALLS
    assert findings == []


def test_stops_at_the_finding_cap(stub_model: Any) -> None:
    """A model that only ever records is cut off at MAX_FINDINGS."""
    records = [
        _RECORD.model_copy(update={"quote": f"distinct quote {i}"}) for i in range(MAX_FINDINGS + 2)
    ]
    model = stub_model(scripted(*records))

    findings = researcher(TASK)

    assert len(findings) == MAX_FINDINGS
    assert model.call_count == MAX_FINDINGS


def test_step_ceiling_catches_a_loop_that_makes_no_progress(stub_model: Any) -> None:
    """The backstop: steps that neither draw down the budget nor record.

    Nothing in the current action set can do that, which is exactly why the
    guard needs a test — it exists so a future change (a dedup rule, a stricter
    `Finding`) cannot silently turn the loop into an unbounded one.
    """
    model = stub_model(scripted(_RECORD))

    with patch.object(researcher_module, "run_step", return_value="ERROR: nothing happened"):
        findings = researcher(TASK)

    assert model.call_count == MAX_STEPS
    assert findings == []


def test_an_early_stop_is_refused_exactly_once(stub_model: Any) -> None:
    """A model that stops with no findings gets one push-back, then is obeyed.

    Refusing forever would be an unbounded loop; refusing never means most
    tasks come back with a single finding, which is what live runs did.
    """
    model = stub_model(scripted(Stop(action="stop", reason="nothing to find")))

    with patch.object(researcher_module, "search") as fake_search:
        findings = researcher(TASK)

    assert model.call_count == 2
    assert "[stop refused]" in model.call_args_list[1].kwargs["user"]
    assert fake_search.call_count == 0
    assert findings == []


def test_stop_is_obeyed_once_the_minimum_is_met(stub_model: Any) -> None:
    """With MIN_FINDINGS in hand there is nothing to push back on."""
    second = _RECORD.model_copy(update={"quote": "40,426 were certified in 2023"})
    model = stub_model(scripted(_RECORD, second, Stop(action="stop", reason="answered")))

    findings = researcher(TASK)

    assert len(findings) == MIN_FINDINGS
    assert model.call_count == 3


def test_stop_is_obeyed_when_the_tool_budget_is_already_spent(stub_model: Any) -> None:
    """Push-back only makes sense if the model can still act on it."""
    steps = [Search(action="search", query="q")] * MAX_TOOL_CALLS
    model = stub_model(scripted(*steps, Stop(action="stop", reason="out of road")))

    with patch.object(researcher_module, "search", return_value=[]):
        findings = researcher(TASK)

    # The budget check ends the loop before the stop is ever read.
    assert model.call_count == MAX_TOOL_CALLS
    assert findings == []


# ── Failures are observations, not exceptions ───────────────


def test_tool_exception_becomes_an_observation(stub_model: Any) -> None:
    """A dead link costs one step, not the run."""
    model = stub_model(
        scripted(
            Fetch(action="fetch", url="https://paywalled.example"),
            Stop(action="stop", reason="blocked"),
        )
    )

    with patch.object(researcher_module, "fetch", side_effect=FetchError("No extractable text")):
        findings = researcher(TASK)  # must not raise

    second_prompt = model.call_args_list[1].kwargs["user"]
    assert "ERROR during fetch" in second_prompt
    assert "No extractable text" in second_prompt
    assert findings == []


def test_model_failure_ends_the_task_without_killing_the_run(stub_model: Any) -> None:
    """`call` gives up after its own retry → this task returns what it has."""
    stub_model(MagicMock(side_effect=ValueError("failed to produce valid NextStep")))

    assert researcher(TASK) == []


def test_a_failed_step_does_not_consume_the_tool_budget_twice(stub_model: Any) -> None:
    """A search that raises still costs exactly one of the six calls."""
    model = stub_model(scripted(Search(action="search", query="q")))

    with patch.object(researcher_module, "search", side_effect=RuntimeError("tavily down")):
        researcher(TASK)

    assert model.call_count == MAX_TOOL_CALLS


# ── What the model is trusted with ──────────────────────────


def test_task_id_is_injected_not_taken_from_the_model() -> None:
    """The `Record` action has no `task_id` field, and this is why."""
    findings: list[Finding] = []

    run_step(_RECORD, task_id=7, findings=findings)

    assert findings[0].task_id == 7
    assert findings[0].quote == _RECORD.quote
    assert findings[0].source_url == _RECORD.source_url
    assert findings[0].confidence == "high"


def test_the_same_quote_cannot_be_recorded_twice() -> None:
    """Two findings on one sentence is one finding the writer would double-cite.

    Whitespace and case differ because the model retypes the quote; that is
    not new evidence.
    """
    findings: list[Finding] = []
    run_step(_RECORD, task_id=1, findings=findings)

    reworded = _RECORD.model_copy(
        update={"claim": "Growth was 43%", "quote": f"  {_RECORD.quote.upper()}\n"}
    )
    result = run_step(reworded, task_id=1, findings=findings)

    assert result.startswith("ERROR: that quote is already recorded")
    assert len(findings) == 1


def test_a_different_quote_from_the_same_page_is_fine() -> None:
    findings: list[Finding] = []
    run_step(_RECORD, task_id=1, findings=findings)

    other = _RECORD.model_copy(update={"quote": "40,426 installations were certified in 2023"})
    run_step(other, task_id=1, findings=findings)

    assert len(findings) == 2


def test_record_result_tells_the_model_where_it_is() -> None:
    findings: list[Finding] = []

    result = run_step(_RECORD, task_id=1, findings=findings)

    assert result == f"Recorded finding 1/{MAX_FINDINGS}."


# ── Observations the model actually has to read ─────────────


def test_fetched_pages_are_truncated() -> None:
    """A 1MB page must not be pasted into the transcript whole."""
    huge = FetchResult(url="https://example.com", text="x" * (MAX_OBSERVATION_CHARS * 3))

    with patch.object(researcher_module, "fetch", return_value=huge):
        observation = run_step(Fetch(action="fetch", url="https://example.com"), 1, [])

    assert "truncated" in observation
    assert len(observation) < MAX_OBSERVATION_CHARS + 200


def test_search_results_are_rendered_as_readable_lines() -> None:
    hits = [
        SearchResult(title="MCS 2024 stats", url="https://mcs.example/2024", snippet="60,000"),
        SearchResult(title="A blog", url="https://blog.example", snippet="copied it"),
    ]

    with patch.object(researcher_module, "search", return_value=hits):
        observation = run_step(Search(action="search", query="q"), 1, [])

    assert "1. MCS 2024 stats" in observation
    assert "https://mcs.example/2024" in observation
    assert "60,000" in observation
    assert "SearchResult(" not in observation, "raw Pydantic reprs waste tokens"


def test_empty_search_says_so() -> None:
    with patch.object(researcher_module, "search", return_value=[]):
        assert run_step(Search(action="search", query="q"), 1, []) == "No results."


def test_calc_result_is_returned_as_text() -> None:
    assert run_step(Calc(action="calc", expression="(1200*0.23)/12"), 1, []) == "23.0"


def test_hostile_calc_is_reported_not_raised() -> None:
    result = run_step(Calc(action="calc", expression="__import__('os')"), 1, [])

    assert result.startswith("ERROR during calc")


# ── The prompt the loop depends on ──────────────────────────


def test_system_prompt_exists_and_demands_verbatim_quotes() -> None:
    """The loop reads this file on every task; a rename must fail here."""
    text = researcher_module._PROMPT_PATH.read_text(encoding="utf-8")

    assert "verbatim" in text.lower()
    for action in ("search", "fetch", "calc", "record", "stop"):
        assert action in text.lower(), f"prompt never mentions the {action} action"
