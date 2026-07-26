"""Episodic memory and static preferences.

These are the two things the agent carries between runs, and they fail in
opposite directions: memory that is never written, and memory that is written
but never read. Both look identical from inside a single run, which is why they
are tested at the graph level rather than by calling the nodes alone.
"""

from unittest.mock import patch

import pytest
import yaml

import agentic_analyst.memory.episodic as episodic
from agentic_analyst.graph import build_graph
from agentic_analyst.memory.preferences import (
    Preferences,
    format_preferences,
    load_preferences,
)
from agentic_analyst.state import AgentState
from tests.conftest import PASSING_CRITIQUE, FakeLLM, failing_critique


# ---------- episodic: the round trip ----------
def test_a_lesson_written_by_one_run_is_read_by_the_next(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """Taskboard 3.3's "done when": run brief A, then brief B, and B sees A.

    This is the only test that proves memory is *wired*. Each half can pass on
    its own while the pair fails — a store nothing queries, or a query against
    a store nothing writes, both look like a healthy run in isolation.
    """
    first = build_graph().invoke(seed_state)
    assert first["memory_context"] == "", "the first run has nothing to remember yet"

    second = build_graph().invoke({**seed_state, "brief": "a related brief"})

    assert "Lessons from past runs" in second["memory_context"]
    assert "Searching for the regulator's own figures." in second["memory_context"]


def test_a_cold_start_is_empty_not_an_error(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    """An empty collection is the normal first run, not a failure."""
    assert episodic.load_memory(seed_state) == {"memory_context": ""}


def test_a_failed_run_is_remembered_and_labelled_as_failed(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """The runs worth learning from are the ones that went badly.

    Routing only successes to memory would leave the agent with an
    unrepresentatively easy history of itself.
    """
    fake_llm.critiques = [failing_critique()]

    build_graph().invoke(seed_state)
    context = episodic.load_memory({**seed_state, "brief": "another brief"})

    assert "failed review" in context["memory_context"]


def test_a_passing_run_is_labelled_passed(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    fake_llm.critiques = [PASSING_CRITIQUE]

    build_graph().invoke(seed_state)
    context = episodic.load_memory({**seed_state, "brief": "another brief"})

    assert "(passed)" in context["memory_context"]


def test_memory_is_written_on_the_caveats_path_too(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """Both terminal paths run through write_memory. A run that exhausted its
    revisions must not vanish from the agent's history."""
    fake_llm.critiques = [failing_critique()]

    build_graph().invoke(seed_state)

    assert fake_llm.calls.count("RunSummary") == 1


def test_the_summariser_uses_the_cheap_tier(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    """Compressing a finished report into three sentences is not strong-tier
    work, and it runs on every single run."""
    with patch("agentic_analyst.memory.episodic.call", side_effect=fake_llm) as summarise:
        build_graph().invoke(seed_state)

    assert summarise.call_args.kwargs["tier"] == "fast"


def test_write_memory_reports_its_own_cost(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    """It makes a model call, so it owes the meter a number — a node that
    forgets is exactly how a run under-reports what it spent (NOTES #6)."""
    result = episodic.write_memory({**seed_state, "critique": PASSING_CRITIQUE})

    assert "cost_usd" in result


def test_memory_is_read_before_planning_not_just_before_writing(
    fake_llm: FakeLLM, seed_state: AgentState
) -> None:
    """The plan should benefit from past lessons, not only the prose.

    Loading memory after the planner would waste it on the cheapest half of
    the run.
    """
    graph = build_graph().get_graph()
    entry = [e.target for e in graph.edges if e.source == "__start__"]

    assert entry == ["load_memory"]


# ---------- episodic: importing must be free ----------
def test_importing_episodic_does_not_touch_the_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`PersistentClient` at module scope would create data/chroma and load an
    embedding model as a side effect of `import`, making `--help` slow and a
    read-only filesystem fatal."""
    episodic._get_collection.cache_clear()
    store = tmp_path / "never-created"

    with patch.object(episodic, "_CHROMA_PATH", store):
        assert not store.exists()


# ---------- preferences ----------
def test_defaults_apply_when_no_prefs_file_exists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with patch("agentic_analyst.memory.preferences._PREFS_PATH", tmp_path / "absent.yaml"):
        prefs = load_preferences()

    assert prefs.tone == Preferences().tone
    assert prefs.length == Preferences().length


def test_prefs_file_overrides_the_defaults(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "prefs.yaml"
    path.write_text(yaml.safe_dump({"tone": "blunt", "length": "one page"}))

    with patch("agentic_analyst.memory.preferences._PREFS_PATH", path):
        prefs = load_preferences()

    assert prefs.tone == "blunt"
    assert prefs.length == "one page"


def test_an_empty_prefs_file_falls_back_to_defaults(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`yaml.safe_load("")` is None, which would explode on `**data`."""
    path = tmp_path / "prefs.yaml"
    path.write_text("")

    with patch("agentic_analyst.memory.preferences._PREFS_PATH", path):
        prefs = load_preferences()

    assert prefs.tone == Preferences().tone


def test_the_shipped_prefs_file_parses() -> None:
    """config/prefs.yaml is committed, so a typo in it breaks every real run."""
    assert load_preferences() is not None


def test_preferences_reach_the_writers_prompt(fake_llm: FakeLLM, seed_state: AgentState) -> None:
    """A preference nothing reads is a config file with no effect."""
    with (
        patch(
            "agentic_analyst.agents.writer.load_preferences",
            return_value=Preferences(tone="blunt", length="one page"),
        ),
        patch("agentic_analyst.agents.writer.call", side_effect=fake_llm) as writer_call,
    ):
        build_graph().invoke(seed_state)

    prompt = writer_call.call_args.kwargs["user"]
    assert "blunt" in prompt
    assert "one page" in prompt


def test_format_preferences_renders_both_knobs() -> None:
    rendered = format_preferences(Preferences(tone="dry", length="short"))

    assert "dry" in rendered
    assert "short" in rendered


@pytest.mark.parametrize("field", ["tone", "length"])
def test_a_partial_prefs_file_keeps_the_other_default(field: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "prefs.yaml"
    path.write_text(yaml.safe_dump({field: "custom"}))

    with patch("agentic_analyst.memory.preferences._PREFS_PATH", path):
        prefs = load_preferences()

    other = "length" if field == "tone" else "tone"
    assert getattr(prefs, field) == "custom"
    assert getattr(prefs, other) == getattr(Preferences(), other)
