"""The judge and the eval harness.

The judge is a model, so these tests cannot check whether its scores are *right*
— that is what the taskboard's human spot-check is for. What they can check is
the design: three independent calls, each shown only what its dimension needs,
and a failure that costs one score rather than the whole sweep.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from agentic_analyst.state import Finding
from evals import judge as judge_module
from evals import run_evals
from evals.golden_set import load_golden_set
from evals.judge import DIMENSIONS, Judgement, judge_report
from tests.conftest import FakeLLM

FINDINGS = [
    Finding(
        task_id=1,
        claim="Meal-kit churn is high",
        quote="Around 60% of subscribers cancel within six months",
        source_url="https://example.com/churn",
        confidence="high",
    )
]
MUST_COVER = ["The churn problem", "Market size", "A named competitor"]


def _judge(fake: FakeLLM) -> Any:
    return judge_report(
        brief_id="test-brief",
        brief="Should a café chain launch meal kits?",
        report="Churn is high [F1].",
        must_cover=MUST_COVER,
        findings=FINDINGS,
    )


def test_each_dimension_is_its_own_call(fake_llm: FakeLLM) -> None:
    """One call per dimension is the whole design.

    Asked for all three at once, a model forms one impression and decomposes it
    — the scores move together and stop being able to disagree. Three calls is
    three times the tokens, bought deliberately.
    """
    verdict = _judge(fake_llm)

    assert fake_llm.calls == ["Judgement"] * len(DIMENSIONS)
    assert set(verdict.judgements) == set(DIMENSIONS)


def test_coverage_sees_the_must_cover_list_and_groundedness_does_not() -> None:
    """Each dimension sees only what it needs, so one cannot drag another down."""
    prompts: dict[str, str] = {}

    def record(tier: str, system: str, user: str, schema: type | None = None) -> Judgement:
        prompts[system.splitlines()[0]] = user
        return Judgement(score=8, justification="fine")

    with patch.object(judge_module, "call", side_effect=record):
        _judge(FakeLLM())

    coverage = prompts["# Coverage"]
    groundedness = prompts["# Groundedness"]

    assert "Must-cover points" in coverage
    assert "Evidence available" not in coverage
    assert "Evidence available" in groundedness
    assert "Must-cover points" not in groundedness


def test_the_evidence_shown_to_the_judge_is_numbered_as_the_writer_numbered_it() -> None:
    """`[F1]` has to mean the same thing to the judge as it did to the writer.

    Numbering that disagreed would turn every correct citation into an apparent
    misattribution, and the groundedness score into noise.
    """
    assert judge_module._format_findings(FINDINGS).startswith("[F1] Meal-kit churn is high")


def test_a_failing_dimension_scores_one_instead_of_ending_the_eval() -> None:
    """Brief seven failing must not cost you the six you already paid for."""

    def explode(tier: str, system: str, user: str, schema: type | None = None) -> Judgement:
        raise ValueError("model would not produce valid JSON")

    with patch.object(judge_module, "call", side_effect=explode):
        verdict = _judge(FakeLLM())

    assert all(j.score == 1 for j in verdict.judgements.values())
    assert "judge failed" in verdict.judgements["coverage"].justification


def test_the_harness_runs_checks_and_judge_and_aggregates(
    fake_llm: FakeLLM, isolate_runs: Path, tmp_path: Path
) -> None:
    """One brief, end to end, with every model call stubbed."""
    brief = load_golden_set(smoke_only=True)[0]

    summary = run_evals.run_evals([brief], offline=True)

    result = summary.results[0]
    assert result.completed
    assert result.hardcheck_passed
    assert result.scores == dict.fromkeys(DIMENSIONS, 8)
    assert summary.pass_rate == 1.0

    markdown, payload = run_evals.write_results(summary, tmp_path / "results")
    assert "Pass rate" in markdown.read_text(encoding="utf-8")
    assert payload.read_text(encoding="utf-8").startswith("{")


def test_judging_cost_is_reported_apart_from_the_run_it_is_judging(
    stub_llm_with_cost: FakeLLM, isolate_runs: Path
) -> None:
    """Otherwise the eval inflates the number it exists to measure."""
    brief = load_golden_set(smoke_only=True)[0]

    result = run_evals.run_evals([brief], offline=True).results[0]

    assert result.judge_cost_usd > 0
    assert result.cost_usd > 0
    assert result.judge_cost_usd != result.cost_usd


def test_a_run_that_did_not_complete_is_not_judged(fake_llm: FakeLLM, isolate_runs: Path) -> None:
    """Paying a strong-tier model to score an empty report is money for nothing."""
    brief = load_golden_set(smoke_only=True)[0]

    with patch.object(run_evals, "execute_run", side_effect=_failed_record):
        result = run_evals.run_evals([brief], offline=True).results[0]

    assert not result.completed
    assert result.scores == {}


def _failed_record(brief: str) -> Any:
    from agentic_analyst.runner import RunRecord

    return RunRecord(
        run_id="20260101T000000Z",
        brief=brief,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:10+00:00",
        error="no API key",
    )
