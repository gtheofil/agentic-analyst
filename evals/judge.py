"""LLM-as-judge: three anchored dimensions, one model call each.

**One call per dimension, deliberately.** Asking for three scores in one
response gets you three numbers that move together — the model forms an overall
impression and decomposes it afterwards, so a report with great prose and no
evidence scores 8/8/8. Separate calls, each seeing only what that dimension
needs, is what makes the dimensions capable of disagreeing. That they *do*
disagree in practice (groundedness routinely lands two points below
actionability) is the evidence that this was worth three times the tokens.

**The judge is not the critic.** The critic is in the loop: it scores a draft
the writer then revises against, so its verdict is part of the system being
measured. The judge is outside it, with different rubrics, and — for coverage —
with the golden brief's `must_cover` list, which the pipeline never sees. A
judge that only saw what the critic saw would be measuring agreement between two
models, which is a number you can improve without improving anything.

**Its scores are evidence, not truth.** The taskboard's acceptance test is that
a human spot-checks two runs and agrees with the judge. A rubric nobody has
calibrated against their own reading is a number generator.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_analyst.llm import call
from agentic_analyst.state import Finding

log = logging.getLogger(__name__)

RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
DIMENSIONS = ("coverage", "groundedness", "actionability")

# The judge reads a whole report and reasons about it — the same job the critic
# does, and the same reason it is on the strong tier. Judging on the cheap tier
# produces scores that correlate with report length.
JUDGE_TIER = "strong"


class Judgement(BaseModel):
    """One dimension's verdict.

    `justification` is required and comes *with* the score rather than instead
    of it, because a number with no reasoning attached cannot be spot-checked —
    and spot-checking is the only thing standing between this and a plausible
    random number. Each rubric ends by demanding something concrete in it: a
    quoted recommendation, the worst misattribution, the missing points.
    """

    score: int = Field(ge=1, le=10)
    justification: str
    missing: list[str] = Field(
        default_factory=list,
        description="Must-cover points not addressed. Coverage only; empty otherwise.",
    )


class Verdict(BaseModel):
    """All three dimensions for one report."""

    brief_id: str
    judgements: dict[str, Judgement]

    @property
    def mean(self) -> float:
        if not self.judgements:
            return 0.0
        return sum(j.score for j in self.judgements.values()) / len(self.judgements)

    @property
    def missing(self) -> list[str]:
        return self.judgements["coverage"].missing if "coverage" in self.judgements else []


def _rubric(dimension: str) -> str:
    return (RUBRIC_DIR / f"{dimension}.md").read_text(encoding="utf-8")


def _format_findings(findings: list[Finding]) -> str:
    """The evidence list, numbered exactly as the writer numbered it.

    Groundedness is unjudgeable without this: "does [F3] support this sentence"
    requires knowing what F3 says. Numbering that disagreed with the writer's
    would silently turn every judgement into a misattribution.
    """
    if not findings:
        return "(none — the report had no evidence available)"
    return "\n\n".join(
        f'[F{i}] {f.claim}\n  quote: "{f.quote}"\n  source: {f.source_url}'
        for i, f in enumerate(findings, 1)
    )


def _user_prompt(
    dimension: str,
    brief: str,
    report: str,
    must_cover: list[str],
    findings: list[Finding],
) -> str:
    """What each dimension is allowed to see.

    Deliberately not the same for all three. Coverage sees the must-cover list
    and not the findings; groundedness sees the findings and not the must-cover
    list. Handing every judge everything would let a low coverage score drag
    groundedness down with it — the correlation this design exists to break.
    """
    parts = [f"# Brief\n{brief}", f"# Report\n{report}"]
    if dimension == "coverage":
        points = "\n".join(f"- {p}" for p in must_cover)
        parts.append(f"# Must-cover points\n{points}")
    if dimension == "groundedness":
        parts.append(f"# Evidence available to the author\n{_format_findings(findings)}")
    return "\n\n".join(parts)


def judge_report(
    brief_id: str,
    brief: str,
    report: str,
    must_cover: list[str],
    findings: list[Finding],
) -> Verdict:
    """Score one report on all three dimensions.

    A dimension whose call fails scores 1 with the failure as its justification,
    rather than taking the eval down. A judge that crashes on brief seven of ten
    costs you the six you already paid for.
    """
    judgements: dict[str, Judgement] = {}
    for dimension in DIMENSIONS:
        try:
            judgements[dimension] = call(
                tier=JUDGE_TIER,
                system=_rubric(dimension),
                user=_user_prompt(dimension, brief, report, must_cover, findings),
                schema=Judgement,
            )
        except (ValueError, RuntimeError) as exc:
            log.warning("judge failed on %s/%s: %s", brief_id, dimension, exc)
            judgements[dimension] = Judgement(score=1, justification=f"judge failed: {exc}")

        log.info(
            "  %s: %d/10 — %s",
            dimension,
            judgements[dimension].score,
            judgements[dimension].justification[:100],
        )

    return Verdict(brief_id=brief_id, judgements=judgements)


def as_row(verdict: Verdict) -> dict[str, Any]:
    """Flatten to the columns the summary table wants."""
    return {
        "brief_id": verdict.brief_id,
        **{d: verdict.judgements[d].score for d in DIMENSIONS if d in verdict.judgements},
        "mean": round(verdict.mean, 2),
    }
