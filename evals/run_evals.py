"""Run the golden set end to end and aggregate the result.

    uv run python -m evals.run_evals --smoke      # the 3 CI briefs
    uv run python -m evals.run_evals              # all 10

For each brief: run the real pipeline, hard-check the run, judge the report,
then write `evals/results/summary.{md,json}`.

Three things are worth saying about the shape of this.

**Both graders run, and they are not interchangeable.** The hard-check verifies
facts and cannot be wrong; the judge scores quality and can be. So the exit code
keys off the hard-check and the run completing, while the judge's scores are
reported and only gate if you ask them to. A CI job that fails on a model's
opinion is a CI job that gets ignored within a week.

**The numbers are per-run, not per-report.** Cost and latency come from the
`RunRecord` the pipeline wrote about itself, so the eval's own judging cost —
which is real money, roughly a third of a run — is reported separately rather
than quietly inflating the thing being measured.

**A failed brief does not end the sweep.** Ten briefs is fifteen minutes and a
dollar; losing that because brief seven hit a rate limit would mean nobody ever
runs the full set.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from agentic_analyst.cli import setup_logging
from agentic_analyst.hardcheck import hardcheck
from agentic_analyst.llm import RUN_METER
from agentic_analyst.runner import execute_run, load_findings, load_report
from agentic_analyst.settings import MODEL_TIERS
from agentic_analyst.state import Critique
from evals.golden_set import GoldenBrief, load_golden_set
from evals.judge import DIMENSIONS, Verdict, judge_report

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The same bar the critic gates a draft on. Deliberately not a second threshold
# invented here: if the eval passed reports the pipeline's own critic would have
# rejected, one of the two numbers is lying and you would not know which.
PASS_MEAN = Critique.PASS_MEAN


class EvalResult(BaseModel):
    """One brief's whole story: what it cost, what it checked out at, how it scored."""

    brief_id: str
    domain: str
    difficulty: str
    run_id: str
    completed: bool
    hardcheck_passed: bool
    hardcheck_detail: list[str] = Field(default_factory=list)
    citation_resolution: float = 0.0
    scores: dict[str, int] = Field(default_factory=dict)
    justifications: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    mean_score: float = 0.0
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    findings: int = 0
    revisions: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Both graders have to agree, and the deterministic one has a veto."""
        return self.completed and self.hardcheck_passed and self.mean_score >= PASS_MEAN


class EvalSummary(BaseModel):
    generated_at: str
    tiers: dict[str, str]
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        return mean([float(r.passed) for r in self.results]) if self.results else 0.0

    @property
    def hardcheck_rate(self) -> float:
        return mean([float(r.hardcheck_passed) for r in self.results]) if self.results else 0.0

    def dimension_means(self) -> dict[str, float]:
        return {
            dimension: round(
                mean([r.scores[dimension] for r in self.results if dimension in r.scores]), 2
            )
            for dimension in DIMENSIONS
            if any(dimension in r.scores for r in self.results)
        }

    def _mean_of(self, attribute: str) -> float:
        values = [getattr(r, attribute) for r in self.results]
        return round(mean(values), 4) if values else 0.0

    def render(self) -> str:
        """The committed artefact. A table a human reads, not a blob CI diffs."""
        dims = self.dimension_means()
        lines = [
            "# Eval results",
            "",
            f"Generated {self.generated_at} · {len(self.results)} briefs · "
            f"tiers: strong=`{self.tiers['strong']}`, fast=`{self.tiers['fast']}`",
            "",
            "## Aggregate",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Pass rate (hard-check **and** mean ≥ {PASS_MEAN:.0f}) | {self.pass_rate:.0%} |",
            f"| Hard-check pass rate | {self.hardcheck_rate:.0%} |",
            *(f"| Mean {d} | {v:.2f} / 10 |" for d, v in dims.items()),
            f"| Mean overall | {mean(dims.values()) if dims else 0:.2f} / 10 |",
            f"| Citation resolution | {self._mean_of('citation_resolution'):.1%} |",
            f"| Mean cost per report | ${self._mean_of('cost_usd'):.4f} |",
            f"| Mean judging cost per report | ${self._mean_of('judge_cost_usd'):.4f} |",
            f"| Mean latency | {self._mean_of('duration_seconds'):.0f}s |",
            "",
            "## Per brief",
            "",
            "| Brief | Domain | Diff. | Hard-check | "
            + " | ".join(d[:4].title() for d in DIMENSIONS)
            + " | Mean | Cost | Time |",
            "|---|---|---|---|" + "---|" * (len(DIMENSIONS) + 3),
        ]
        for r in sorted(self.results, key=lambda r: r.brief_id):
            scores = " | ".join(str(r.scores.get(d, "—")) for d in DIMENSIONS)
            lines.append(
                f"| {r.brief_id} | {r.domain} | {r.difficulty} | "
                f"{'pass' if r.hardcheck_passed else '**FAIL**'} | {scores} | "
                f"{r.mean_score:.1f} | ${r.cost_usd:.4f} | {r.duration_seconds:.0f}s |"
            )

        lines += ["", "## Judge reasoning", ""]
        for r in sorted(self.results, key=lambda r: r.brief_id):
            lines.append(f"### {r.brief_id}")
            if r.error:
                lines += [f"Run failed: {r.error}", ""]
            for dimension in DIMENSIONS:
                if dimension in r.justifications:
                    lines.append(
                        f"- **{dimension} {r.scores.get(dimension, '—')}/10** — "
                        f"{r.justifications[dimension]}"
                    )
            if r.missing:
                lines.append(f"- **Not covered:** {'; '.join(r.missing)}")
            if not r.hardcheck_passed:
                lines += ["- **Hard-check:**"] + [f"  - {d}" for d in r.hardcheck_detail]
            lines.append("")
        return "\n".join(lines)


def evaluate(brief: GoldenBrief, *, offline: bool = False) -> EvalResult:
    """Run one golden brief, check it, judge it."""
    log.info("── %s (%s, %s)", brief.id, brief.domain, brief.difficulty)
    record = execute_run(brief.prompt)

    checks = hardcheck(record.run_id, offline=offline)
    log.info("  hardcheck: %s", "pass" if checks.passed else "FAIL")

    # Judged after the run, so `record.cost_usd` is already banked and the
    # judge's own spend is measured separately rather than added to the thing
    # it is measuring.
    with RUN_METER.track() as judging:
        verdict: Verdict | None = (
            judge_report(
                brief_id=brief.id,
                brief=brief.prompt,
                report=load_report(record.run_id) or "",
                must_cover=brief.must_cover,
                findings=load_findings(record.run_id),
            )
            if record.status == "succeeded"
            else None
        )

    return EvalResult(
        brief_id=brief.id,
        domain=brief.domain,
        difficulty=brief.difficulty,
        run_id=record.run_id,
        completed=record.status == "succeeded",
        hardcheck_passed=checks.passed,
        hardcheck_detail=[r.line() for r in checks.results if r.status != "pass"],
        citation_resolution=checks.citation_resolution,
        scores={d: j.score for d, j in verdict.judgements.items()} if verdict else {},
        justifications=(
            {d: j.justification for d, j in verdict.judgements.items()} if verdict else {}
        ),
        missing=verdict.missing if verdict else [],
        mean_score=round(verdict.mean, 2) if verdict else 0.0,
        cost_usd=record.cost_usd,
        judge_cost_usd=judging.usd,
        duration_seconds=record.duration_seconds or 0.0,
        findings=record.findings,
        revisions=record.revisions,
        error=record.error,
    )


def run_evals(briefs: list[GoldenBrief], *, offline: bool = False) -> EvalSummary:
    results: list[EvalResult] = []
    for index, brief in enumerate(briefs, 1):
        log.info("[%d/%d]", index, len(briefs))
        try:
            results.append(evaluate(brief, offline=offline))
        except Exception as exc:  # noqa: BLE001 — one brief must not end the sweep
            log.exception("brief %s blew up entirely", brief.id)
            results.append(
                EvalResult(
                    brief_id=brief.id,
                    domain=brief.domain,
                    difficulty=brief.difficulty,
                    run_id="",
                    completed=False,
                    hardcheck_passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return EvalSummary(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        tiers=dict(MODEL_TIERS),
        results=results,
    )


def write_results(summary: EvalSummary, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = out_dir / "summary.md"
    payload = out_dir / "summary.json"
    markdown.write_text(summary.render(), encoding="utf-8")
    payload.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return markdown, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden set and aggregate the results.")
    parser.add_argument("--smoke", action="store_true", help="Only the CI smoke subset.")
    parser.add_argument("--only", default=None, help="Run one brief by id.")
    parser.add_argument("--offline", action="store_true", help="Skip source-URL checks.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="Where to write results.")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exit non-zero below this pass rate (0-1). Off by default: see the module docstring.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    briefs = load_golden_set(smoke_only=args.smoke)
    if args.only:
        briefs = [b for b in briefs if b.id == args.only]
        if not briefs:
            print(f"no such brief: {args.only}")
            return 1

    summary = run_evals(briefs, offline=args.offline)
    markdown, payload = write_results(summary, args.out)

    print()
    print(summary.render())
    print()
    print(f"Wrote {markdown} and {payload}")

    # Deterministic failures gate unconditionally; the judge's opinion only if
    # asked. Anything a model decides is a poor thing to block a merge on.
    incomplete = [r.brief_id for r in summary.results if not r.completed]
    failed_checks = [r.brief_id for r in summary.results if r.completed and not r.hardcheck_passed]
    if incomplete:
        print(f"FAIL: {len(incomplete)} run(s) did not complete: {', '.join(incomplete)}")
    if failed_checks:
        print(f"FAIL: {len(failed_checks)} hard-check failure(s): {', '.join(failed_checks)}")
    below_bar = summary.pass_rate < args.min_pass_rate
    if below_bar:
        print(
            f"FAIL: pass rate {summary.pass_rate:.0%} below the required {args.min_pass_rate:.0%}"
        )

    return 1 if (incomplete or failed_checks or below_bar) else 0


if __name__ == "__main__":
    raise SystemExit(main())
