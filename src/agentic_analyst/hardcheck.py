"""Deterministic checks on a finished run. No model involved, on purpose.

The critic is a model grading a model, and the judge in `evals/` is another
one. Both are useful and neither can tell you whether `[F7]` refers to anything
that exists, or whether a cited URL is a live page. Those are facts, and facts
get checked by code:

* every `[F<id>]` in the report resolves to a finding that was actually gathered
* every source URL answers an HTTP request
* no section was left empty
* the run did not blow its cost cap

The distinction is the point of the whole eval layer. A judge scores *quality*
and can be wrong about it; a hard check verifies *claims about reality* and
cannot. When they disagree, the hard check wins — a beautifully written report
citing findings that do not exist is the single worst output this system can
produce, because it is the one a reader cannot detect by reading.

Exit code is the interface: 0 if nothing failed, 1 otherwise, so CI and
`run_evals.py` both branch on the same answer.
"""

import logging
import re
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from .runner import RunRecord, load_findings, load_record, load_report, run_dir
from .settings import SETTINGS
from .state import Finding

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[F(\d+)\]")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

URL_TIMEOUT = 10.0
# A server that dislikes HEAD, or dislikes robots, is not evidence that the
# page is missing — these downgrade to a warning rather than failing a run.
TOLERATED_STATUSES = frozenset({401, 403, 405, 406, 429, 999})

Status = Literal["pass", "warn", "fail"]


class CheckResult(BaseModel):
    name: str
    status: Status
    detail: str

    def line(self) -> str:
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.status]
        return f"[{mark}] {self.name}: {self.detail}"


class HardCheckReport(BaseModel):
    """Every check for one run, plus the numbers the eval aggregate wants."""

    run_id: str
    results: list[CheckResult]
    citations_total: int = 0
    citations_resolved: int = 0
    urls_checked: int = 0
    urls_ok: int = 0

    @property
    def passed(self) -> bool:
        """Warnings do not fail a run; only failures do.

        A three-state result exists precisely so the open web's ordinary
        weather — a 403 from a bot-hostile server — does not have to be either
        ignored or treated as a fabricated citation.
        """
        return not any(r.status == "fail" for r in self.results)

    @property
    def citation_resolution(self) -> float:
        """Share of `[F<id>]` tags pointing at a real finding, 0-1.

        1.0 when a report cites nothing at all, which is why this is reported
        *alongside* the citation-count check rather than instead of it.
        """
        if self.citations_total == 0:
            return 1.0
        return self.citations_resolved / self.citations_total

    def render(self) -> str:
        head = f"hardcheck {self.run_id}: {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([head, *(r.line() for r in self.results)])


# ── The individual checks ───────────────────────────────────


def check_run_completed(record: RunRecord | None) -> CheckResult:
    if record is None:
        return CheckResult(
            name="run record",
            status="fail",
            detail="no run.json — nothing here to check",
        )
    if record.status != "succeeded":
        return CheckResult(
            name="run record",
            status="fail",
            detail=f"run status is {record.status} ({record.error or 'no error recorded'})",
        )
    return CheckResult(
        name="run record",
        status="pass",
        detail=f"succeeded in {record.duration_seconds or 0:.0f}s",
    )


def check_citations(report: str, findings: list[Finding]) -> tuple[CheckResult, int, int]:
    """Every `[F<id>]` must resolve to a finding that exists.

    Findings are numbered by position, 1-based, the way `writer._format_findings`
    numbers them — which is why `runner` persists them in order. A citation to
    `[F9]` in a run with six findings is a fabricated source, and it is the
    failure this whole script exists for.
    """
    cited = [int(m) for m in CITATION.findall(report)]
    valid = {i for i in range(1, len(findings) + 1)}
    dangling = sorted({i for i in cited if i not in valid})
    resolved = len(cited) - sum(1 for i in cited if i in dangling)

    if not cited:
        return (
            CheckResult(
                name="citations",
                status="fail",
                detail=f"the report cites nothing, with {len(findings)} findings available",
            ),
            0,
            0,
        )
    if dangling:
        return (
            CheckResult(
                name="citations",
                status="fail",
                detail=(
                    f"{len(dangling)} citation(s) resolve to nothing: "
                    f"{', '.join(f'[F{i}]' for i in dangling)} "
                    f"(only F1-F{len(findings)} exist)"
                ),
            ),
            len(cited),
            resolved,
        )
    return (
        CheckResult(
            name="citations",
            status="pass",
            detail=f"{len(cited)} citation(s), all resolving to {len(set(cited))} finding(s)",
        ),
        len(cited),
        resolved,
    )


def check_evidence_used(report: str, findings: list[Finding]) -> CheckResult:
    """How much of the gathered evidence made it into the report.

    Not a failure — the writer is allowed to discard weak findings, and should.
    It is here because a report citing two of eleven findings is either a very
    selective writer or a researcher that wasted 80% of its budget, and neither
    is visible from any other number.
    """
    if not findings:
        return CheckResult(name="evidence use", status="warn", detail="no findings gathered")
    used = {int(m) for m in CITATION.findall(report)}
    share = len(used & set(range(1, len(findings) + 1))) / len(findings)
    return CheckResult(
        name="evidence use",
        status="pass" if share >= 0.5 else "warn",
        detail=f"{len(used)}/{len(findings)} findings cited ({share:.0%})",
    )


def check_sections(report: str) -> CheckResult:
    """No heading may be followed by nothing.

    An empty section is what a model produces when it has run out of evidence
    but not out of structure — the outline of a report where the content should
    be. A heading followed by a *deeper* heading is fine: that is a parent, not
    an empty section.
    """
    lines = report.splitlines()
    headings = [(i, m) for i, line in enumerate(lines) if (m := HEADING.match(line))]
    empty: list[str] = []

    for position, (index, match) in enumerate(headings):
        level = len(match.group(1))
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = [line for line in lines[index + 1 : end] if line.strip()]
        if body:
            continue
        next_is_deeper = (
            position + 1 < len(headings) and len(headings[position + 1][1].group(1)) > level
        )
        if not next_is_deeper:
            empty.append(match.group(2).strip())

    if not report.strip():
        return CheckResult(name="sections", status="fail", detail="the report is empty")
    if empty:
        return CheckResult(
            name="sections",
            status="fail",
            detail=f"{len(empty)} empty section(s): {', '.join(repr(h) for h in empty)}",
        )
    return CheckResult(
        name="sections",
        status="pass",
        detail=f"{len(headings)} section(s), none empty",
    )


def check_cost(record: RunRecord | None, cap: float) -> CheckResult:
    if record is None:
        return CheckResult(name="cost", status="fail", detail="no run.json to read a cost from")
    if cap > 0 and record.cost_usd > cap:
        return CheckResult(
            name="cost",
            status="fail",
            detail=f"${record.cost_usd:.4f} over the ${cap:.2f} cap",
        )
    return CheckResult(
        name="cost",
        status="pass",
        detail=f"${record.cost_usd:.4f} of a ${cap:.2f} cap across {record.model_calls} calls",
    )


def check_urls(findings: list[Finding], *, offline: bool = False) -> tuple[CheckResult, int, int]:
    """Every cited source must still answer.

    HEAD rather than GET: this asks "does this page exist", and downloading the
    body to answer that would be slow and rude at eleven URLs a run. Some
    servers refuse HEAD, which is why a 405 is a warning rather than a failure —
    the check is for dead links, not for pedantry about verbs.
    """
    urls = sorted({f.source_url for f in findings})
    if not urls:
        return CheckResult(name="source urls", status="warn", detail="no sources to check"), 0, 0
    if offline:
        return (
            CheckResult(
                name="source urls",
                status="warn",
                detail=f"{len(urls)} URL(s) not checked (offline)",
            ),
            0,
            0,
        )

    dead: list[str] = []
    tolerated: list[str] = []
    ok = 0
    with httpx.Client(follow_redirects=True, timeout=URL_TIMEOUT) as client:
        for url in urls:
            try:
                response = client.head(url)
            except httpx.HTTPError as exc:
                dead.append(f"{url} ({type(exc).__name__})")
                continue
            if response.status_code < 400:
                ok += 1
            elif response.status_code in TOLERATED_STATUSES:
                tolerated.append(f"{url} ({response.status_code})")
            else:
                dead.append(f"{url} ({response.status_code})")

    if dead:
        return (
            CheckResult(
                name="source urls",
                status="fail",
                detail=f"{len(dead)}/{len(urls)} unreachable: {'; '.join(dead)}",
            ),
            len(urls),
            ok,
        )
    if tolerated:
        return (
            CheckResult(
                name="source urls",
                status="warn",
                detail=(
                    f"{ok}/{len(urls)} reachable; {len(tolerated)} refused the request "
                    f"rather than 404ing: {'; '.join(tolerated)}"
                ),
            ),
            len(urls),
            ok,
        )
    return (
        CheckResult(name="source urls", status="pass", detail=f"all {len(urls)} reachable"),
        len(urls),
        ok,
    )


# ── Running them all ────────────────────────────────────────


def hardcheck(
    run_id: str, *, offline: bool = False, cost_cap: float | None = None
) -> HardCheckReport:
    """Run every check against one finished run."""
    record = load_record(run_id)
    report_text = load_report(run_id) or ""
    findings = load_findings(run_id)
    cap = SETTINGS.max_run_cost_usd if cost_cap is None else cost_cap

    citations, cited_total, cited_resolved = check_citations(report_text, findings)
    urls, urls_checked, urls_ok = check_urls(findings, offline=offline)

    return HardCheckReport(
        run_id=run_id,
        results=[
            check_run_completed(record),
            citations,
            check_evidence_used(report_text, findings),
            check_sections(report_text),
            check_cost(record, cap),
            urls,
        ],
        citations_total=cited_total,
        citations_resolved=cited_resolved,
        urls_checked=urls_checked,
        urls_ok=urls_ok,
    )


def run_id_from_path(argument: str) -> str:
    """Accept either a run id or a path to a run directory.

    `hardcheck.py runs/20260726T120000Z` is what anyone actually types after
    tab-completing, so accepting only the bare id would be a papercut for no
    reason.
    """
    path = Path(argument)
    if path.exists() and path.is_dir():
        return path.resolve().name
    return argument


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a finished run's report against its run.")
    parser.add_argument("run", help="A run id, or a path to runs/<id>/.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the source-URL checks (no network).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=f"Cost cap in USD (default: ${SETTINGS.max_run_cost_usd:.2f}).",
    )
    args = parser.parse_args(argv)

    run_id = run_id_from_path(args.run)
    if not run_dir(run_id).exists():
        print(f"no such run: {run_id}")
        return 1

    result = hardcheck(run_id, offline=args.offline, cost_cap=args.max_cost)
    print(result.render())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
