"""The deterministic checks.

The load-bearing test is `test_a_citation_to_a_deleted_finding_fails`: it is the
exact sabotage the taskboard asks for, and the reason the whole script exists.
Everything else here is guarding the checks against being *too* eager, which is
the failure mode that gets a check switched off.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_analyst import hardcheck as hc
from agentic_analyst import runner


@pytest.fixture
def finished_run(stub_llm: Any, isolate_runs: Path) -> str:
    """A real (stubbed) run on disk, so the checks read what the runner writes."""
    return runner.execute_run("Assess office energy options").run_id


def test_a_good_run_passes(finished_run: str) -> None:
    result = hc.hardcheck(finished_run, offline=True)

    assert result.passed, result.render()
    assert result.citation_resolution == 1.0


def test_a_citation_to_a_deleted_finding_fails(finished_run: str, isolate_runs: Path) -> None:
    """Delete the evidence, keep the report: this must not pass.

    A report that cites `[F1]` when no F1 exists is the one failure a reader
    cannot catch by reading — the citation looks exactly like a real one.
    """
    path = isolate_runs / finished_run / "findings.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    result = hc.hardcheck(finished_run, offline=True)

    assert not result.passed
    assert result.citation_resolution < 1.0


def test_an_empty_section_fails() -> None:
    report = "# Report\n\n## Findings\n\nSomething.\n\n## Recommendations\n"

    assert hc.check_sections(report).status == "fail"


def test_a_parent_heading_is_not_an_empty_section() -> None:
    """`# Title` immediately followed by `## Section` is normal structure.

    Getting this wrong would fail every well-formed report, which is how a
    check ends up commented out.
    """
    report = "# Report\n## Findings\nSomething.\n"

    assert hc.check_sections(report).status == "pass"


def test_a_report_that_cites_nothing_fails() -> None:
    result, total, resolved = hc.check_citations("# Report\nNo citations here.", [])

    assert result.status == "fail"
    assert (total, resolved) == (0, 0)


def test_cost_over_the_cap_fails(finished_run: str, isolate_runs: Path) -> None:
    record = runner.load_record(finished_run)
    assert record is not None
    over = record.model_copy(update={"cost_usd": 5.0})
    (isolate_runs / finished_run / "run.json").write_text(over.model_dump_json(), encoding="utf-8")

    result = hc.hardcheck(finished_run, offline=True, cost_cap=0.5)

    assert not result.passed


def test_a_dead_url_fails_but_a_bot_block_only_warns(
    finished_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """403 is a server with opinions; 404 is a source that is not there.

    Treating them the same means either tolerating fabricated URLs or failing
    runs over Cloudflare, and both end with the check being ignored.
    """
    findings = runner.load_findings(finished_run)

    def respond(status: int) -> Any:
        def head(self: Any, url: str) -> httpx.Response:
            return httpx.Response(status, request=httpx.Request("HEAD", url))

        return head

    monkeypatch.setattr(httpx.Client, "head", respond(403))
    assert hc.check_urls(findings)[0].status == "warn"

    monkeypatch.setattr(httpx.Client, "head", respond(404))
    assert hc.check_urls(findings)[0].status == "fail"


def test_a_timeout_fails(finished_run: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(self: Any, url: str) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow")

    monkeypatch.setattr(httpx.Client, "head", timeout)

    assert hc.check_urls(runner.load_findings(finished_run))[0].status == "fail"


def test_the_exit_code_is_the_interface(finished_run: str, isolate_runs: Path) -> None:
    """CI and `run_evals.py` both branch on this, so it is worth asserting."""
    assert hc.main([finished_run, "--offline"]) == 0

    (isolate_runs / finished_run / "findings.json").write_text("[]", encoding="utf-8")
    assert hc.main([finished_run, "--offline"]) == 1


def test_a_run_directory_path_is_accepted_not_just_an_id(
    finished_run: str, isolate_runs: Path
) -> None:
    assert hc.run_id_from_path(str(isolate_runs / finished_run)) == finished_run
