"""The runner and the `runs/` registry.

What matters here is that a run leaves behind everything a later check needs —
the report *and* the evidence it cites — and that a failure is a record rather
than a traceback.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agentic_analyst import runner
from agentic_analyst.llm import RUN_METER


def test_a_run_writes_report_record_and_evidence(stub_llm: Any, isolate_runs: Path) -> None:
    record = runner.execute_run("Assess office energy options")

    directory = isolate_runs / record.run_id
    assert record.status == "succeeded"
    assert (directory / "report.md").read_text(encoding="utf-8").startswith("Energy use")
    assert json.loads((directory / "findings.json").read_text(encoding="utf-8"))
    assert json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    assert json.loads((directory / "critique.json").read_text(encoding="utf-8"))["passed"] is True


def test_findings_are_persisted_in_citation_order(stub_llm: Any, isolate_runs: Path) -> None:
    """`[F1]` means "the first finding in this list", so the order is the contract.

    Persisting them unordered would leave `hardcheck.py` resolving citations
    against a different numbering than the writer used — and still passing.
    """
    record = runner.execute_run("Assess office energy options")

    findings = runner.load_findings(record.run_id)
    raw = json.loads((isolate_runs / record.run_id / "findings.json").read_text(encoding="utf-8"))
    assert [f.claim for f in findings] == [item["claim"] for item in raw]


def test_a_failing_run_is_recorded_not_raised(stub_llm: Any, isolate_runs: Path) -> None:
    """A background task that raises is a traceback nobody reads."""
    with patch.object(runner, "build_graph", side_effect=RuntimeError("no API key")):
        record = runner.execute_run("Assess office energy options")

    assert record.status == "failed"
    assert record.error is not None and "no API key" in record.error
    # The record still lands on disk, so a poller is told what happened.
    assert runner.load_record(record.run_id) == record


def test_the_meter_is_zeroed_per_run_not_per_process(
    stub_llm_with_cost: Any, isolate_runs: Path
) -> None:
    """A long-lived server would otherwise carry run 1's spend into run 2.

    The CLI never saw this — its process is one run — which is exactly why it
    is worth a test: the bug only exists in the deployment nobody runs locally.
    """
    first = runner.execute_run("Assess office energy options")
    assert first.cost_usd > 0

    RUN_METER.total_cost_usd += 99.0  # as if another run had spent it
    second = runner.execute_run("Assess office energy options")

    assert second.cost_usd == pytest.approx(first.cost_usd)


def test_a_run_id_from_a_url_cannot_escape_the_registry() -> None:
    with pytest.raises(ValueError, match="invalid run id"):
        runner.run_dir("../../etc/passwd")


def test_an_unknown_run_reads_as_none(isolate_runs: Path) -> None:
    assert runner.load_record("20200101T000000Z") is None
    assert runner.load_report("20200101T000000Z") is None
    assert runner.load_findings("20200101T000000Z") == []
