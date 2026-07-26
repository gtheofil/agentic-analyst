"""The HTTP interface: submit a brief, poll for the report.

`TestClient` runs FastAPI's background tasks synchronously once the response
has been returned, so a POST here completes the whole (stubbed) run before the
next line executes. That is a difference from production worth naming: these
tests prove the wiring and the contract, not the concurrency.
"""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agentic_analyst.api import api

client = TestClient(api)


def test_submitting_a_brief_returns_a_run_id_and_a_poll_url(
    stub_llm: Any, isolate_runs: Path
) -> None:
    response = client.post("/briefs", json={"brief": "Assess office energy options"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["poll"] == f"/runs/{body['run_id']}"


def test_polling_a_finished_run_returns_the_report(stub_llm: Any, isolate_runs: Path) -> None:
    run_id = client.post("/briefs", json={"brief": "Assess office energy options"}).json()["run_id"]

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["run"]["status"] == "succeeded"
    assert body["run"]["run_id"] == run_id
    assert "[F1]" in body["report"]


def test_an_unknown_run_is_a_404(isolate_runs: Path) -> None:
    assert client.get("/runs/20200101T000000Z").status_code == 404


def test_a_traversal_attempt_is_a_400_not_a_file_read(isolate_runs: Path) -> None:
    """`run_id` comes off the URL, so it is validated rather than trusted.

    Starlette rejects the encoded-slash forms itself, before routing — but
    `%2e%2e` decodes to a path segment it is perfectly happy to hand over, and
    without the check in `run_dir` that segment goes straight into a filesystem
    path. Depending on the framework to sanitise your inputs is depending on a
    behaviour nobody wrote down.
    """
    response = client.get("/runs/%2e%2e")

    assert response.status_code == 400
    assert "invalid run id" in response.json()["detail"]


def test_a_too_short_brief_is_rejected_before_any_model_call(isolate_runs: Path) -> None:
    """422 from the schema, not a run that costs money to discover it was junk."""
    assert client.post("/briefs", json={"brief": "hi"}).status_code == 422


def test_listing_runs_returns_newest_first(stub_llm: Any, isolate_runs: Path) -> None:
    (isolate_runs / "20240101T000000Z").mkdir(parents=True)
    (isolate_runs / "20250101T000000Z").mkdir(parents=True)

    assert client.get("/runs").json() == ["20250101T000000Z", "20240101T000000Z"]
