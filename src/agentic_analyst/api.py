"""HTTP interface: submit a brief, poll for the report.

Two endpoints, because a research run takes two to four minutes and an HTTP
request that blocks for that long is a request that times out somewhere it
cannot see — a proxy, a load balancer, a browser. So `POST /briefs` accepts the
work and returns an id immediately, and `GET /runs/{id}` reports on it. That is
the same shape any real job-queue deployment has; the only thing missing here is
a queue that survives a restart, which the README roadmap names rather than
half-builds.

`BackgroundTasks` is FastAPI's in-process version of that queue: the work starts
after the response is sent, in this process, with no persistence. If the server
dies mid-run the run dies with it and its `run.json` stays `running` forever.
Stated plainly because the fix (Celery, RQ, anything with a broker) is a
deployment decision, not a code-quality one.
"""

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runner import RunRecord, execute_run, list_run_ids, load_record, load_report, new_run_id

log = logging.getLogger(__name__)

api = FastAPI(
    title="agentic-analyst",
    version="0.1.0",
    summary="Turn a business brief into a cited research report.",
)


class BriefRequest(BaseModel):
    brief: str = Field(min_length=10, max_length=2000)


class RunAccepted(BaseModel):
    """What `POST /briefs` returns: an id and where to look for the answer."""

    run_id: str
    status: str
    poll: str


class RunResponse(BaseModel):
    """A run's metadata, plus its report once one exists.

    The report is inlined rather than linked because the client that just
    polled for it wants it — a second round trip to fetch a markdown file is a
    round trip for nothing.
    """

    run: RunRecord
    report: str | None = None


@api.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/briefs", response_model=RunAccepted, status_code=202)
def submit_brief(request: BriefRequest, background: BackgroundTasks) -> RunAccepted:
    """Accept a brief and start researching it. Returns before the work is done.

    202, not 201: the run is accepted, not complete, and nothing is created at
    the returned location yet in the sense a 201 promises. `execute_run` writes
    its `run.json` first thing, so by the time the client can poll there is
    already an honest `running` record to read.
    """
    run_id = new_run_id()
    background.add_task(execute_run, request.brief, run_id, entrypoint="api")
    log.info("accepted brief as run %s", run_id)
    return RunAccepted(run_id=run_id, status="accepted", poll=f"/runs/{run_id}")


@api.get("/runs", response_model=list[str])
def list_runs(limit: int = 20) -> list[str]:
    """Recent run ids, newest first."""
    return list_run_ids()[:limit]


@api.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    """Status and, once it exists, the report.

    A run that is still working returns 200 with `status="running"` and no
    report — a poller wants to be told "not yet", not to guess it from a 404.
    A genuinely unknown id is the 404.
    """
    try:
        record = load_record(run_id)
    except ValueError as exc:  # a malformed id, rejected by `run_dir`
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")

    return RunResponse(run=record, report=load_report(run_id))
