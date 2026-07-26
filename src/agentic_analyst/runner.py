"""One run, start to finish — and the directory of runs it leaves behind.

Everything that used to live in `run.py`'s `main()` is here as a plain
function, because there are now three callers that need it: the CLI, the API's
background worker, and the eval harness. The CLI keeping the orchestration to
itself would mean the API re-implementing tracing, cost reporting and error
handling slightly differently, and drifting from it thereafter.

**`runs/` is the registry.** Each run writes a directory containing the report
and a `run.json` describing it, and `GET /runs/{id}` is a directory read. That
is a deliberate choice for a demo, not an oversight: a database would need
schema, migrations, a container and a connection string to store facts that are
already files on disk. What it costs is honest to state — no querying across
runs beyond a glob, no concurrent writers, no retention policy — and it is in
the README roadmap.

**Runs are serialised** by a process-wide lock. `RUN_METER` is a module-level
singleton, so two runs sharing a process would bill each other's calls to each
other and trip each other's budget guard. Serialising them is the honest fix
until the meter becomes run-scoped; the alternative — a lock-free meter that is
silently wrong under concurrency — is the kind of bug that only shows up in a
demo.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .graph import build_graph
from .llm import RUN_METER
from .observability import callback_handler, flush, get_tracer
from .settings import PROJECT_ROOT
from .state import AgentState, Critique, Finding, Task

log = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"

REPORT_FILE = "report.md"
RECORD_FILE = "run.json"
FINDINGS_FILE = "findings.json"
PLAN_FILE = "plan.json"
CRITIQUE_FILE = "critique.json"

# Serialises runs within one process — see the module docstring.
_RUN_LOCK = threading.Lock()

RunStatus = Literal["running", "succeeded", "failed"]


class RunRecord(BaseModel):
    """What happened in one run, as JSON on disk next to its report.

    Written twice: once as `running` the moment the run starts, so a poller has
    something truthful to read while the graph is still working, and once at the
    end with the outcome. A run whose process is killed therefore stays
    `running` forever, which is at least an accurate description of what the
    registry knows — inferring `failed` from a stale timestamp would be a guess.
    """

    run_id: str
    brief: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    cost_usd: float = 0.0
    model_calls: int = 0
    findings: int = 0
    revisions: int = 0
    passed_review: bool | None = None
    trace_url: str | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        started = datetime.fromisoformat(self.started_at)
        return (datetime.fromisoformat(self.finished_at) - started).total_seconds()


# ── The registry: runs/ as a read-only-ish database ─────────


def new_run_id() -> str:
    """A UTC timestamp, which sorts lexicographically and reads unambiguously."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_dir(run_id: str) -> Path:
    """Where one run's artefacts live.

    `run_id` arrives from an HTTP path parameter, so it is checked rather than
    trusted: without this, `GET /runs/..%2f..%2fetc` is a file read outside the
    registry. Timestamps are alphanumeric, so refusing everything else costs
    nothing.
    """
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"invalid run id: {run_id!r}")
    return RUNS_DIR / run_id


def list_run_ids() -> list[str]:
    """Every run on disk, newest first (ids sort chronologically)."""
    if not RUNS_DIR.exists():
        return []
    return sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)


def load_record(run_id: str) -> RunRecord | None:
    """The run's metadata, or None if there is no such run."""
    path = run_dir(run_id) / RECORD_FILE
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def load_report(run_id: str) -> str | None:
    """The run's report, or None if it has not written one yet."""
    path = run_dir(run_id) / REPORT_FILE
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_findings(run_id: str) -> list[Finding]:
    """The evidence the report is allowed to cite, in citation order.

    Order is load-bearing: the writer numbers findings `[F1]…[Fn]` by position,
    so the hard-check resolves a citation by indexing this list. Persisting them
    is what makes that check possible after the process has exited.
    """
    path = run_dir(run_id) / FINDINGS_FILE
    if not path.exists():
        return []
    return [Finding.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_artefacts(directory: Path, state: dict[str, Any]) -> None:
    """Persist everything a later check or a human might want to see.

    The report alone is not enough: `scripts/hardcheck.py` has to resolve every
    `[F<id>]` against the findings that existed, and "the model said so" is not
    a check. Anything an audit needs has to outlive the process that produced it.
    """
    plan: list[Task] = state.get("plan", [])
    findings: list[Finding] = state.get("findings", [])
    critique: Critique | None = state.get("critique")

    _write_json(directory / PLAN_FILE, [t.model_dump() for t in plan])
    _write_json(directory / FINDINGS_FILE, [f.model_dump() for f in findings])
    if critique is not None:
        _write_json(
            directory / CRITIQUE_FILE,
            # `passed` is a plain property so it stays out of the schema the
            # model fills; it is recorded here because the *graph's* verdict is
            # exactly what an auditor of this run wants to see.
            {**critique.model_dump(), "passed": critique.passed},
        )
    (directory / REPORT_FILE).write_text(state.get("draft", ""), encoding="utf-8")


def initial_state(brief: str) -> AgentState:
    """The empty state every run starts from."""
    return AgentState(
        brief=brief,
        memory_context="",
        plan=[],
        findings=[],
        draft="",
        critique=None,
        revision_count=0,
        cost_usd=0.0,
    )


# ── Executing a run ─────────────────────────────────────────


def execute_run(brief: str, run_id: str | None = None) -> RunRecord:
    """Take one brief through the graph and write the result to `runs/<id>/`.

    Never raises for an ordinary failure: a model that gives up, a missing key
    or an exhausted quota comes back as a `RunRecord` with `status="failed"` and
    the spend so far. The caller is an HTTP background task as often as a
    terminal, and a background task that raises is a traceback nobody reads.
    """
    run_id = run_id or new_run_id()
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    with _RUN_LOCK:
        # Zero the meter *per run*, not per process. The CLI never noticed the
        # difference because its process is one run; a long-lived API server
        # would carry the first run's spend into the second and trip the budget
        # guard on a run that had not spent anything.
        RUN_METER.reset()

        record = RunRecord(
            run_id=run_id,
            brief=brief,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        _write_json(directory / RECORD_FILE, record.model_dump())

        log.info("run %s | brief: %s", run_id, brief)
        final_state: dict[str, Any] = {}
        error: str | None = None
        trace_url: str | None = None

        try:
            # One root span per run, so the whole thing is a single trace rather
            # than seven unrelated ones. Everything below nests inside it
            # automatically: LangGraph's nodes via the callback handler, and the
            # model calls via the OpenTelemetry context `llm.py` writes into.
            with get_tracer().start_as_current_observation(
                as_type="span",
                name="analyst-run",
                input={"brief": brief},
                metadata={"run_id": run_id},
            ) as root:
                trace_url = get_tracer().get_trace_url()
                try:
                    final_state = dict(
                        build_graph().invoke(
                            initial_state(brief),
                            config={"callbacks": [callback_handler()]},
                        )
                    )
                    root.update(output={"draft": final_state.get("draft", "")})
                except RuntimeError as exc:
                    # Configuration problems (no API key, a rate limit that
                    # outlasted the retries, an empty model response) are the
                    # user's business, not a 40-line LangGraph traceback.
                    error = str(exc)
                    root.update(level="ERROR", status_message=error)
                    log.error("run %s failed: %s", run_id, error)
        finally:
            # In a `finally` because a crashed run is the one whose trace you
            # most want. Spans ship on a background thread, so a process that
            # exits without this sends nothing at all — silently.
            flush()

        critique: Critique | None = final_state.get("critique")
        record = record.model_copy(
            update={
                "status": "failed" if error else "succeeded",
                "finished_at": datetime.now(UTC).isoformat(),
                # The graph's summed `cost_usd` is authoritative when the run
                # completed; a run that died halfway never returned a state, so
                # the meter is the only witness to what it had already spent.
                "cost_usd": final_state.get("cost_usd", RUN_METER.total_cost_usd),
                "model_calls": RUN_METER.calls,
                "findings": len(final_state.get("findings", [])),
                "revisions": final_state.get("revision_count", 0),
                "passed_review": critique.passed if critique is not None else None,
                "trace_url": trace_url,
                "error": error,
            }
        )

        _save_artefacts(directory, final_state)
        _write_json(directory / RECORD_FILE, record.model_dump())

    log.info("meter: %s", RUN_METER.report())
    return record
