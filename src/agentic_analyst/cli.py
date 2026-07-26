"""The terminal interface: `analyst run`, `analyst show`, `analyst serve`.

Thin on purpose. Everything here is argument parsing and printing; the run
itself belongs to `runner.py`, which the API calls too. A CLI that owns the
orchestration is a CLI the API has to re-implement.
"""

import logging

import typer

from .hardcheck import hardcheck
from .runner import RunRecord, execute_run, list_run_ids, load_record, load_report, run_dir

app = typer.Typer(
    name="analyst",
    help="Turn a business brief into a cited research report.",
    no_args_is_help=True,
    add_completion=False,
)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure log output.

    Only entry points do this. Library modules just call
    `logging.getLogger(__name__)` and let whoever imports them decide where the
    output goes — otherwise importing the package would hijack the logging
    config of any application using it.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # The Google SDK is chatty at INFO; we only want its warnings.
    logging.getLogger("google_genai").setLevel(logging.WARNING)


def print_summary(record: RunRecord) -> None:
    """The end-of-run block, identical whichever entry point produced it."""
    if record.error:
        typer.secho(f"error: {record.error}", fg=typer.colors.RED, err=True)
        typer.echo(f"Spent before failing: ${record.cost_usd:.6f}", err=True)
    else:
        typer.echo(f"Wrote {(run_dir(record.run_id) / 'report.md').resolve()}")

    verdict = {True: "passed review", False: "failed review", None: "unreviewed"}[
        record.passed_review
    ]
    typer.echo(
        f"Run {record.run_id}: {record.status}, {verdict}, "
        f"{record.findings} findings, {record.revisions} revision(s)"
    )
    typer.echo(f"Cost: ${record.cost_usd:.6f} across {record.model_calls} model calls")
    if record.duration_seconds is not None:
        typer.echo(f"Time: {record.duration_seconds:.0f}s")
    if record.trace_url:
        typer.echo(f"Trace: {record.trace_url}")


def _resolve(run_id: str) -> str:
    """Turn `latest` into a real run id, or fail with a usable message."""
    if run_id != "latest":
        return run_id
    ids = list_run_ids()
    if not ids:
        typer.secho('no runs yet — try `analyst run "<brief>"`', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    return ids[0]


@app.command()
def run(
    brief: str = typer.Argument(..., help="The business question to research."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log every model call."),
) -> None:
    """Research one brief and write a cited report to runs/<id>/."""
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    record = execute_run(brief)
    print_summary(record)
    # A failed run exits non-zero so a shell script, a Makefile or CI can tell.
    if record.status == "failed":
        raise typer.Exit(1)


@app.command()
def show(
    run_id: str = typer.Argument("latest", help="A run id, or `latest`."),
    report: bool = typer.Option(False, "--report", "-r", help="Print the report itself."),
) -> None:
    """Show what a past run did, and optionally its report."""
    resolved = _resolve(run_id)
    record = load_record(resolved)
    if record is None:
        typer.secho(f"no such run: {resolved}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Brief: {record.brief}")
    print_summary(record)

    if report:
        typer.echo("")
        typer.echo(load_report(resolved) or "(no report written)")


@app.command()
def runs(limit: int = typer.Option(10, help="How many runs to list.")) -> None:
    """List recent runs, newest first."""
    ids = list_run_ids()[:limit]
    if not ids:
        typer.echo("no runs yet")
        return
    for run_id in ids:
        record = load_record(run_id)
        if record is None:
            typer.echo(f"{run_id}  (no run.json — from an older version)")
            continue
        typer.echo(f"{run_id}  {record.status:<9} ${record.cost_usd:.4f}  {record.brief[:60]}")


@app.command()
def check(
    run_id: str = typer.Argument("latest", help="A run id, or `latest`."),
    offline: bool = typer.Option(False, "--offline", help="Skip the source-URL checks."),
) -> None:
    """Hard-check a run: citations resolve, sources answer, nothing empty, cost in cap."""
    result = hardcheck(_resolve(run_id), offline=offline)
    typer.echo(result.render())
    if not result.passed:
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
) -> None:
    """Serve the HTTP API (POST /briefs, GET /runs/{id})."""
    import uvicorn

    setup_logging()
    uvicorn.run("agentic_analyst.api:api", host=host, port=port)


if __name__ == "__main__":
    app()
