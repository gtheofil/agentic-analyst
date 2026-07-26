"""CLI entry point: turn one brief into one report.

python -m agentic_analyst.run "Assess energy options for an office building"
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_analyst.graph import build_graph
from agentic_analyst.llm import RUN_METER
from agentic_analyst.observability import callback_handler, flush, get_tracer
from agentic_analyst.state import AgentState

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure log output.

    Only the entry point does this. Library modules just call
    `logging.getLogger(__name__)` and let whoever imports them decide where
    the output goes — otherwise importing the package would hijack the logging
    config of any application using it.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # The Google SDK is chatty at INFO; we only want its warnings.
    logging.getLogger("google_genai").setLevel(logging.WARNING)


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


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: agentic-analyst "<brief>"')
        raise SystemExit(1)

    setup_logging()
    brief = sys.argv[1]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("run %s | brief: %s", timestamp, brief)

    # One root span per run, so the whole thing is a single trace rather than
    # seven unrelated ones. Everything below nests inside it automatically:
    # LangGraph's nodes via the callback handler, and the model calls via the
    # OpenTelemetry context that `llm.py` writes into. Neither has to be handed
    # a parent explicitly.
    trace_url: str | None = None
    try:
        with get_tracer().start_as_current_observation(
            as_type="span",
            name="analyst-run",
            input={"brief": brief},
            metadata={"run_id": timestamp},
        ) as root:
            trace_url = get_tracer().get_trace_url()
            try:
                final_state = build_graph().invoke(
                    initial_state(brief),
                    config={"callbacks": [callback_handler()]},
                )
            except RuntimeError as exc:
                # Configuration problems (no API key, a rate limit that
                # outlasted the retries, an empty model response) are the
                # user's business, not a 40-line LangGraph traceback. Report
                # the spend either way: a run that died halfway still cost real
                # money, and hiding that is how you learn about it on the
                # invoice.
                root.update(level="ERROR", status_message=str(exc))
                print(f"error: {exc}", file=sys.stderr)
                print(f"Spent before failing: {RUN_METER.report()}", file=sys.stderr)
                if trace_url:
                    print(f"Trace: {trace_url}", file=sys.stderr)
                raise SystemExit(1) from exc

            # The root span's own I/O is the trace's I/O — `set_trace_io` is
            # deprecated in v4 precisely because this is the same thing.
            root.update(output={"draft": final_state["draft"]})
    finally:
        # In a `finally` for the same reason the failure branch above reports
        # the spend: a crashed run is the one whose trace you most want. Spans
        # are batched on a background thread, so a CLI that exits without this
        # sends nothing at all — silently, with no error anywhere.
        flush()

    report_path = run_dir / "report.md"
    report_path.write_text(final_state["draft"], encoding="utf-8")

    # Two views of the same spend, and they should agree: `cost_usd` is summed
    # by the graph from what each node reported, RUN_METER counts raw API calls.
    log.info("meter: %s", RUN_METER.report())
    print(f"Wrote {report_path.resolve()}")
    print(f"Cost: ${final_state['cost_usd']:.6f} across {RUN_METER.calls} model calls")
    if trace_url:
        print(f"Trace: {trace_url}")


if __name__ == "__main__":
    main()
