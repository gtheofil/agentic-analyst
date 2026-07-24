# src/agentic_analyst/run.py
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_analyst.graph import build_graph
from agentic_analyst.state import AgentState


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m agentic_analyst.run "<brief>"')
        raise SystemExit(1)

    brief = sys.argv[1]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    final_state = build_graph().invoke(
        AgentState(
            brief=brief,
            memory_context="",
            plan=[],
            findings=[],
            draft="",
            critique=None,
            revision_count=0,
            cost_usd=0.0,
        )
    )

    report_path = run_dir / "report.md"
    report_path.write_text(final_state["draft"], encoding="utf-8")

    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()

# python -m agentic_analyst.run "Assess energy options for an office building"
