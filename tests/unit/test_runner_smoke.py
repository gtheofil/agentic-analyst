"""The legacy entry point: one brief in, one report file out.

`python -m agentic_analyst.run "<brief>"` is documented in the Phase 1-3 README,
so it keeps working even though `analyst run` is now the real interface. This
test is what makes that a guarantee rather than an intention.
"""

from pathlib import Path
from typing import Any

import pytest

from agentic_analyst import run


def test_runner_writes_report(
    stub_llm: Any,
    isolate_runs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["run", "test brief"])

    run.main()

    reports = list(isolate_runs.glob("*/report.md"))
    assert len(reports) == 1
    assert "[F1]" in reports[0].read_text(encoding="utf-8")


def test_runner_exits_without_a_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run"])

    with pytest.raises(SystemExit):
        run.main()
