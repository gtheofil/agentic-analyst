# tests/test_runner_smoke.py
from unittest.mock import patch

from agentic_analyst import run
from agentic_analyst.state import Task


def fake_call(tier, system, user, schema=None):
    if schema is not None:
        return schema(tasks=[Task(id=1, goal="Assess baseline energy use")])
    return "Energy use is high.[unverified]"


def test_runner_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "test brief"])

    with (
        patch("agentic_analyst.agents.planner.call", side_effect=fake_call),
        patch("agentic_analyst.agents.writer.call", side_effect=fake_call),
    ):
        run.main()

    reports = list(tmp_path.glob("runs/*/report.md"))
    assert len(reports) == 1
    assert "[unverified]" in reports[0].read_text(encoding="utf-8")
