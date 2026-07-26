"""The CLI entry point: one brief in, one report file out."""

from pathlib import Path

import pytest

from agentic_analyst import run


def test_runner_writes_report(
    stub_llm: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # chdir into a temp dir so the run writes its `runs/` folder there and not
    # into the repo; monkeypatch undoes both automatically after the test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "test brief"])

    run.main()

    reports = list(tmp_path.glob("runs/*/report.md"))
    assert len(reports) == 1
    assert "[F1]" in reports[0].read_text(encoding="utf-8")


def test_runner_exits_without_a_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run"])

    with pytest.raises(SystemExit):
        run.main()
