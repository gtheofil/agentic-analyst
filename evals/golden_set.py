"""Loading the golden set.

A golden brief is a brief plus the things a competent answer could not omit.
Those `must_cover` bullets are the only part of the eval the pipeline never
sees — the agent gets the brief and nothing else — which is what makes coverage
a test rather than a restatement of the input.

The `smoke` flag marks the subset CI runs on every pull request. Three briefs,
because a full ten-brief sweep is roughly a dollar and fifteen minutes and
nobody merges through that; the full set is for prompt changes you actually
want measured.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

Difficulty = Literal["easy", "medium", "hard"]


class GoldenBrief(BaseModel):
    """One evaluation case, validated at load time rather than at 3am in CI."""

    id: str
    domain: str
    difficulty: Difficulty
    smoke: bool = False
    brief: str
    must_cover: list[str] = Field(min_length=3, max_length=5)
    rationale: str = ""

    @property
    def prompt(self) -> str:
        """The brief as the agent receives it — YAML folding leaves newlines."""
        return " ".join(self.brief.split())


def load_golden_set(smoke_only: bool = False) -> list[GoldenBrief]:
    """Every golden brief, in id order so results tables are stable run to run."""
    briefs = [
        GoldenBrief.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(GOLDEN_DIR.glob("*.yaml"))
    ]
    if smoke_only:
        briefs = [b for b in briefs if b.smoke]
    return sorted(briefs, key=lambda b: b.id)
