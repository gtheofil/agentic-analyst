"""Typed contracts that flow between agents.

This is the load-bearing design decision of the whole system: every agent's
output is one of these Pydantic models, so a downstream agent can never receive
malformed input. A boundary is schema-validated, and a parse failure becomes a
retry rather than a crash three nodes later.

`AgentState` is the single object LangGraph threads through the entire graph.
Each node reads some of its fields and writes others.
"""

import operator
from typing import Annotated, ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field


class Task(BaseModel):
    """One research task, produced by the planner.

    The planner decomposes a brief into 3-6 of these. `depends_on` lets a task
    declare that it needs the output of earlier tasks first.
    """

    id: int
    goal: str
    depends_on: list[int] = Field(default_factory=list)


class Finding(BaseModel):
    """A single evidenced claim, produced by the researcher.

    `quote` must be *verbatim* text from the source: it is what lets the writer
    cite the claim and lets the citation checker later verify it is real.
    """

    task_id: int
    claim: str
    quote: str
    source_url: str
    confidence: Literal["high", "medium", "low"]


class Fix(BaseModel):
    """One concrete problem the critic wants corrected.

    Severity is a separate field rather than a prefix in the text because the
    pass rule keys off it: a single `critical` fix blocks a draft no matter how
    well it scored. Something the router branches on cannot live inside prose.
    """

    severity: Literal["critical", "major", "minor"]
    issue: str  # what is wrong
    location: str  # where — the section, claim or citation it applies to
    suggestion: str  # what to do about it


class Scores(BaseModel):
    """The three rubric dimensions, scored 0-10 against the anchors in critic.md.

    Kept as separate fields, not one overall number, because the pass rule is
    defined on their mean — and because "groundedness 3, structure 9" is
    actionable feedback where "6/10" is not.
    """

    groundedness: int = Field(ge=0, le=10)
    structure: int = Field(ge=0, le=10)
    actionability: int = Field(ge=0, le=10)

    @property
    def mean(self) -> float:
        return (self.groundedness + self.structure + self.actionability) / 3


class Critique(BaseModel):
    """The critic's verdict on a draft.

    `weakest_claim` is mandatory on purpose: forcing the critic to name the
    single weakest claim stops it from rubber-stamping drafts.

    Note what is *not* here: a `passed` field. Whether a draft ships is the
    graph's decision, not the model's, so `passed` is a computed property over
    the scores and fixes rather than a boolean the critic reports about itself.
    A model that likes its own work can still say 9/10 — it cannot also say
    "and therefore ship it". Plain `@property`, not `computed_field`, so it
    stays out of the JSON schema the model is asked to fill.
    """

    scores: Scores
    weakest_claim: str
    required_fixes: list[Fix] = Field(default_factory=list)

    PASS_MEAN: ClassVar[float] = 7.0

    @property
    def critical_fixes(self) -> list[Fix]:
        return [f for f in self.required_fixes if f.severity == "critical"]

    @property
    def passed(self) -> bool:
        """Mean of the three dimensions >= 7 AND no critical fixes."""
        return self.scores.mean >= self.PASS_MEAN and not self.critical_fixes


class AgentState(TypedDict):
    """The state object LangGraph threads through every node.

    `critique` is None until the critic has run; `revision_count` guards the
    writer<->critic loop so it is guaranteed to terminate.

    Note `cost_usd`: by default a node returning a key *overwrites* it, so if
    each node reported its own spend the last node would erase the rest. The
    `Annotated[..., operator.add]` marker is a LangGraph **reducer** — it tells
    the graph to combine the old and new values with `+` instead of replacing.
    Nodes therefore return their own cost and the graph sums it.
    """

    brief: str
    memory_context: str
    plan: list[Task]
    findings: list[Finding]
    draft: str
    critique: Critique | None
    revision_count: int
    cost_usd: Annotated[float, operator.add]
