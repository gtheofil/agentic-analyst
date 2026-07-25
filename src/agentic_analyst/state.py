"""Typed contracts that flow between agents.

This is the load-bearing design decision of the whole system: every agent's
output is one of these Pydantic models, so a downstream agent can never receive
malformed input. A boundary is schema-validated, and a parse failure becomes a
retry rather than a crash three nodes later.

`AgentState` is the single object LangGraph threads through the entire graph.
Each node reads some of its fields and writes others.
"""

import operator
from typing import Annotated, Literal, TypedDict

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


class Critique(BaseModel):
    """The critic's verdict on a draft.

    `weakest_claim` is mandatory on purpose: forcing the critic to name the
    single weakest claim stops it from rubber-stamping drafts.
    """

    score: int = Field(ge=1, le=10)  # anchored 1-10 rubric
    passed: bool
    weakest_claim: str
    required_fixes: list[str] = Field(default_factory=list)


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
