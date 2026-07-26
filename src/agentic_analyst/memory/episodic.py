"""Episodic memory: the agent's diary of its own past runs.

After a run finishes we compress it into a 3-sentence summary (brief, surprises,
what worked) with a cheap Haiku-class call and store it in a persistent Chroma
collection. At the start of a new run, `load_memory` embeds the incoming brief
and retrieves the most similar past summaries, so the agent starts informed by
its own history instead of from zero.

Chroma's default embedder (all-MiniLM-L6-v2) is used here — good enough to show
the mechanism for a demo. A production system would swap in a stronger embedding
model; this is called out in the README.
"""

import logging
import uuid
from typing import Any

import chromadb
from pydantic import BaseModel, Field

from ..llm import RUN_METER, call
from ..settings import DATA_DIR
from ..state import AgentState

log = logging.getLogger(__name__)

_CHROMA_PATH = DATA_DIR / "chroma"
_COLLECTION = "run_history"
_TOP_K = 3

# PersistentClient writes to disk, so memories survive between runs — an
# in-memory client would forget everything on exit and defeat the point.
_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
_collection = _client.get_or_create_collection(_COLLECTION)


class RunSummary(BaseModel):
    """The 3-sentence episodic record the summariser must produce."""

    brief: str = Field(description="One sentence: what the run was asked to do.")
    surprises: str = Field(description="One sentence: what was unexpected or hard.")
    what_worked: str = Field(description="One sentence: a tactic worth reusing.")


_SUMMARY_SYSTEM = (
    "You compress a completed research run into exactly three sentences for a "
    "diary the agent reads before future runs. Sentence 1: the brief. "
    "Sentence 2: what was surprising or difficult. Sentence 3: what worked and "
    "is worth repeating. Be concrete and terse — this is advice to your future self."
)


def load_memory(state: AgentState) -> dict[str, Any]:
    """Start-of-run node: retrieve similar past runs into `memory_context`.

    Reads:  state["brief"]
    Writes: state["memory_context"]
    """
    count = _collection.count()
    if count == 0:
        log.info("episodic memory empty; starting cold")
        return {"memory_context": ""}

    # Chroma embeds query_texts with the default embedder and returns the
    # nearest stored summaries by meaning — not keyword overlap.
    results = _collection.query(
        query_texts=[state["brief"]],
        n_results=min(_TOP_K, count),
    )
    docs = results["documents"][0]  # first (and only) query's hits
    metas = results["metadatas"][0]

    if not docs:
        return {"memory_context": ""}

    lines = ["## Lessons from past runs"]
    for doc, meta in zip(docs, metas):
        tag = "passed" if meta.get("passed") else "failed review"
        lines.append(f"- ({tag}) {doc}")

    log.info("loaded %d past run(s) into memory_context", len(docs))
    return {"memory_context": "\n".join(lines)}


def write_memory(state: AgentState) -> dict[str, Any]:
    """End-of-run node: summarise the run and persist it to Chroma.

    Runs from *both* terminal paths and tags whether the draft passed, so a
    hard/failed brief becomes learnable experience rather than being lost.

    Reads:  state["brief"], state["draft"], state["critique"]
    Writes: state["cost_usd"]  (the summariser's spend)
    """
    critique = state["critique"]
    passed = critique.passed if critique is not None else False

    user_msg = (
        f"BRIEF:\n{state['brief']}\n\n"
        f"FINAL DRAFT:\n{state['draft']}\n\n"
        f"REVIEW OUTCOME: {'passed' if passed else 'failed after max revisions'}"
    )

    with RUN_METER.track() as spend:
        summary = call(
            tier="fast", 
            system=_SUMMARY_SYSTEM,
            user=user_msg,
            schema=RunSummary,
        )

    doc = (
        f"Brief: {summary.brief} "
        f"Surprises: {summary.surprises} "
        f"What worked: {summary.what_worked}"
    )

    _collection.add(
        documents=[doc],
        ids=[f"run-{uuid.uuid4().hex[:12]}"],  # unique id, survives deletions
        metadatas=[{"passed": passed}],
    )

    log.info("stored episodic memory (passed=%s, cost $%.6f)", passed, spend.usd)
    return {"cost_usd": spend.usd}