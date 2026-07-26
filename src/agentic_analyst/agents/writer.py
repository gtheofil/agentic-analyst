"""Writer node: plan + findings in, cited markdown report out.

The writer may only cite evidence the researcher actually gathered. Findings
are numbered `F1…Fn` here rather than in `Finding` itself, because the ID is a
property of *this report* — a citation label the reader can look up — not of
the finding, which may be reused across runs once memory lands.

An unsupported claim has exactly two legal endings: it is dropped, or it is
attributed to a finding. There is no `[unverified]` any more; Phase 1 needed
one because there were no findings to cite, and keeping it would leave the
model a way to write whatever it liked and tag its way out.
"""

import logging
from typing import Any

from ..llm import RUN_METER, call
from ..memory.preferences import format_preferences, load_preferences
from ..settings import PROMPTS_DIR
from ..state import AgentState, Critique, Finding, Task

log = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "writer.md"


def _format_plan(plan: list[Task]) -> str:
    """Render the typed plan as the plain text the prompt expects."""
    lines = []
    for task in plan:
        line = f"{task.id}. {task.goal}"
        if task.depends_on:
            line += f" (depends on {task.depends_on})"
        lines.append(line)
    return "\n".join(lines)


def _format_findings(findings: list[Finding]) -> str:
    """Render findings as the citable evidence list, one `F<n>` per finding.

    The quote is included in full: the writer needs to see the evidence to
    judge what it does and does not support, and the Phase 4 hard-check will
    match the report's tags back to these same IDs.
    """
    if not findings:
        return "(none — the researcher found no usable evidence)"

    blocks = [
        f"[F{i}] (task {f.task_id}, confidence: {f.confidence})\n"
        f"  claim: {f.claim}\n"
        f'  quote: "{f.quote}"\n'
        f"  source: {f.source_url}"
        for i, f in enumerate(findings, 1)
    ]
    return "\n\n".join(blocks)


def _format_revision(draft: str, critique: Critique) -> str:
    """Render the previous draft and the critic's fixes as a revision brief.

    Without this the writer is re-run on byte-identical inputs and can only
    re-roll the dice: it has no way to know what was wrong, so a "revision" is
    an expensive coin flip. Fixes are ordered critical-first because a critical
    one is a veto — no amount of polish elsewhere passes the draft while it
    stands.
    """
    order = {"critical": 0, "major": 1, "minor": 2}
    fixes = sorted(critique.required_fixes, key=lambda f: order[f.severity])

    lines = [
        "## Your previous draft",
        draft,
        "",
        "## Review outcome",
        (
            f"Scores — groundedness {critique.scores.groundedness}/10, "
            f"structure {critique.scores.structure}/10, "
            f"actionability {critique.scores.actionability}/10 "
            f"(mean {critique.scores.mean:.1f}; {Critique.PASS_MEAN:.0f} is needed to pass)."
        ),
        f"Weakest claim identified: {critique.weakest_claim}",
        "",
        "## Required fixes",
    ]
    if fixes:
        lines.extend(
            f"{i}. [{f.severity}] {f.location} — {f.issue}\n   Suggested: {f.suggestion}"
            for i, f in enumerate(fixes, 1)
        )
    else:
        lines.append("(none itemised — raise the weakest claim above to the evidence.)")
    return "\n".join(lines)


def writer(state: AgentState) -> dict[str, Any]:
    """Turn the plan and findings into a cited markdown draft.

    Runs in two modes off the same prompt. On the first pass `critique` is None
    and this writes from scratch. After a failed review the graph routes back
    here, and the previous draft plus the critic's required fixes are appended
    to the input — so a revision is a targeted edit rather than a re-roll.

    Reads:  state["brief"], state["memory_context"], state["plan"],
            state["findings"], state["critique"], state["draft"],
            state["revision_count"]
    Writes: state["draft"], state["revision_count"], state["cost_usd"]

    The counter is incremented *here*, on the pass that actually rewrites, so
    `revision_count` means "revisions performed". Incrementing it in the critic
    instead would count critiques — including the first one, which follows the
    original draft and revises nothing — and the loop would stop a revision
    early while reporting one more than it did.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    findings = state["findings"]
    critique = state["critique"]

    user_prompt = (
        f"Brief:\n{state['brief']}\n\n"
        f"Background:\n{state['memory_context']}\n\n"
        f"{format_preferences(load_preferences())}\n\n"
        f"Plan:\n{_format_plan(state['plan'])}\n\n"
        f"Findings:\n{_format_findings(findings)}\n"
    )

    revising = critique is not None
    if critique is not None:
        user_prompt += (
            "\n---\n"
            "You have written this report before and it did not pass review. "
            "Revise it: address every required fix below, keep what was already "
            "working, and obey the same citation rules. Return the full corrected "
            "report, not a diff or a summary of your changes.\n\n"
            f"{_format_revision(state['draft'], critique)}\n"
        )

    with RUN_METER.track() as spend:
        draft = call(tier="fast", system=system_prompt, user=user_prompt)

    revision_count = state["revision_count"] + (1 if revising else 0)

    log.info(
        "writer %s %d chars from %d findings (revision %d, cost $%.6f)",
        "revised to" if revising else "produced",
        len(draft),
        len(findings),
        revision_count,
        spend.usd,
    )

    # revision_count has no reducer, so returning it overwrites the old value
    # (unlike cost_usd, which the graph sums).
    return {"draft": draft, "revision_count": revision_count, "cost_usd": spend.usd}
