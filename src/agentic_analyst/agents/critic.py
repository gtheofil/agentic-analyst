"""Critic node: draft + findings in, a Critique verdict out."""

import logging
from typing import Any

from ..llm import RUN_METER, call
from ..settings import PROMPTS_DIR
from ..state import AgentState, Critique, Finding

log = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "critic.md"


def _render_sources(findings: list[Finding]) -> str:
    """Flatten the findings into a labelled evidence block for the critic.

    Findings have no id of their own, so they are labelled `F1`, `F2`, ... in
    order — the *same* labels the writer used when it cited them. That match
    matters: the critic's main job is checking whether `[F3]` in the draft
    really is backed by F3, and it cannot do that if it sees the evidence under
    different names than the draft cites.

    Each entry exposes the researcher's `claim`, the verbatim `quote` that backs
    it, the `source_url`, and the `confidence`, so the critic can check whether
    the draft's assertions actually match the evidence — and can be sceptical of
    anything leaning on a low-confidence finding.
    """
    if not findings:
        return "(no sources provided — every factual claim in the draft is unsupported)"
    lines = []
    for i, f in enumerate(findings, start=1):
        lines.append(
            f"[F{i}] claim: {f.claim}\n"
            f'    quote: "{f.quote}"\n'
            f"    source: {f.source_url} (confidence: {f.confidence})"
        )
    return "\n".join(lines)


def critic(state: AgentState) -> dict[str, Any]:
    """Judge the current draft against the findings that backed it.

    Reads:  state["draft"], state["findings"]
    Writes: state["critique"], state["cost_usd"]

    Note it does *not* touch `revision_count`. The writer owns that counter,
    incrementing it when it actually performs a revision, so the number means
    "revisions done" rather than "critiques issued".
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    # The critic needs both the sources and the draft in one user message so it
    # can judge groundedness by aligning the draft's claims to the findings.
    user_msg = f"## Sources\n{_render_sources(state['findings'])}\n\n## Draft\n{state['draft']}"

    with RUN_METER.track() as spend:
        # schema=Critique makes `call` return a validated Critique, so no
        # isinstance check is needed to satisfy the type checker.
        result = call(
            tier="fast",
            system=system_prompt,
            user=user_msg,
            schema=Critique,
        )

    # `passed` is computed from the scores and severities, not read off the
    # model's response — see Critique in state.py.
    log.info(
        "critic scored g=%d s=%d a=%d (mean %.1f) -> passed=%s (cost $%.6f)",
        result.scores.groundedness,
        result.scores.structure,
        result.scores.actionability,
        result.scores.mean,
        result.passed,
        spend.usd,
    )
    for fix in result.required_fixes:
        log.info("  [%s] %s — %s", fix.severity, fix.location, fix.issue)
    log.info("  weakest claim: %s", result.weakest_claim)

    return {"critique": result, "cost_usd": spend.usd}
