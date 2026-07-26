"""Critic node: draft + findings in, a Critique verdict out."""

import logging
from typing import Any

from ..llm import RUN_METER, call
from ..settings import PROMPTS_DIR
from ..state import AgentState, Critique, Finding

log = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "critic.md"


def _render_sources(findings: list[Finding]) -> str:
    """Flatten the findings into a numbered evidence block for the critic.

    Findings have no id of their own, so we number them [1], [2], ... in order.
    Each line exposes the researcher's `claim`, the verbatim `quote` that backs
    it, the `source_url`, and the `confidence`, so the critic can check whether
    the draft's assertions actually match the evidence — and can be sceptical of
    anything leaning on a low-confidence finding.
    """
    if not findings:
        return "(no sources provided)"
    lines = []
    for i, f in enumerate(findings, start=1):
        lines.append(
            f"[{i}] claim: {f.claim}\n"
            f'    quote: "{f.quote}"\n'
            f"    source: {f.source_url} (confidence: {f.confidence})"
        )
    return "\n".join(lines)


def critic(state: AgentState) -> dict[str, Any]:
    """Judge the current draft against the findings that backed it.

    Reads:  state["draft"], state["findings"], state["revision_count"]
    Writes: state["critique"], state["revision_count"], state["cost_usd"]
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    # The critic needs both the sources and the draft in one user message so it
    # can judge groundedness by aligning the draft's claims to the findings.
    user_msg = (
        "## Sources\n"
        f"{_render_sources(state['findings'])}\n\n"
        "## Draft\n"
        f"{state['draft']}"
    )

    with RUN_METER.track() as spend:
        # schema=Critique makes `call` return a validated Critique, so no
        # isinstance check is needed to satisfy the type checker.
        result = call(
            tier="strong",
            system=system_prompt,
            user=user_msg,
            schema=Critique,
        )

    log.info(
        "critic scored %d/10 (passed=%s, %d fixes, cost $%.6f)",
        result.score,
        result.passed,
        len(result.required_fixes),
        spend.usd,
    )
    log.info("  weakest claim: %s", result.weakest_claim)

    # Increment here so the writer<->critic loop has a single, reliable counter
    # to terminate on. revision_count has no reducer, so returning the new value
    # overwrites the old one (unlike cost_usd, which is summed).
    return {
        "critique": result,
        "revision_count": state["revision_count"] + 1,
        "cost_usd": spend.usd,
    }