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
from ..settings import PROMPTS_DIR
from ..state import AgentState, Finding, Task

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


def writer(state: AgentState) -> dict[str, Any]:
    """Turn the plan and findings into a cited markdown draft.

    Reads:  state["brief"], state["memory_context"], state["plan"], state["findings"]
    Writes: state["draft"], state["cost_usd"]
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    findings = state["findings"]

    user_prompt = (
        f"Brief:\n{state['brief']}\n\n"
        f"Background:\n{state['memory_context']}\n\n"
        f"Plan:\n{_format_plan(state['plan'])}\n\n"
        f"Findings:\n{_format_findings(findings)}\n"
    )

    with RUN_METER.track() as spend:
        draft = call(tier="strong", system=system_prompt, user=user_prompt)

    log.info(
        "writer produced %d chars from %d findings (cost $%.6f)",
        len(draft),
        len(findings),
        spend.usd,
    )

    return {"draft": draft, "cost_usd": spend.usd}
