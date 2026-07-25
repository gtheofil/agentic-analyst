"""Writer node: plan in, markdown report out.

Phase 1 has no researcher yet, so there are no findings to cite. The writer
still emits citation tags — every claim gets `[unverified]` — so the citation
convention and the Saturday hard-check regex are settled now rather than
retrofitted later.
"""

import logging
from typing import Any

from ..llm import RUN_METER, call
from ..settings import PROMPTS_DIR
from ..state import AgentState, Task

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


def writer(state: AgentState) -> dict[str, Any]:
    """Turn the plan into a markdown draft.

    Reads:  state["brief"], state["memory_context"], state["plan"]
    Writes: state["draft"], state["cost_usd"]
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    user_prompt = (
        f"Brief:\n{state['brief']}\n\n"
        f"Background:\n{state['memory_context']}\n\n"
        f"Plan:\n{_format_plan(state['plan'])}\n"
    )

    with RUN_METER.track() as spend:
        draft = call(tier="fast", system=system_prompt, user=user_prompt)

    log.info("writer produced %d chars (cost $%.6f)", len(draft), spend.usd)

    return {"draft": draft, "cost_usd": spend.usd}
