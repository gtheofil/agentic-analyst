"""Planner node: brief in, list of research tasks out."""

import logging
from typing import Any

from pydantic import BaseModel

from ..llm import RUN_METER, call
from ..settings import PROMPTS_DIR
from ..state import AgentState, Task

log = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "planner.md"


class Plan(BaseModel):
    """Wrapper so `llm.call` can return a structured list of tasks.

    Gemini's structured output needs an object at the top level, not a bare
    array, so the tasks are nested one level down.
    """

    tasks: list[Task]


def planner(state: AgentState) -> dict[str, Any]:
    """Decompose the brief in `state` into a list of research tasks.

    Reads:  state["brief"]
    Writes: state["plan"], state["cost_usd"]
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    with RUN_METER.track() as spend:
        # `call` is overloaded: passing schema=Plan makes the return type Plan,
        # so no isinstance check is needed to satisfy the type checker.
        result = call(
            tier="fast",
            system=system_prompt,
            user=state["brief"],
            schema=Plan,
        )

    log.info("planner produced %d tasks (cost $%.6f)", len(result.tasks), spend.usd)
    for task in result.tasks:
        log.info("  task %d: %s", task.id, task.goal)

    return {"plan": result.tasks, "cost_usd": spend.usd}
