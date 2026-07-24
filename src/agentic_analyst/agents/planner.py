from pathlib import Path

from pydantic import BaseModel

from ..llm import call
from ..state import AgentState, Task

# Walk up from src/agents/planner.py -> src/agents -> agentic_analyst
# -> src -> repo root -> prompts/
_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "planner.md"


class Plan(BaseModel):
    """Wrapper so `llm.call` can return a structured list of tasks."""

    tasks: list[Task]


def planner(state: AgentState) -> dict[str, list[Task]]:
    """Decompose the brief in `state` into a list of research tasks.

    Reads:  state["brief"]
    Writes: state["plan"]
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    result = call(
        tier="sonnet",
        system=system_prompt,
        user=state["brief"],
        schema=Plan,
    )

    # `call` returns `str | BaseModel`; with schema=Plan it will be a Plan.
    assert isinstance(result, Plan), "planner expected structured Plan output"

    return {"plan": result.tasks}
