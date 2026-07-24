from pathlib import Path

from ..llm import call
from ..state import AgentState

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "writer.md"


def writer(state: AgentState) -> dict[str, str]:
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    brief = state["brief"]
    memory_context = state["memory_context"]
    plan = state["plan"]

    # Turn list[Task] into plain text for the prompt
    plain_text = "\n".join(
        f"{task.id}.{task.goal}" + (f"(depends on {task.depends_on})" if task.depends_on else "")
        for task in plan
    )

    draft = call(
        tier="sonnet",
        system=system_prompt,
        user=(f"Brief:\n{brief}\n\nBackground:\n{memory_context}\n\nPlan:\n{plain_text}\n\n"),
    )
    assert isinstance(draft, str)
    return {"draft": draft}
