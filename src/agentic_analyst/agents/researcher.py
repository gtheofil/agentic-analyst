from typing import Literal

from pydantic import BaseModel, Field

from agentic_analyst.state import Task, Finding
from agentic_analyst.tools.search import search, SearchArgs
from agentic_analyst.tools.fetch import fetch, FetchArgs
from agentic_analyst.tools.calc import calc
from agentic_analyst.llm import call

SYSTEM_PROMPT = "..."  # instructions that force verbatim quotes etc.


# --- the actions the model may choose from -------------------------------
class Search(BaseModel):
    action: Literal["search"]
    query: str


class Fetch(BaseModel):
    action: Literal["fetch"]
    url: str


class Calc(BaseModel):
    action: Literal["calc"]
    expression: str


class Record(BaseModel):  # your old record_finding
    action: Literal["record"]
    claim: str
    quote: str
    source_url: str
    confidence: Literal["high", "medium", "low"]


class Stop(BaseModel):
    action: Literal["stop"]
    reason: str


class NextStep(BaseModel):
    step: Search | Fetch | Calc | Record | Stop = Field(discriminator="action")


# --- ACT + OBSERVE: run one validated step, never crash ------------------
def run_step(step, task_id: int, findings: list[Finding]) -> str:
    try:
        if isinstance(step, Search):
            return str(search(SearchArgs(query=step.query)))

        if isinstance(step, Fetch):
            return str(fetch(FetchArgs(url=step.url)))

        if isinstance(step, Calc):
            return str(calc(step.expression))

        if isinstance(step, Record):
            finding = Finding(
                task_id=task_id,  # injected -- the model never supplies it
                claim=step.claim,
                quote=step.quote,
                source_url=step.source_url,
                confidence=step.confidence,
            )
            findings.append(finding)
            return f"recorded finding {len(findings)}/4"

        return f"unknown action: {step.action}"

    except Exception as e:  # error becomes an observation, loop continues
        return f"ERROR during {step.action}: {e}"


# --- the loop -----------------------------------------------------------
def researcher(task: Task) -> list[Finding]:
    findings: list[Finding] = []
    transcript = f"Task {task.id}: {task.goal}\n"

    tool_calls_used = 0
    while tool_calls_used < 6 and len(findings) < 4:

        # A. REASON -- ask the model for its next action, as structured output
        step = call(
            tier="fast",
            system=SYSTEM_PROMPT,
            user=transcript,
            schema=NextStep,
        ).step

        # B. DONE? -- the model chose to stop
        if isinstance(step, Stop):
            break

        # ACT + OBSERVE -- run it, append the result to the transcript
        observation = run_step(step, task.id, findings)
        transcript += f"\n[did {step.action}] result:\n{observation}\n"

        # Only search/fetch/calc count against the 6-tool-call budget;
        # Record is capped separately by the len(findings) < 4 guard.
        if isinstance(step, (Search, Fetch, Calc)):
            tool_calls_used += 1

    return findings