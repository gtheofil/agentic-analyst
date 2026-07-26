"""Researcher: one task in, evidenced findings out.

A bounded ReAct loop. Each turn the model sees the transcript so far and picks
exactly one action; the result is appended and it is asked again. Three
properties are load-bearing and each is enforced here rather than in the
prompt, because a prompt is a request and a loop guard is a guarantee:

* **It terminates.** Six tool calls, four findings, and a hard step ceiling
  that catches the case where the model does neither (e.g. keeps emitting
  malformed `record` actions).
* **It never crashes the run.** A tool raising is caught and handed back to the
  model as an observation, so a dead link costs one step, not the whole report.
* **`task_id` is injected, never supplied.** The model is not trusted to
  attribute its own findings to the right task.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel

from ..llm import call
from ..settings import PROMPTS_DIR
from ..state import Finding, Task
from ..tools.calc import calc
from ..tools.fetch import FetchArgs, fetch
from ..tools.search import SearchArgs, SearchResult, search

log = logging.getLogger(__name__)

_PROMPT_PATH = PROMPTS_DIR / "researcher.md"

MAX_TOOL_CALLS = 6  # search/fetch/calc per task
MAX_FINDINGS = 4  # recorded findings per task
MIN_FINDINGS = 2  # below this, one early `stop` is pushed back (see below)
MAX_STEPS = 12  # model turns per task; the backstop for wasted steps
MAX_OBSERVATION_CHARS = 4_000  # a fetched page can be 1MB — the transcript can't


# ── The actions the model may choose from ───────────────────
# One class per action, so `run_step` dispatches on a type rather than picking
# apart a dict. Note the union below is deliberately NOT `Field(discriminator=
# "action")`: Gemini's structured output rejects the `oneOf` + `discriminator`
# JSON-Schema pair outright, and every call fails before it reaches the network.
# A plain union serialises as `anyOf`, which it accepts, and Pydantic still
# picks the right member off the `action` literal.


class Search(BaseModel):
    action: Literal["search"]
    query: str


class Fetch(BaseModel):
    action: Literal["fetch"]
    url: str


class Calc(BaseModel):
    action: Literal["calc"]
    expression: str


class Record(BaseModel):
    action: Literal["record"]
    claim: str
    quote: str
    source_url: str
    confidence: Literal["high", "medium", "low"]


class Stop(BaseModel):
    action: Literal["stop"]
    reason: str


Step = Search | Fetch | Calc | Record | Stop


class NextStep(BaseModel):
    """Wrapper so the model returns a JSON object rather than a bare union."""

    step: Step


# ── Rendering observations back to the model ────────────────


def _truncate(text: str) -> str:
    """Cap one observation so a long page can't crowd out the transcript."""
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return text[:MAX_OBSERVATION_CHARS] + f"\n… [truncated at {MAX_OBSERVATION_CHARS} chars]"


def _format_results(results: list[SearchResult]) -> str:
    """Search hits as numbered lines. `str(list_of_models)` wastes tokens on
    Pydantic reprs and reads badly to the model."""
    if not results:
        return "No results."
    return "\n".join(
        f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)
    )


def _same_quote(a: str, b: str) -> bool:
    """Whether two quotes are the same evidence, ignoring re-typing noise."""
    return re.sub(r"\s+", " ", a).strip().lower() == re.sub(r"\s+", " ", b).strip().lower()


# ── ACT + OBSERVE: run one validated step, never raise ──────


def run_step(step: Step, task_id: int, findings: list[Finding]) -> str:
    """Execute one action and return what the model should see as its result.

    Every failure path returns a string. The loop above treats an error as an
    observation and carries on, which is what lets the model route around a
    paywalled page instead of the run dying on it.
    """
    try:
        if isinstance(step, Search):
            return _format_results(search(SearchArgs(query=step.query)))

        if isinstance(step, Fetch):
            return _truncate(fetch(FetchArgs(url=step.url)).text)

        if isinstance(step, Calc):
            return str(calc(step.expression))

        if isinstance(step, Record):
            # Pushed to record a second finding, the model will happily file
            # the same quote twice under a reworded claim. Two findings citing
            # one sentence is one finding wearing a hat, and the writer would
            # cite it as two sources.
            if any(_same_quote(f.quote, step.quote) for f in findings):
                return (
                    "ERROR: that quote is already recorded for this task. Quote a different "
                    "sentence, fetch another source, or stop."
                )

            findings.append(
                Finding(
                    task_id=task_id,  # injected — the model never supplies it
                    claim=step.claim,
                    quote=step.quote,
                    source_url=step.source_url,
                    confidence=step.confidence,
                )
            )
            log.info("  finding %d/%d: %s", len(findings), MAX_FINDINGS, step.claim)
            return f"Recorded finding {len(findings)}/{MAX_FINDINGS}."

        return f"ERROR: unknown action {step.action!r}"

    except Exception as exc:
        # Deliberately broad: any tool failure is a fact about the world the
        # model should react to, not an exception the graph should propagate.
        log.info("  %s failed: %s", step.action, exc)
        return f"ERROR during {step.action}: {exc}"


# ── The loop ────────────────────────────────────────────────


def _budget_line(tool_calls_used: int, findings: int) -> str:
    return (
        f"\n[budget] tool calls left: {MAX_TOOL_CALLS - tool_calls_used} | "
        f"findings recorded: {findings}/{MAX_FINDINGS}\n"
    )


def researcher(task: Task) -> list[Finding]:
    """Work one task to exhaustion of its budget and return its findings.

    Returns whatever was gathered before stopping — possibly an empty list.
    A task that finds nothing is a normal outcome, and the writer is told to
    say so rather than invent coverage.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    findings: list[Finding] = []
    transcript = f"Task {task.id}: {task.goal}\n"
    tool_calls_used = 0
    pushed_back = False
    stopped = "step ceiling"

    for _ in range(MAX_STEPS):
        if tool_calls_used >= MAX_TOOL_CALLS:
            stopped = "tool budget"
            break
        if len(findings) >= MAX_FINDINGS:
            stopped = "finding cap"
            break

        # REASON — ask for the next action as structured output.
        try:
            step = call(
                tier="fast",
                system=system_prompt,
                user=transcript + _budget_line(tool_calls_used, len(findings)),
                schema=NextStep,
            ).step
        except (ValueError, RuntimeError) as exc:
            # The model gave up producing valid JSON, or returned nothing.
            # One task failing must not take the whole report down with it.
            log.warning("task %d: model call failed, ending task (%s)", task.id, exc)
            stopped = "model failure"
            break

        if isinstance(step, Stop):
            # Left to itself the model declares victory on one finding — asking
            # it not to (the "at least two" line in the prompt) does not hold.
            # So the first premature stop is refused, once, and only while
            # there is budget to act on the refusal. Once means the loop still
            # terminates and a task with genuinely thin evidence still ends.
            too_thin = len(findings) < MIN_FINDINGS
            can_still_act = tool_calls_used < MAX_TOOL_CALLS
            if too_thin and can_still_act and not pushed_back:
                pushed_back = True
                transcript += (
                    f"\n[stop refused] You have recorded {len(findings)} finding(s); this task "
                    f"needs at least {MIN_FINDINGS}. Record another supported claim — the page "
                    "you already fetched will usually carry one — or fetch a second source. "
                    "If the evidence really is not there, stop again and say what is missing.\n"
                )
                continue
            stopped = f"model stopped: {step.reason}"
            break

        # ACT + OBSERVE.
        observation = run_step(step, task.id, findings)
        transcript += f"\n[did {step.action}] result:\n{observation}\n"

        # Only real tool calls draw down the budget; `record` is capped
        # separately by MAX_FINDINGS, and MAX_STEPS catches everything else.
        if isinstance(step, Search | Fetch | Calc):
            tool_calls_used += 1

    log.info(
        "task %d: %d findings in %d tool calls (%s)",
        task.id,
        len(findings),
        tool_calls_used,
        stopped,
    )
    return findings
