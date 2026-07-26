# Build notes — failure modes and what they cost

Written while building, not afterwards. Each entry is a real failure with the
symptom I actually saw, what it turned out to be, and what I changed. The
pattern worth noticing across all of them: **the ones that hurt were invisible
to the tests I had.**

Reference run (Phase 2 complete, 26 Jul 2026): 3 tasks, 6 findings, 19 model
calls, $0.0222, ~90 seconds end to end, all nodes on `gemini-3.5-flash-lite`.
Roughly $0.004 of that is pacing-independent overhead; moving the writer to the
`strong` tier adds about half a cent.

---

## 1. A schema the model provider silently could not accept

**Symptom.** The researcher had never run. Every call died inside the Google
SDK before a request was sent, with a Pydantic validation error about
`extra_forbidden` on keys I had never written.

**Diagnosis.** I modelled the researcher's next action as a Pydantic
*discriminated* union — the idiomatic way to say "exactly one of these five
shapes, keyed on `action`". Pydantic serialises that to JSON Schema as `oneOf`
plus `discriminator`. Gemini's structured-output converter accepts neither key
and rejects the whole schema. A plain (undiscriminated) union serialises as
`anyOf`, which it does accept.

**Fix.** Dropped `Field(discriminator="action")`. The type is otherwise
unchanged, and Pydantic still resolves the right member off the `action`
literal, so nothing downstream moved.

**The interesting part.** No mocked test could have caught this, because the
schema conversion happens *inside* the SDK call that mocks replace. The
regression test therefore asserts against the SDK's converter directly
(`tests/unit/test_researcher.py::test_next_step_schema_is_accepted_by_gemini`).
Generalising: when you mock a boundary, you stop testing the boundary. Anything
the real dependency validates needs its own test, or you find out in
production.

---

## 2. The agent declared victory after one finding — and the prompt could not stop it

**Symptom.** Live runs came back with exactly one finding per task, every time,
with a confident `stop` reason claiming the task was fully answered.

**Diagnosis.** Nothing was broken. The model genuinely believed one figure
answered a two-part question. I first tried the obvious fix — "record at least
two findings" in the system prompt, with reasoning about why. It changed
nothing on `flash-lite`.

**Fix.** Made it a loop guard instead of a request: the first premature `stop`
below two findings is refused and pushed back into the transcript, once, and
only while tool budget remains. The second `stop` is always obeyed.

**The interesting part.** This is the design rule the whole system runs on — a
prompt is a request, a loop guard is a guarantee. Anything you *need* to be
true belongs in code. The "once" matters as much as the push-back: refusing
indefinitely would have traded a quality bug for an unbounded loop.

---

## 3. Fixing that created a subtler failure: the same evidence, twice

**Symptom.** Two findings per task, as intended — but both quoting the *same
sentence* from the same page, under two reworded claims.

**Diagnosis.** I had asked for a number and got a number. Pushed to produce a
second finding with no new evidence, the cheapest thing the model can do is
re-file the evidence it already has. The writer would then have cited one
sentence as two independent sources, which is worse than one honest finding —
it manufactures the appearance of corroboration.

**Fix.** `run_step` rejects a `record` whose quote (whitespace- and
case-normalised) is already recorded for that task, returning an error
observation so the model looks for different evidence or stops.

**The interesting part.** Every guardrail moves the failure rather than
removing it. I only saw this one because I was reading the actual findings, not
the count. Metrics would have shown a clean improvement from 1 to 2 findings
per task.

---

## 4. Errors as observations, not exceptions

**Symptom.** A researcher task fetched an IRS PDF; `trafilatura` returned an
empty tree. Another hit a page over the 1MB size cap.

**Diagnosis.** Not a bug — this is the normal weather of the open web. The
question is only what the loop does with it.

**Fix (by design, and it held).** Every tool failure inside `run_step` is caught
and returned to the model as an `ERROR: …` string. In both live cases the model
read the error, chose a different source, and completed the task. A dead link
costs one step out of six, not the run.

**The interesting part.** The tempting alternative — retry the tool — is wrong
here. The model has more context than a retry policy does: it knows the page
was a PDF and that another result in the search list will do.

---

## 5. A free-tier rate limit killed a run that had already been paid for

**Symptom.** Twenty calls into a four-task run, a `429 RESOURCE_EXHAUSTED`
propagated through LangGraph as a 60-line traceback. Everything the run had
produced was lost; the ~$0.02 already spent was not.

**Diagnosis.** Two separate mistakes.
1. The retry policy treated all `4xx` as permanent. That is right for 400 and
   403, but a 429 says *not now*, not *never* — and the response even carries a
   `RetryInfo` saying exactly when the window reopens.
2. Nothing paced the calls. The free tier allows 15 requests/minute; a run makes
   about 20 and a four-task run about 30. The system was discovering the quota
   by breaching it, at whatever moment it happened to breach it.

**Fix.** Retry on 429 specifically, waiting the duration the API asked for
rather than a guessed backoff; plus a client-side throttle (default 12
requests/minute, configurable, off for paid keys) so the limit is normally
never reached. Terminal provider failures now surface as `LLMError`, a
`RuntimeError` subclass, so agents catch failures without importing anything
Gemini-specific and the CLI prints one line of advice plus the spend so far.

**The interesting part.** Pacing beats retrying. A retry pays for the failed
call, the wait, *and* the retry; spacing calls costs only the wait. And the
provider-specific exception never escaping `llm.py` is what keeps the "swap the
provider by editing one file" claim true.

---

## 6. A test stub that quietly stopped covering the graph

**Symptom.** Four pipeline tests failing after the researcher node was added —
and worse, failing by *reaching for a real API client*.

**Diagnosis.** The `stub_llm` fixture patched `call` in the planner and writer
modules by name. Adding a third node that calls the model left a hole the
fixture knew nothing about. In the same change, the researcher's spend was
never added to `cost_usd`, so a run under-reported its own cost by roughly 90%
— the researcher is 19 of every 21 calls.

**Fix.** The fixture patches every agent module and dispatches on the requested
schema; the researcher node wraps its task loop in `RUN_METER.track()`; the
cost test now asserts against the number of calls actually made rather than a
hardcoded 2.

**The interesting part.** Both failures are the same shape: a per-module
mechanism (patch this name, report this cost) that a new module can silently
opt out of by existing. Cost metering in particular has to be collected at a
choke point — `llm.call` — and only *attributed* per node, or every new node is
a new place to forget.

---

## 7. Cost that was correct and a log line that was not

**Symptom (Phase 1).** Token counts in the logs did not match the money.

**Diagnosis.** Gemini reports thinking tokens separately from visible output
tokens, but bills both at the output rate. The cost calculation included them;
the log line printed `candidates_token_count` alone, under-reporting output by
~68% on a thinking model.

**Fix.** One `_billed_output_tokens()` helper that both the meter and the log
line go through, with a regression test asserting the logged number equals the
billed one.

**The interesting part.** Nothing failed. Two numbers that should have been the
same drifted apart and the system was perfectly happy. Anything computed twice
will eventually disagree — so compute it once.

---

## Recurring themes (the short version for an interview)

1. **Prompts are requests; code is guarantees.** Anything that must be true —
   termination, budgets, no duplicate evidence, `task_id` attribution — lives in
   the loop, not the prompt. I tried the prompt first and have the failed
   attempt to show for it.
2. **Mocking a boundary stops you testing the boundary.** The most expensive bug
   was invisible to a full green suite.
3. **Every guardrail moves the failure.** Fixing "one finding per task" created
   "the same finding twice". Read the outputs, not just the counts.
4. **Fail into a usable state.** Tool errors become observations; one task's
   model failure ends that task, not the report; a crashed run still prints what
   it spent.
5. **Pace, don't retry.** Rate limits are predictable and cheaper to avoid than
   to recover from.
