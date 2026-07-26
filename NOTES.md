# Build notes — failure modes and what they cost

Written while building, not afterwards. Each entry is a real failure with the
symptom I actually saw, what it turned out to be, and what I changed. The
pattern worth noticing across all of them: **the ones that hurt were invisible
to the tests I had.**

Reference run (Phase 2 complete, 26 Jul 2026): 3 tasks, 6 findings, 19 model
calls, $0.0222, ~90 seconds end to end, all nodes on `gemini-3.5-flash-lite`.
Roughly $0.004 of that is pacing-independent overhead; moving the writer to the
`strong` tier adds about half a cent.

Reference run (Phase 3 complete, 26 Jul 2026): 4 tasks, 8 findings, 30 model
calls, **$0.0850**, ~200 seconds. Nearly 4× Phase 2, and the breakdown says why:

| Node | Cost | Share |
|---|---|---|
| researcher (23 calls, `fast`) | $0.0314 | 37% |
| writer (1 call, `strong`) | $0.0294 | 35% |
| critic (1 call, `strong`) | $0.0143 | 17% |
| planner (1 call, `strong`) | $0.0095 | 11% |
| memory summariser (1 call, `fast`) | $0.0005 | 0.6% |

Two things worth internalising. **Two calls are 52% of the run** — the writer and
critic each process the entire draft on the strong tier, so the expensive part of
this system is no longer the part that does the most work. And **a revision costs
$0.044**, another writer plus another critic, so a run that uses both revisions
costs roughly $0.17. That is the number Phase 4.9's budget guard has to be sized
against, and it is a fifth of a dollar for one report.

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

## 8. The same test-stub hole as #6, in the very next node I added

**Symptom.** The suite took 107 seconds and one cost assertion failed by an odd
margin — 0.1024 against an expected 0.11. In the logs: `rate limited; the API
asked for 59s, waiting 60s`.

**Diagnosis.** The "fully mocked" suite was calling the real Gemini API. The
critic node had been added, `conftest.py` patched planner, researcher and writer
by name, and the critic reached straight through the fixture to a live client.
The stray `0.0024` was a real request's real cost landing in the total.

This is **#6 again, in the next node I wrote after writing #6 up**. I had
correctly diagnosed the class of bug — "a per-module mechanism that a new module
can silently opt out of by existing" — and then fixed only that instance of it.

**Fix.** The fixture no longer has a list. `modules_importing_call()` walks the
package and patches every module whose `call` is `llm.call`, so a module cannot
opt out by existing — it is found by the same walk that finds the others. Three
tests guard the walk itself: that it finds the known callers, that it is never
silently empty, and that a full graph run never constructs a real client.

**The interesting part.** Writing the lesson down did not prevent the repeat.
The note said what the bug *was*; what I needed was for the fix to be structural
so the next node couldn't reintroduce it. A retrospective is not a control. The
tell was there in plain sight — a "mocked" suite has no business taking 107
seconds — and I read that as slowness rather than as evidence.

---

## 9. The quality gate was scoring its own homework

**Symptom.** None. Everything passed, every run finished, the logs looked right.

**Diagnosis.** `Critique` had a `passed: bool` field, so `passed` was a value the
*model* returned about its own verdict, and `route_after_critic` branched on it
directly. A model that returned `score=3, passed=True` shipped the draft. The
critic prompt even ended with the pass rule — mean of the three dimensions ≥ 7,
no critical fixes — annotated *"the code enforces it"*. The code did not enforce
it, and could not have: `Critique` carried a single `score`, so there was no mean
to take, and `required_fixes` was a `list[str]`, so "no critical fixes" was
unaskable.

**Fix.** The schema now carries what the rule needs — `Scores` with three
dimensions, and `Fix` with a `severity` — and `passed` became a computed
`@property` over them. Deliberately a plain `@property` rather than a
`computed_field`, so it stays out of the JSON schema the model is asked to fill:
the model can still rate its own work 10/10, but it can no longer also say
"and therefore ship it". A test asserts `passed` is absent from
`Critique.model_json_schema()`, because the day it reappears is the day the gate
quietly stops being one.

**The interesting part.** This is theme #1 exactly inverted, in the one place it
matters most, and I wrote the inversion myself while believing the opposite —
the prompt says the code guarantees it, which is precisely how it escaped
review. A comment claiming an invariant is not an invariant. Worse, it actively
suppresses the question, because the next reader (me) sees the claim and stops
looking.

---

## 10. Every guardrail moves the failure — including into a node that can't act

**Symptom.** Forced-fail runs looped, revised, and finalised, exactly as the
mermaid said. The second draft was simply never better than the first.

**Diagnosis.** `route_after_critic` sent a failed draft back to `"writer"`, and
the graph comment described it as *"revision mode, consumes fixes"*. But
`writer()` built its prompt from brief, memory, plan and findings — it never read
`state["critique"]`. It received a byte-identical prompt and could only re-roll
the dice. The `required_fixes` the critic had paid a strong-tier call to produce
were written into state and read by nobody.

**Fix.** The writer takes a second mode: when a critique is present it appends
the previous draft, the three scores, and the fixes ordered critical-first. And
`revision_count` moved from the critic to the writer, because the critic was
counting *critiques issued* — the first of which follows the original draft and
revises nothing — so `MAX_REVISIONS = 2` bought exactly one rewrite while the
caveats section reported two.

**The interesting part.** Both halves of this were **routing that looked right
in the diagram**. The mermaid has an arrow from critic back to writer, the graph
has that edge, the logs print the hop — and the loop was decorative. A dataflow
bug hides perfectly behind correct control flow, because every observable
artefact of the run is exactly what you expected to see. The counter is the same
shape: an off-by-one in something *named* `revision_count` is invisible until you
ask what one revision is.

---

## 11. The dashboard said the run was free

**Symptom.** First fully traced run. The trace was perfect: 33 generations, all
8 graph nodes nested under one root, every prompt and completion captured,
token counts exactly matching the terminal. And the cost read **$0.00** — on
the trace, and on every single generation — while the CLI printed $0.047414.

**Diagnosis.** Langfuse takes `usage_details` and `cost_details` as
dictionaries of mutually exclusive buckets. For **usage** it derives the total
by summing the buckets; `{"input": 1449, "output": 40, "thinking": 0}` came
back with `total: 1489` filled in. For **cost** it does not. No error, no
warning, no hint in the response — the buckets were stored verbatim and
`calculatedTotalCost` stayed at zero.

Confirmed with a two-generation probe rather than another paid run: identical
payloads, one with an explicit `total`, one without.

```
A-no-total     calcTotalCost=0          costDetails={input, output, thinking}
B-with-total   calcTotalCost=0.000675   costDetails={input, output, thinking, total}
```

**The fix** sends `total` explicitly, and sends the value the meter just booked
rather than recomputing it. The dashboard total and the terminal total are now
the same number by construction: a second run reconciles at $0.056164 across 37
calls on both sides.

**The interesting part.** Two things, and the second one nearly got me.

First, the failure mode. Nothing threw. The instrumentation I would have
checked — is the trace there, are the nodes nested, are the tokens right — was
all green. The one number the whole exercise was *for* was silently zero, and
the only way to notice was to compare it against a number I already had.
Tracing that reports nothing looks identical to tracing that reports nothing
wrong.

Second: when I first read the fixed run back through the API, it showed 33 of
37 generations, 2 of 8 nodes, and an empty trace name. That looks exactly like
dropped spans — a real, plausible bug in my own flushing. It was ingestion lag.
Polling until the observation count stopped changing turned "my spans are being
dropped" into "I read too early." **Verifying against an eventually-consistent
system too quickly manufactures bugs that do not exist**, and I would have
happily spent an hour fixing that one.

---

## Recurring themes (the short version for an interview)

1. **Prompts are requests; code is guarantees.** Anything that must be true —
   termination, budgets, no duplicate evidence, `task_id` attribution — lives in
   the loop, not the prompt. I tried the prompt first and have the failed
   attempt to show for it. The sharpest version is #9: a *comment* claiming the
   code enforced a rule, above code that didn't.
2. **Mocking a boundary stops you testing the boundary.** The most expensive bug
   was invisible to a full green suite.
3. **Every guardrail moves the failure.** Fixing "one finding per task" created
   "the same finding twice". Read the outputs, not just the counts.
4. **Fail into a usable state.** Tool errors become observations; one task's
   model failure ends that task, not the report; a crashed run still prints what
   it spent.
5. **Pace, don't retry.** Rate limits are predictable and cheaper to avoid than
   to recover from.
6. **Never let a component grade itself.** The critic's verdict is the model's;
   whether that verdict passes is the graph's. Anything the system branches on
   has to be computed from the model's output, never returned by it.
7. **Correct control flow hides broken dataflow.** #10 routed perfectly and
   revised nothing. When a loop runs the right number of times, check that the
   thing going round it is actually changing.
8. **A written-up lesson is not a control.** I documented #6 and reintroduced it
   in the next node I wrote. What stopped it recurring was deleting the
   hand-maintained list, not describing why it was dangerous.
9. **The absence of an error is not evidence of success.** #11's cost was zero
   with everything green. The check that caught it was comparing a new number
   against one I already trusted — which is only possible because the meter
   existed first.
