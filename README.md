# agentic-analyst

A multi-agent system that turns a one-line business brief into a cited research report.

> **Status: Phase 4b of 7.** The pipeline is
> `load_memory → planner → researcher → writer ⇄ critic → write_memory`.
> Claims are cited to gathered evidence, a scored critic gates the draft and
> sends failures back for revision, and each run leaves a summary of itself
> behind for the next one. Every run emits one nested Langfuse trace — each
> node, each tool call, and each model call with its tokens and cost. Reports
> are graded two ways: a deterministic hard-check that verifies citations and
> live sources, and an anchored LLM judge over a 10-brief golden set. There is
> an HTTP API and a CLI. The full README follows in Phase 5.

## Quickstart

```bash
uv sync --dev                          # install
cp .env.example .env                   # then paste your Gemini key into .env
uv run analyst run "Assess energy options for an office building"
```

The report is written to `runs/<timestamp>/report.md`, and the total cost of
the run is printed at the end.

```bash
uv run analyst runs               # what has been run
uv run analyst show latest -r     # a past run, with its report
uv run analyst check latest       # hard-check it: citations, sources, cost
uv run analyst serve              # the HTTP API on :8000
```

## The API

A run takes two to four minutes, which is longer than anything between the
client and the server will hold a connection open. So submission and collection
are separate requests:

```bash
curl -X POST localhost:8000/briefs -H 'content-type: application/json' \
  -d '{"brief": "Assess energy options for an office building"}'
# {"run_id": "20260726T165538Z", "status": "accepted", "poll": "/runs/20260726T165538Z"}

curl localhost:8000/runs/20260726T165538Z
# {"run": {"status": "running", ...}, "report": null}      ... then, once finished:
# {"run": {"status": "succeeded", "cost_usd": 0.0871, ...}, "report": "# Energy..."}
```

`runs/` is the registry — there is no database, and `GET /runs/{id}` is a
directory read. Deliberate for a demo, and named as a limitation rather than
hidden: no cross-run queries, no concurrent writers, no retention policy.

## Evals

Two graders, and they are not the same kind of thing:

```bash
uv run python scripts/hardcheck.py runs/<id>      # facts: exit 0 or 1
uv run python -m evals.run_evals --smoke          # quality: 3 briefs
uv run python -m evals.run_evals                  # the full golden set
```

**The hard-check verifies claims about reality and cannot be wrong.** Every
`[F<id>]` in the report resolves to a finding that was actually gathered, every
source URL still answers, no section is empty, the run stayed inside its cost
cap. It is deterministic, it costs nothing, and its exit code gates CI.

**The judge scores quality and can be wrong.** Three anchored dimensions —
coverage, groundedness, actionability — one model call each, because asking for
three scores in one response gets three numbers that move together. Coverage is
graded against `must_cover` points the pipeline never sees; groundedness is
graded against the findings, with the same numbering the writer used. Its scores
are reported, not merged, and gate nothing unless you pass `--min-pass-rate`.

Results land in [evals/results/summary.md](evals/results/summary.md) — the
aggregate table plus every judge justification, which is what makes the scores
spot-checkable rather than decorative.

Tracing is optional. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to
`.env` (free project at [cloud.langfuse.com](https://cloud.langfuse.com)) and
each run prints a trace URL alongside its cost. Leave them out and the run
behaves identically, just unobserved.

No API key? Everything except a live run still works — the package imports,
lints, type-checks and passes its full mocked test suite with no key present:

```bash
uv run ruff check . && uv run mypy src tests && uv run pytest
```

## Layout

| Path | What lives there |
| --- | --- |
| [src/agentic_analyst/state.py](src/agentic_analyst/state.py) | The typed contracts every agent reads and writes |
| [src/agentic_analyst/llm.py](src/agentic_analyst/llm.py) | The single door to the model: retries, cost metering, structured output |
| [src/agentic_analyst/settings.py](src/agentic_analyst/settings.py) | Config, model tiers, prices, paths |
| [src/agentic_analyst/observability.py](src/agentic_analyst/observability.py) | The only module that knows Langfuse exists |
| [src/agentic_analyst/agents/](src/agentic_analyst/agents/) | One module per graph node |
| [src/agentic_analyst/memory/](src/agentic_analyst/memory/) | What survives a run: episodic (earned) and preferences (told) |
| [src/agentic_analyst/graph.py](src/agentic_analyst/graph.py) | Which node runs, and in what order |
| [src/agentic_analyst/runner.py](src/agentic_analyst/runner.py) | One run start to finish, and the `runs/` registry it writes |
| [src/agentic_analyst/api.py](src/agentic_analyst/api.py) | `POST /briefs`, `GET /runs/{id}` |
| [src/agentic_analyst/cli.py](src/agentic_analyst/cli.py) | `analyst run \| show \| runs \| check \| serve` |
| [src/agentic_analyst/hardcheck.py](src/agentic_analyst/hardcheck.py) | The deterministic checks. No model involved |
| [evals/golden/](evals/golden/) | 10 briefs with the points a good answer cannot omit |
| [evals/judge.py](evals/judge.py) | LLM-as-judge: three anchored dimensions, one call each |
| [prompts/](prompts/) | System prompts, one markdown file per agent |
| [config/prefs.yaml](config/prefs.yaml) | Writer tone and length, editable without touching code |
| [tests/unit/](tests/unit/) | Fast, fully mocked. Run by default |
| [tests/integration/](tests/integration/) | Hits the real API. Run with `uv run pytest -m integration` |

## Design notes

**Every boundary is schema-validated.** Agents exchange Pydantic models, not
free text, so a malformed hand-off fails at the boundary that produced it
rather than three nodes downstream.

**Two kinds of retry, kept apart.** A 5xx or a timeout is retried verbatim by
`tenacity`. A schema violation is *not* — the same request would fail the same
way — so the model is re-prompted once with its own broken output attached.

**Agents ask for a capability, not a model.** `call(tier="strong", ...)`
resolves through `MODEL_TIERS`, so changing provider is a two-line edit in
`settings.py`.

**The critic does not decide whether it passed.** It returns three scored
dimensions and a list of fixes carrying a severity. Whether that constitutes a
pass — mean ≥ 7, no `critical` fix — is computed by the graph from those
numbers, never read off the model's own response. A model that likes its work
can score it 10/10; it cannot also vote to ship it.

**The revision loop is a loop, not a re-roll.** A failed draft goes back to the
writer *with* the previous draft and the critic's fixes attached, ordered
critical-first. `revision_count` is incremented by the writer, on the pass that
actually rewrites, so the number means revisions performed. Two is the ceiling,
after which the report ships with an appended `## Unresolved review issues`
section rather than silently or never.

**Tracing and cost metering are not the same feature.** Langfuse reports what a
run cost after it finished; `RUN_METER` tells the running program what it has
spent so far, which is what a budget guard has to branch on. They read the same
`_cost_breakdown`, so the dashboard and the terminal cannot drift — one live run
reconciles at $0.056164 across 37 calls on both. Tracing is deliberately
optional and fails soft: with no keys `get_tracer()` returns a disabled client
that silently discards spans, so no call site anywhere contains `if tracing:`.

**The budget guard is two layers, because the honest one is ugly.** A cap check
sits inside `llm.call`, so no node can spend past it by existing — that is the
guarantee, and it raises. On its own it would mean a run dying mid-writer with
the money already spent, so the graph *also* asks whether it is over budget
before each of the two expensive hand-offs and routes to a terminal node that
finishes the report without another model call. The guarantee is the exception;
the routing is what makes hitting it survivable.

**Memory is two different things.** `memory/episodic.py` is what the agent
earned — Chroma-backed summaries of its own past runs, retrieved by similarity
to the incoming brief and loaded *before* planning, so the plan benefits too.
`memory/preferences.py` is what a human told it, read from `config/prefs.yaml`.
Both terminal paths write memory: a run that failed review is the one most worth
remembering.
