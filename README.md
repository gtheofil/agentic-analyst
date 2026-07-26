# agentic-analyst

A multi-agent system that turns a one-line business brief into a cited research report.

> **Status: Phase 3 of 7.** The pipeline is
> `load_memory → planner → researcher → writer ⇄ critic → write_memory`.
> Claims are cited to gathered evidence, a scored critic gates the draft and
> sends failures back for revision, and each run leaves a summary of itself
> behind for the next one. Tracing and an eval harness land in later phases.
> The full README follows in Phase 5.

## Quickstart

```bash
uv sync --dev                          # install
cp .env.example .env                   # then paste your Gemini key into .env
uv run python -m agentic_analyst.run "Assess energy options for an office building"
```

The report is written to `runs/<timestamp>/report.md`, and the total cost of
the run is printed at the end.

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
| [src/agentic_analyst/agents/](src/agentic_analyst/agents/) | One module per graph node |
| [src/agentic_analyst/memory/](src/agentic_analyst/memory/) | What survives a run: episodic (earned) and preferences (told) |
| [src/agentic_analyst/graph.py](src/agentic_analyst/graph.py) | Which node runs, and in what order |
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

**Memory is two different things.** `memory/episodic.py` is what the agent
earned — Chroma-backed summaries of its own past runs, retrieved by similarity
to the incoming brief and loaded *before* planning, so the plan benefits too.
`memory/preferences.py` is what a human told it, read from `config/prefs.yaml`.
Both terminal paths write memory: a run that failed review is the one most worth
remembering.
