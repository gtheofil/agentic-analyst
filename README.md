# agentic-analyst

A multi-agent system that turns a one-line business brief into a cited research report.

> **Status: Phase 1 of 7.** The pipeline is `planner → writer` and nothing is
> researched yet, so every claim is tagged `[unverified]`. Research tools, a
> critic loop, memory, tracing and an eval harness land in later phases. The
> full README follows in Phase 5.

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
| [src/agentic_analyst/graph.py](src/agentic_analyst/graph.py) | Which node runs, and in what order |
| [prompts/](prompts/) | System prompts, one markdown file per agent |
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
