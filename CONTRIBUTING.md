# Contributing

Thanks for your interest in improving this project.

## Local setup

```bash
uv sync --dev
cp .env.example .env
```

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests evals scripts
uv run pytest
```

## Pull requests

- Keep changes focused and explain the reason for the change.
- Prefer a small, well-tested fix over a broad rewrite.
- Add or update tests when behavior changes.
