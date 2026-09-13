# Project documentation

This folder provides a quick overview for reviewers and collaborators.

## Contents

- [Architecture overview](#architecture-overview)
- [Demo workflow](#demo-workflow)
- [Portfolio positioning](#portfolio-positioning)
- [Operational notes](#operational-notes)

## Architecture overview

The application follows a typed graph pattern. Each stage has explicit inputs,
outputs, and validation rules, which makes the system easier to reason about than
an unstructured agent loop.

The core stages are:

1. planner
2. researcher
3. writer
4. critic
5. hard-check
6. run persistence

## Demo workflow

For a quick product demo:

```bash
uv sync --dev
cp .env.example .env
uv run analyst run "Assess energy options for an office building"
```

Then inspect the generated report in `runs/<timestamp>/report.md`.

## Portfolio positioning

This repo is best presented as an applied systems project in multi-agent research,
not as a generic chatbot. The value is in the structure: evidence tracking,
review loops, budget control, and output validation.

## Operational notes

- The CLI is the primary local interface.
- The API is good for demonstration and integration work.
- The project is intentionally clear about its demo-level runtime assumptions.
- It is designed to be explainable to technical reviewers.
