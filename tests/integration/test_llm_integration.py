"""Integration tests for llm.call() — hits the real Gemini API.

Run with: pytest -m integration
Skipped automatically if GOOGLE_API_KEY is not set.
"""

import os

import pytest
from pydantic import BaseModel

from agentic_analyst.llm import call

# ── Module-level skip if no API key ─────────────────────────
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
]


# ── Test 1: plain text mode (no schema) ─────────────────────
def test_call_returns_text_when_no_schema() -> None:
    """When schema is None, call() should return a plain string."""
    result = call(
        tier="fast",
        system="You are terse. Respond in exactly one word.",
        user="Say 'hello'.",
    )

    assert isinstance(result, str)
    assert len(result) > 0


# ── Test 2: structured mode (with schema) ───────────────────
class ColorPick(BaseModel):
    """Trivial schema — the model should always be able to produce this."""

    color: str
    reason: str


def test_call_returns_pydantic_instance_when_schema_given() -> None:
    """When a schema is passed, call() should return a validated instance."""
    result = call(
        tier="fast",
        system="You pick colours. Reply strictly in the requested JSON shape.",
        user="Pick a colour that suits a coffee shop. Give a one-sentence reason.",
        schema=ColorPick,
    )

    # Type checks
    assert isinstance(result, ColorPick)
    assert isinstance(result.color, str)
    assert isinstance(result.reason, str)

    # Shape checks (non-empty, sensible)
    assert result.color.strip(), "color should not be empty"
    assert result.reason.strip(), "reason should not be empty"
