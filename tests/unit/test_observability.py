"""Unit tests for tracing.

Nothing here reaches Langfuse: the autouse `no_tracing` fixture in conftest.py
strips the keys, so `get_tracer()` returns its null-object client and every
span is discarded. That is deliberate — the instrumentation still executes, it
just has nowhere to send anything.
"""

from unittest.mock import MagicMock

import pytest

import agentic_analyst.llm as llm
import agentic_analyst.observability as obs
from agentic_analyst.llm import _cost_breakdown, _usage_breakdown
from agentic_analyst.settings import PRICES, SETTINGS


def _usage(prompt: int, output: int, thoughts: int = 0) -> MagicMock:
    """A stand-in for the SDK's usage metadata object."""
    return MagicMock(
        prompt_token_count=prompt,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
    )


MODEL = "gemini-3.6-flash"


# ════════════════════════════════════════════════════════════════════
# 1. Tracing is optional, and half-configured is not configured
# ════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("public", "secret", "expected"),
    [
        ("pk-lf-x", "sk-lf-x", True),
        (None, None, False),
        ("pk-lf-x", None, False),  # one key alone is a typo, not a config
        (None, "sk-lf-x", False),
    ],
)
def test_tracing_needs_both_keys(
    monkeypatch: pytest.MonkeyPatch, public: str | None, secret: str | None, expected: bool
) -> None:
    monkeypatch.setattr(SETTINGS, "langfuse_public_key", public)
    monkeypatch.setattr(SETTINGS, "langfuse_secret_key", secret)
    assert SETTINGS.tracing_enabled is expected


def test_tracer_is_a_null_object_not_none() -> None:
    """The disabled path must have the same shape as the enabled one.

    A `Langfuse | None` return type would put a branch at every call site, and
    the one place someone forgot it would be an AttributeError mid-run.
    """
    tracer = obs.get_tracer()
    assert tracer is not None

    # Spans still open and close with no keys; they are simply discarded.
    with tracer.start_as_current_observation(as_type="span", name="probe") as span:
        span.update(output="discarded")

    obs.flush()  # must not raise


def test_callback_handler_constructs_without_keys() -> None:
    """A fresh clone with no credentials must still be able to build the graph."""
    assert obs.callback_handler() is not None


def test_tracer_is_cached() -> None:
    """One client per process — reconnecting per call would be absurd."""
    assert obs.get_tracer() is obs.get_tracer()


# ════════════════════════════════════════════════════════════════════
# 2. Token buckets
# ════════════════════════════════════════════════════════════════════


def test_usage_buckets_are_mutually_exclusive_and_complete() -> None:
    """Every token counted exactly once, so the parts sum to the whole.

    Langfuse derives the total by summing the buckets. Double-counting thinking
    tokens under `output` would inflate the dashboard's totals over the real
    ones without any error appearing anywhere.
    """
    usage = _usage(prompt=100, output=50, thoughts=30)
    buckets = _usage_breakdown(usage)

    assert buckets == {"input": 100, "output": 50, "thinking": 30}
    assert sum(buckets.values()) == 180


def test_usage_breakdown_of_nothing_is_empty() -> None:
    assert _usage_breakdown(None) == {}
    assert _cost_breakdown(MODEL, None) == {}


def test_thinking_is_priced_at_the_output_rate() -> None:
    """Gemini reports thinking separately but bills it as output.

    Splitting the bucket is presentation; pricing it differently would be a
    bug. This pins the distinction down.
    """
    costs = _cost_breakdown(MODEL, _usage(prompt=0, output=1_000_000, thoughts=1_000_000))
    assert costs["thinking"] == pytest.approx(PRICES[MODEL]["output"])
    assert costs["thinking"] == pytest.approx(costs["output"])


# ════════════════════════════════════════════════════════════════════
# 3. The drift guard
# ════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "usage",
    [
        _usage(prompt=1000, output=500, thoughts=250),
        _usage(prompt=0, output=0, thoughts=0),
        _usage(prompt=7, output=3, thoughts=0),
        _usage(prompt=1_000_000, output=1_000_000, thoughts=1_000_000),
    ],
)
def test_traced_cost_equals_metered_cost(usage: MagicMock) -> None:
    """What the dashboard is told and what the terminal prints must agree.

    This is the regression test for the hazard `_billed_output_tokens` warns
    about: pricing the same call in two places is how the two numbers silently
    stop matching. `RUN_METER.record` and the tracer both read
    `_cost_breakdown`, and this asserts they cannot diverge.
    """
    metered = llm.RUN_METER.record(MODEL, usage)
    traced = sum(_cost_breakdown(MODEL, usage).values())

    assert traced == pytest.approx(metered)


def test_trace_receives_an_explicit_cost_total(
    mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost total must be sent, not left to be derived.

    Verified against the live API: Langfuse sums the *usage* buckets into a
    total but does not do the same for *cost*. A generation shipped without an
    explicit `total` reports $0.00, with no error and no warning — the one
    number tracing exists to provide, silently missing.
    """
    mock_client.models.generate_content.return_value = MagicMock(
        usage_metadata=_usage(prompt=1000, output=100, thoughts=50),
        text="hello",
    )
    generation = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_observation.return_value.__enter__.return_value = generation
    monkeypatch.setattr(llm, "get_tracer", lambda: tracer)

    llm._generate_and_meter(MODEL, "user", MagicMock(system_instruction="sys"), tier="fast")

    costs = generation.update.call_args.kwargs["cost_details"]
    buckets = {k: v for k, v in costs.items() if k != "total"}

    assert "total" in costs, "Langfuse will report $0.00 without this"
    assert costs["total"] == pytest.approx(sum(buckets.values()))
    # And it is the same number the terminal prints.
    assert costs["total"] == pytest.approx(llm.RUN_METER.total_cost_usd)


def test_metered_output_tokens_match_the_buckets() -> None:
    """`total_output_tokens` counts what is billed as output: visible + thinking."""
    usage = _usage(prompt=100, output=50, thoughts=30)
    llm.RUN_METER.record(MODEL, usage)

    buckets = _usage_breakdown(usage)
    assert llm.RUN_METER.total_output_tokens == buckets["output"] + buckets["thinking"]
    assert llm.RUN_METER.total_input_tokens == buckets["input"]
