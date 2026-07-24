from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

import agentic_analyst.llm as llm
from agentic_analyst.llm import call


class Toy(BaseModel):
    n: int


# ── Reset the shared RunMeter before every test ─────────────────────
# RUN_METER is module-level global state. Without this, cost/counters
# would leak between tests and make assertions unreliable.
@pytest.fixture(autouse=True)
def reset_meter():
    llm.RUN_METER.total_cost_usd = 0.0
    llm.RUN_METER.total_input_tokens = 0
    llm.RUN_METER.total_output_tokens = 0
    llm.RUN_METER.calls = 0
    yield


# ════════════════════════════════════════════════════════════════════
# 1. Schema retry-once  (bad JSON → re-prompt with error text)
# ════════════════════════════════════════════════════════════════════


@patch("agentic_analyst.llm.client")
def test_retry_once_then_succeeds(mock_client):
    """First response is unparseable, second is valid → returns the model."""
    bad = MagicMock(parsed=None, text="garbage", usage_metadata=None)
    good = MagicMock(parsed=Toy(n=42), usage_metadata=None)
    mock_client.models.generate_content.side_effect = [bad, good]

    result = call("haiku", "sys", "user", schema=Toy)

    assert result == Toy(n=42)
    assert mock_client.models.generate_content.call_count == 2


@patch("agentic_analyst.llm.client")
def test_retry_once_then_gives_up(mock_client):
    """Both responses unparseable → raises ValueError after exactly 2 attempts."""
    bad = MagicMock(parsed=None, text="still garbage", usage_metadata=None)
    mock_client.models.generate_content.side_effect = [bad, bad]

    with pytest.raises(ValueError, match="failed to produce valid Toy"):
        call("haiku", "sys", "user", schema=Toy)

    assert mock_client.models.generate_content.call_count == 2


# ════════════════════════════════════════════════════════════════════
# 2. Tenacity retry  (transient errors retried, client errors are not)
# ════════════════════════════════════════════════════════════════════


@patch("agentic_analyst.llm.client")
def test_tenacity_retries_on_server_error(mock_client):
    """ServerError (5xx) is transient → tenacity retries up to 3 attempts."""
    good = MagicMock(parsed=None, text="ok", usage_metadata=None)
    mock_client.models.generate_content.side_effect = [
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        good,
    ]

    result = call("haiku", "sys", "user")  # text path, no schema

    assert result == "ok"
    assert mock_client.models.generate_content.call_count == 3


@patch("agentic_analyst.llm.client")
def test_tenacity_does_not_retry_client_error(mock_client):
    """ClientError (4xx) is permanent → raises immediately, NO retry."""
    mock_client.models.generate_content.side_effect = genai_errors.ClientError(
        400, {"error": {"message": "bad request"}}
    )

    with pytest.raises(genai_errors.ClientError):
        call("haiku", "sys", "user")

    assert mock_client.models.generate_content.call_count == 1
