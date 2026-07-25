"""Unit tests for the single door to the model.

Nothing here touches the network: the `mock_client` fixture (see
tests/conftest.py) replaces the Gemini client with a MagicMock.
"""

from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

import agentic_analyst.llm as llm
from agentic_analyst.llm import call
from agentic_analyst.settings import SETTINGS


class Toy(BaseModel):
    n: int


def _usage(prompt: int, output: int, thoughts: int = 0) -> MagicMock:
    """A stand-in for the SDK's usage metadata object."""
    return MagicMock(
        prompt_token_count=prompt,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
    )


# ════════════════════════════════════════════════════════════════════
# 1. The API key is only needed when a real call happens
# ════════════════════════════════════════════════════════════════════


def test_import_works_without_api_key() -> None:
    """The module is already imported at the top of this file with no key set.

    This is the guarantee that lets CI, linting and the mocked test suite run
    on a machine that has never seen a Gemini key.
    """
    assert llm.call is not None


def test_client_raises_a_helpful_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SETTINGS, "google_api_key", None)
    llm._get_client.cache_clear()

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is not set"):
        llm._get_client()

    llm._get_client.cache_clear()


# ════════════════════════════════════════════════════════════════════
# 2. Schema retry-once  (bad JSON → re-prompt with the error text)
# ════════════════════════════════════════════════════════════════════


def test_retry_once_then_succeeds(mock_client: MagicMock) -> None:
    """First response is unparseable, second is valid → returns the model."""
    bad = MagicMock(parsed=None, text="garbage", usage_metadata=None)
    good = MagicMock(parsed=Toy(n=42), usage_metadata=None)
    mock_client.models.generate_content.side_effect = [bad, good]

    result = call("fast", "sys", "user", schema=Toy)

    assert result == Toy(n=42)
    assert mock_client.models.generate_content.call_count == 2


def test_retry_prompt_includes_the_broken_output(mock_client: MagicMock) -> None:
    """The re-prompt must show the model what it got wrong, or it just repeats it."""
    bad = MagicMock(parsed=None, text="garbage", usage_metadata=None)
    good = MagicMock(parsed=Toy(n=1), usage_metadata=None)
    mock_client.models.generate_content.side_effect = [bad, good]

    call("fast", "sys", "original user prompt", schema=Toy)

    second_prompt = mock_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert "original user prompt" in second_prompt
    assert "garbage" in second_prompt


def test_retry_once_then_gives_up(mock_client: MagicMock) -> None:
    """Both responses unparseable → raises ValueError after exactly 2 attempts."""
    bad = MagicMock(parsed=None, text="still garbage", usage_metadata=None)
    mock_client.models.generate_content.side_effect = [bad, bad]

    with pytest.raises(ValueError, match="failed to produce valid Toy"):
        call("fast", "sys", "user", schema=Toy)

    assert mock_client.models.generate_content.call_count == 2


# ════════════════════════════════════════════════════════════════════
# 3. Tenacity retry  (transient errors retried, client errors are not)
# ════════════════════════════════════════════════════════════════════


def test_tenacity_retries_on_server_error(mock_client: MagicMock) -> None:
    """ServerError (5xx) is transient → tenacity retries up to 3 attempts."""
    good = MagicMock(text="ok", usage_metadata=None)
    mock_client.models.generate_content.side_effect = [
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        good,
    ]

    result = call("fast", "sys", "user")  # text path, no schema

    assert result == "ok"
    assert mock_client.models.generate_content.call_count == 3


def test_tenacity_does_not_retry_client_error(mock_client: MagicMock) -> None:
    """ClientError (4xx) is permanent → raises immediately, NO retry."""
    mock_client.models.generate_content.side_effect = genai_errors.ClientError(
        400, {"error": {"message": "bad request"}}
    )

    with pytest.raises(genai_errors.ClientError):
        call("fast", "sys", "user")

    assert mock_client.models.generate_content.call_count == 1


# ════════════════════════════════════════════════════════════════════
# 4. Empty responses fail loudly
# ════════════════════════════════════════════════════════════════════


def test_empty_text_response_raises(mock_client: MagicMock) -> None:
    """A safety block or token-budget exhaustion gives `.text is None`.

    Returning that None would surface as a confusing crash inside the writer,
    so `call` raises here with the finish_reason attached instead.
    """
    blocked = MagicMock(
        text=None,
        usage_metadata=None,
        candidates=[MagicMock(finish_reason="MAX_TOKENS")],
    )
    mock_client.models.generate_content.return_value = blocked
    mock_client.models.generate_content.side_effect = None

    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        call("fast", "sys", "user")


# ════════════════════════════════════════════════════════════════════
# 5. Cost metering
# ════════════════════════════════════════════════════════════════════


def test_meter_records_cost_from_token_usage(mock_client: MagicMock) -> None:
    """gemini-2.5-flash: $0.30/1M in, $2.50/1M out."""
    mock_client.models.generate_content.return_value = MagicMock(
        text="ok", usage_metadata=_usage(prompt=1_000_000, output=1_000_000)
    )
    mock_client.models.generate_content.side_effect = None

    call("fast", "sys", "user")

    assert llm.RUN_METER.calls == 1
    assert llm.RUN_METER.total_cost_usd == pytest.approx(2.80)


def test_meter_counts_thinking_tokens_as_output(mock_client: MagicMock) -> None:
    """Gemini bills thinking tokens at the output rate but reports them apart."""
    mock_client.models.generate_content.return_value = MagicMock(
        text="ok", usage_metadata=_usage(prompt=0, output=0, thoughts=1_000_000)
    )
    mock_client.models.generate_content.side_effect = None

    call("fast", "sys", "user")

    assert llm.RUN_METER.total_output_tokens == 1_000_000
    assert llm.RUN_METER.total_cost_usd == pytest.approx(2.50)


def test_logged_output_tokens_match_what_was_billed(
    mock_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """The log line must report billed output, not just the visible answer.

    Regression test: it used to print `candidates_token_count` alone, which
    under-reported output by 68% on a model that thinks — while the cost was
    computed correctly. The two numbers disagreed and nothing noticed.
    """
    mock_client.models.generate_content.return_value = MagicMock(
        text="ok", usage_metadata=_usage(prompt=100, output=188, thoughts=401)
    )
    mock_client.models.generate_content.side_effect = None

    with caplog.at_level("INFO", logger="agentic_analyst.llm"):
        call("fast", "sys", "user")

    assert "out=589" in caplog.text, "log should report 188 visible + 401 thinking"
    assert llm.RUN_METER.total_output_tokens == 589


def test_track_reports_only_its_own_block(mock_client: MagicMock) -> None:
    """`RUN_METER.track()` is how a node reports its share of the spend."""
    mock_client.models.generate_content.return_value = MagicMock(
        text="ok", usage_metadata=_usage(prompt=1_000_000, output=0)
    )
    mock_client.models.generate_content.side_effect = None

    call("fast", "sys", "outside the block")  # $0.30, should not be counted

    with llm.RUN_METER.track() as spend:
        call("fast", "sys", "inside the block")  # $0.30, should be

    assert spend.usd == pytest.approx(0.30)
    assert llm.RUN_METER.total_cost_usd == pytest.approx(0.60)
