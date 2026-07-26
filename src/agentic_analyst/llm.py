"""The single door to the model.

Every LLM call in the system goes through `call()`. That gives one place to
put retries, cost metering, structured-output handling and logging, instead of
scattering them across every agent.

Two kinds of retry live here and they are deliberately separate:

* **Transport retries** (`_raw_generate`, via tenacity) — the network or the
  provider failed. Retrying the identical request is the right move.
* **Schema retry** (`call`, by hand) — the request succeeded but the model
  returned JSON that does not match the schema. Retrying the identical request
  would just fail again, so we re-prompt *with the failure text included*.
"""

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import overload

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

from agentic_analyst.settings import MODEL_TIERS, PRICES, SETTINGS

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 8192


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Build the Gemini client on first use, then reuse it.

    Constructing this at import time would make `import agentic_analyst.llm`
    fail whenever GOOGLE_API_KEY is absent — breaking a fresh clone, CI, and
    every mocked test. Building it lazily means the key is only needed at the
    moment a real API call happens, and the error you get is this one rather
    than a stack trace from inside the SDK.
    """
    if not SETTINGS.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free from https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=SETTINGS.google_api_key)


# ════════════════════════════════════════════════════════════════════
# Cost metering
# ════════════════════════════════════════════════════════════════════


@dataclass
class CostDelta:
    """How much a block of work cost. Filled in when its `track()` block exits."""

    usd: float = 0.0


def _billed_output_tokens(
    usage: types.GenerateContentResponseUsageMetadata | None,
) -> int:
    """Output tokens you actually pay for: visible answer + invisible thinking.

    Gemini reports thinking tokens separately but bills them at the output rate.
    Anything that reports or prices output must go through here — computing it
    in two places is how the log line and the cost silently drift apart.
    """
    if usage is None:
        return 0
    return (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)


@dataclass
class RunMeter:
    """Running total of tokens and spend for the process.

    Deliberately a plain mutable object rather than something threaded through
    the graph: every call site would otherwise have to pass it along. Note it
    is process-global, so it is not safe across concurrent runs — fine for the
    single-run CLI, and revisited if the graph ever fans out in parallel.
    """

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls: int = 0

    def record(
        self,
        model_id: str,
        usage: types.GenerateContentResponseUsageMetadata | None,
    ) -> float:
        """Add one API call's usage to the totals and return what it cost."""
        if usage is None:
            return 0.0
        in_tok = usage.prompt_token_count or 0
        out_tok = _billed_output_tokens(usage)
        rates = PRICES[model_id]
        cost = in_tok / 1_000_000 * rates["input"] + out_tok / 1_000_000 * rates["output"]
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cost_usd += cost
        self.calls += 1
        return cost

    @contextmanager
    def track(self) -> Iterator[CostDelta]:
        """Measure the spend of everything inside the `with` block.

        Lets a graph node report *its own* cost without each node having to
        remember the before/after arithmetic:

            with RUN_METER.track() as spend:
                result = call(...)
            return {"plan": ..., "cost_usd": spend.usd}
        """
        delta = CostDelta()
        before = self.total_cost_usd
        try:
            yield delta
        finally:
            delta.usd = self.total_cost_usd - before

    def reset(self) -> None:
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

    def report(self) -> str:
        return (
            f"calls={self.calls}  in={self.total_input_tokens}  "
            f"out={self.total_output_tokens}  cost=${self.total_cost_usd:.6f}"
        )


RUN_METER = RunMeter()


# ════════════════════════════════════════════════════════════════════
# The call itself
# ════════════════════════════════════════════════════════════════════

_TRANSPORT_ERRORS = (
    genai_errors.ServerError,
    httpx.TimeoutException,
    httpx.ConnectError,
)
RATE_LIMITED = 429
MAX_RATE_LIMIT_WAIT = 65.0  # a per-minute quota can never need longer


@dataclass
class Throttle:
    """Client-side spacing between API calls.

    The free tier allows 15 requests per minute per model, and one researcher
    task alone can spend eight. Without this the graph discovers the quota by
    hitting it, halfway through a run, having already paid for everything
    before the failure. Spacing the calls is cheaper than retrying them, so
    the retry below is the safety net rather than the mechanism.

    `min_interval = 0` disables it — that is what the mocked test suite uses.
    """

    min_interval: float
    _last_call: float = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        gap = self.min_interval - (time.monotonic() - self._last_call)
        if gap > 0:
            log.debug("throttling %.1fs to stay under the rate limit", gap)
            time.sleep(gap)
        self._last_call = time.monotonic()


THROTTLE = Throttle(
    min_interval=60.0 / SETTINGS.max_requests_per_minute
    if SETTINGS.max_requests_per_minute > 0
    else 0.0
)


def _is_retryable(exc: BaseException) -> bool:
    """Transport failures, and rate limiting — but no other 4xx.

    A 429 is the one client error worth retrying: it says "not now", not
    "never". A 400 or a 403 would fail identically forever.
    """
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == RATE_LIMITED


def _server_retry_hint(exc: BaseException) -> float | None:
    """Seconds the API itself asked us to wait, if it said.

    A 429 carries a RetryInfo telling you exactly when the window reopens.
    Guessing with exponential backoff when the server has already told you
    the answer means waiting either too long or not long enough.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in details.get("error", {}).get("details", []):
            delay = item.get("retryDelay") if isinstance(item, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                with suppress(ValueError):
                    return float(delay[:-1])
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) if match else None


_BACKOFF = wait_exponential(multiplier=1, min=1, max=10)


def _wait_strategy(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    hint = _server_retry_hint(exc) if exc else None
    if hint is not None:
        wait = min(hint + 1.0, MAX_RATE_LIMIT_WAIT)
        log.warning("rate limited; the API asked for %.0fs, waiting %.0fs", hint, wait)
        return wait
    return _BACKOFF(retry_state)


@retry(
    stop=stop_after_attempt(4),
    wait=_wait_strategy,
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _raw_generate(
    model_id: str,
    contents: types.ContentListUnion,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Single, tenacity-guarded door to the network.

    Every API call in this module goes through here, so transient errors
    (5xx / timeouts / 429) are retried in exactly one place. Any other
    ClientError (4xx) is permanent and bubbles up unretried.
    """
    THROTTLE.wait()
    return _get_client().models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )


class LLMError(RuntimeError):
    """The provider failed in a way retrying will not fix.

    Agents catch `RuntimeError`, never `google.genai.errors.*` — that is what
    keeps the provider swappable in `settings.py` alone, and it is why a
    quota-exhausted API ends one task instead of dumping a LangGraph traceback
    over an otherwise finished run.
    """


def _generate_and_meter(
    model_id: str,
    user: str,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """One metered, logged round trip."""
    try:
        resp = _raw_generate(model_id, user, config)
    except genai_errors.APIError as exc:
        if getattr(exc, "code", None) == RATE_LIMITED:
            raise LLMError(
                "Gemini rate limit still hit after retries. The free tier allows "
                f"15 requests/minute; this run is pacing at "
                f"{SETTINGS.max_requests_per_minute}/minute. Lower "
                "MAX_REQUESTS_PER_MINUTE in .env, or wait a minute and re-run."
            ) from exc
        raise LLMError(f"Gemini call failed: {exc}") from exc
    usage = resp.usage_metadata
    cost = RUN_METER.record(model_id, usage)
    log.info(
        "llm call model=%s in=%s out=%s (thinking=%s) cost=$%.6f",
        model_id,
        usage.prompt_token_count if usage else "?",
        _billed_output_tokens(usage),
        (usage.thoughts_token_count or 0) if usage else "?",
        cost,
    )
    return resp


def _finish_reason(resp: types.GenerateContentResponse) -> str:
    """Why the model stopped — the useful half of an empty-response error."""
    if resp.candidates and resp.candidates[0].finish_reason is not None:
        return str(resp.candidates[0].finish_reason)
    return "unknown"


# These two @overload lines are type-checker-only signatures. They tell mypy
# (and your editor) that `call(...)` with no schema returns `str`, while
# `call(..., schema=Plan)` returns `Plan`. Without them the return type is the
# vague `str | BaseModel` and every caller needs an `assert isinstance(...)`.
# Only the third definition below actually runs.
@overload
def call(tier: str, system: str, user: str, schema: None = None) -> str: ...


@overload
def call[ModelT: BaseModel](tier: str, system: str, user: str, schema: type[ModelT]) -> ModelT: ...


def call[ModelT: BaseModel](
    tier: str,
    system: str,
    user: str,
    schema: type[ModelT] | None = None,
) -> str | ModelT:
    """Send one prompt to the model for the given capability tier.

    Args:
        tier: "strong" or "fast" — see `settings.MODEL_TIERS`.
        system: System instruction (usually the contents of a prompts/*.md file).
        user: The turn-specific input.
        schema: Optional Pydantic model. If given, the model is constrained to
            JSON matching it and a validated instance is returned.

    Raises:
        RuntimeError: the model returned no text (safety block, or the token
            budget ran out before it produced any).
        ValueError: the model could not produce valid JSON for `schema`, even
            after being shown its own broken output once.
    """
    model_id = MODEL_TIERS[tier]

    # ── Path 1: plain text (no schema) ──────────────────────────
    if schema is None:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        resp = _generate_and_meter(model_id, user, config)
        if resp.text is None:
            # `.text` is None whenever the response has no text part — most
            # often a safety block, or MAX_TOKENS reached while the model was
            # still thinking. Failing loudly here beats returning None and
            # crashing somewhere downstream with no explanation.
            raise RuntimeError(
                f"Model {model_id} returned no text (finish_reason={_finish_reason(resp)})."
            )
        return resp.text

    # ── Path 2: structured output with retry-once ───────────────
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # First attempt
    resp = _generate_and_meter(model_id, user, config)
    if isinstance(resp.parsed, schema):
        return resp.parsed

    # Retry once with the failure context (schema retry — NOT tenacity's job)
    log.warning("schema parse failed for %s, re-prompting once", schema.__name__)
    retry_user = (
        f"{user}\n\n"
        f"---\n"
        f"Your previous response could not be parsed against the required schema.\n"
        f"Raw output was:\n{resp.text}\n\n"
        f"Please return valid JSON matching the schema exactly."
    )
    resp = _generate_and_meter(model_id, retry_user, config)
    if isinstance(resp.parsed, schema):
        return resp.parsed

    # Give up
    raise ValueError(
        f"Model failed to produce valid {schema.__name__} after retry. "
        f"Last raw output: {resp.text!r}"
    )
