from dataclasses import dataclass
from typing import cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agentic_analyst.settings import PRICES, SETTINGS

client = genai.Client(api_key=SETTINGS.google_api_key)

TIER_MAP = {
    "sonnet": "gemini-2.5-pro",
    "haiku": "gemini-2.5-flash",
}


@dataclass
class RunMeter:
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls: int = 0

    def record(
        self,
        model_id: str,
        usage: types.GenerateContentResponseUsageMetadata | None,
    ) -> float:
        if usage is None:
            return 0.0
        in_tok = usage.prompt_token_count or 0
        out_tok = (usage.candidates_token_count or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        rates = PRICES[model_id]
        cost = in_tok / 1_000_000 * rates["input"] + out_tok / 1_000_000 * rates["output"]
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cost_usd += cost
        self.calls += 1
        return cost

    def report(self) -> str:
        return (
            f"calls={self.calls}  in={self.total_input_tokens}  "
            f"out={self.total_output_tokens}  cost=${self.total_cost_usd:.6f}"
        )


RUN_METER = RunMeter()

_RETRYABLE = (
    genai_errors.ServerError,
    httpx.TimeoutException,
    httpx.ConnectError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _raw_generate(
    model_id: str,
    contents: types.ContentListUnion,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Single, tenacity-guarded door to the model.

    Every API call in this module goes through here, so transient errors
    (5xx / timeouts) are retried in exactly one place. Retries 3x with
    exponential backoff; ClientError (4xx) is NOT retryable and bubbles up.
    """
    if SETTINGS.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env or environment.")
    return client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )


def call(
    tier: str,
    system: str,
    user: str,
    schema: type[BaseModel] | None = None,
) -> str | BaseModel:
    model_id = TIER_MAP[tier]

    # ── Path 1: plain text (no schema) ──────────────────────────
    if schema is None:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=2048,
        )
        resp = _raw_generate(model_id, user, config)
        RUN_METER.record(model_id, resp.usage_metadata)
        return cast(BaseModel, resp.text)

    # ── Path 2: structured output with retry-once ───────────────
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # First attempt
    resp = _raw_generate(model_id, user, config)
    RUN_METER.record(model_id, resp.usage_metadata)
    if resp.parsed is not None:
        return cast(BaseModel, resp.parsed)

    # Retry once with the failure context (schema retry — NOT tenacity's job)
    retry_user = (
        f"{user}\n\n"
        f"---\n"
        f"Your previous response could not be parsed against the required schema.\n"
        f"Raw output was:\n{resp.text}\n\n"
        f"Please return valid JSON matching the schema exactly."
    )
    resp = _raw_generate(model_id, retry_user, config)
    RUN_METER.record(model_id, resp.usage_metadata)
    if resp.parsed is not None:
        return cast(BaseModel, resp.parsed)

    # Give up
    raise ValueError(
        f"Model failed to produce valid {schema.__name__} after retry. "
        f"Last raw output: {resp.text!r}"
    )
