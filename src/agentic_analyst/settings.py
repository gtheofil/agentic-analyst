"""Configuration: everything environment-dependent or model-dependent lives here.

Nothing else in the codebase reads `os.environ` directly, and nothing else
hardcodes a model ID. When the provider changes, this file changes and the
agents do not.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ───────────────────────────────────────────────────
# settings.py lives at <repo>/src/agentic_analyst/settings.py, so parents[2]
# is the repo root. Resolved once here so no other module has to count
# directories back up the tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Things the agent writes (the Chroma store) vs. things a human writes for it
# (prefs.yaml). Kept apart so `data/` can be .gitignored wholesale while
# `config/` is committed.
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

MODEL_TIERS: dict[str, str] = {
    "strong": "gemini-3.6-flash",
    "fast": "gemini-3.5-flash-lite",
}

PRICES: dict[str, dict[str, float]] = {
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional on purpose: the package must import, lint, type-check and run its
    # mocked test suite with no key present. The key is only required at the
    # moment a real API call is made — see `llm._get_client`.
    google_api_key: str | None = None
    tavily_api_key: str | None = None

    # The Gemini free tier allows 15 requests/minute/model, and a four-task run
    # makes roughly thirty calls. Default a little under the ceiling so a run
    # is paced rather than rate-limited; raise it on a paid key, or set 0 to
    # turn the client-side throttle off entirely.
    max_requests_per_minute: int = 12


SETTINGS = Settings()
