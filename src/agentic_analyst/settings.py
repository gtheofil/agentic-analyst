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

    # What one run may spend before the budget guard refuses further calls.
    # A four-task run costs ~$0.085, and one that uses both revisions ~$0.17,
    # so $0.50 is roughly 3x the worst observed run: high enough never to fire
    # on a healthy run, low enough that a pathological one cannot quietly cost
    # a dollar. 0 disables the guard.
    max_run_cost_usd: float = 0.50

    # Langfuse tracing. Optional for exactly the reason the Gemini key is: a
    # fresh clone with no credentials must still import, lint, type-check and
    # pass the mocked suite. With no keys the tracer no-ops and the run behaves
    # identically, just unobserved.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    # Langfuse v4 renamed this from LANGFUSE_HOST. EU region by default; US
    # projects need https://us.cloud.langfuse.com, and a self-hosted instance
    # its own URL.
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def tracing_enabled(self) -> bool:
        """Whether traces should be emitted at all.

        Both halves of the key pair or neither: one on its own is a typo, not a
        configuration, and Langfuse would fail at export time rather than here.
        Every module asks this question here instead of re-deriving it, so no
        two of them can disagree about whether tracing is on.
        """
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


SETTINGS = Settings()
