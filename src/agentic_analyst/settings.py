from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Constants (not environment-loaded) ──────────────────────
PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(..., description="Gemini API key from AI Studio")


SETTINGS = Settings()  # type: ignore[call-arg]
