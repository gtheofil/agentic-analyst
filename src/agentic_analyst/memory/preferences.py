"""Static user preferences: how the writer should style its output.

Unlike episodic memory (which the agent *earns* over many runs), preferences are
human-authored standing instructions read from prefs.yaml. They are folded into
the writer prompt so a user can steer tone and length without touching code.
No LLM, no vector DB — just config.
"""

import logging

import yaml
from pydantic import BaseModel

from ..settings import CONFIG_DIR

log = logging.getLogger(__name__)

_PREFS_PATH = CONFIG_DIR / "prefs.yaml"


class Preferences(BaseModel):
    """Writer style knobs, with sensible defaults if prefs.yaml is absent."""

    tone: str = "professional and neutral"
    length: str = "concise, roughly 600-800 words"


def load_preferences() -> Preferences:
    """Read prefs.yaml into a Preferences object, falling back to defaults."""
    if not _PREFS_PATH.exists():
        log.info("no prefs.yaml found; using default preferences")
        return Preferences()

    data = yaml.safe_load(_PREFS_PATH.read_text(encoding="utf-8")) or {}
    prefs = Preferences(**data)
    log.info("loaded preferences: tone=%r length=%r", prefs.tone, prefs.length)
    return prefs


def format_preferences(prefs: Preferences) -> str:
    """Render preferences as a prompt fragment the writer can drop in."""
    return (
        "## Style preferences\n"
        f"- Tone: {prefs.tone}\n"
        f"- Length: {prefs.length}"
    )