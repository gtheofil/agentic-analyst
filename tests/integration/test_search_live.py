"""Integration (live) test for the search wrapper.

Hits the real Tavily API. Runs ONLY when you ask for it explicitly and a key
is present. Not part of the everyday/CI run.

    uv run pytest -m integration        # run this
    uv run pytest -m "not integration"  # skip it (default CI command)
    uv run pytest tests/integration/test_search_live.py -v -m integration -s
"""

import pytest

from agentic_analyst.settings import SETTINGS
from agentic_analyst.tools.search import SearchArgs, search

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SETTINGS.tavily_api_key,  # ← reads .env via pydantic
        reason="TAVILY_API_KEY not set",
    ),
]


def test_live_smoke() -> None:
    results = search(SearchArgs(query="LangGraph tool calling", k=3))

    for i, r in enumerate(results, start=1):
        print(f"\n{i}. {r.title}")
        print(f"   {r.url}")
        print(f"   {r.snippet[:120]}...")  # first 120 chars of the snippet

    # We got something back.
    assert len(results) > 0
    # Real results have real http(s) URLs.
    assert all(r.url.startswith("http") for r in results)
    # And the clean shape survived the round trip.
    assert all(r.title and r.snippet for r in results)
