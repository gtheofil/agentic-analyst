"""Unit tests for the search wrapper.

These fake the Tavily client entirely
"""

from unittest.mock import MagicMock

import pytest

from agentic_analyst.tools.search import SearchArgs, SearchResult, search


def test_search_filters_blocklist_and_shapes(monkeypatch):
    # Canned raw Tavily response: 3 hits, one on a blocked domain.
    fake_response = {
        "results": [
            {"title": "Good one",  "url": "https://example.com/a", "content": "aaa"},
            {"title": "Pinterest", "url": "https://pinterest.com/x", "content": "bad"},
            {"title": "Good two",  "url": "https://example.org/b", "content": "bbb"},
        ]
    }

    fake_client = MagicMock()
    fake_client.search.return_value = fake_response

    # Whenever search() calls _get_client(), hand back the fake instead.
    monkeypatch.setattr(
        "agentic_analyst.tools.search._get_client",
        lambda: fake_client,
    )

    results = search(SearchArgs(query="anything", k=5))

    # The pinterest hit is dropped → 2 survive.
    assert len(results) == 2
    # Output is our clean model, not raw dicts.
    assert all(isinstance(r, SearchResult) for r in results)
    # Tavily's `content` was renamed to our `snippet`.
    assert results[0].snippet == "aaa"
    # No blocked domain slipped through.
    assert all("pinterest.com" not in r.url for r in results)


def test_search_respects_k(monkeypatch):
    # 4 clean hits available, but we only ask for 2.
    fake_response = {
        "results": [
            {"title": "1", "url": "https://a.com", "content": "1"},
            {"title": "2", "url": "https://b.com", "content": "2"},
            {"title": "3", "url": "https://c.com", "content": "3"},
            {"title": "4", "url": "https://d.com", "content": "4"},
        ]
    }
    fake_client = MagicMock()
    fake_client.search.return_value = fake_response
    monkeypatch.setattr(
        "agentic_analyst.tools.search._get_client",
        lambda: fake_client,
    )

    results = search(SearchArgs(query="anything", k=2))

    assert len(results) == 2


def test_k_is_capped():
    # k > 5 must be rejected by the model itself, before any call is made.
    with pytest.raises(ValueError):
        SearchArgs(query="anything", k=99)