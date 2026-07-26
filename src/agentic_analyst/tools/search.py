from functools import lru_cache

from tavily import TavilyClient

from ..settings import SETTINGS
from pydantic import BaseModel, Field

_BLOCKLIST: frozenset[str] = frozenset(
    {"pinterest.com", "quora.com", "wsj.com", "ft.com", "nytimes.com"}
)

@lru_cache(maxsize=1)
def _get_client() -> TavilyClient:
    """Build the Tavily client once, on first real use.

    Lazy on purpose: the module must import, lint, type-check and run its
    mocked tests with no key present. The key is only required the moment a
    real search happens — mirrors `llm._get_client`.
    """
    if not SETTINGS.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env to run live searches."
        )
    return TavilyClient(api_key=SETTINGS.tavily_api_key)

class SearchArgs(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=5)

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

def _is_blocked(url: str) -> bool:
    return any(bad in url for bad in _BLOCKLIST)

def search(args:SearchArgs) -> list[SearchResult]:
    client=_get_client()

    response = client.search(query=args.query, max_results=args.k+3)

    results: list[SearchResult] = []

    for hit in response.get("results", ""):
        url = hit.get("url", "")
        if not url or _is_blocked(url):
            continue

        results.append(
            SearchResult(
                title=hit.get("title", ""),
                url=url,
                snippet=hit.get("content", ""),
            )
        )
        if len(results) >= args.k:
            break

    return results