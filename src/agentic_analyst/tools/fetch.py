from __future__ import annotations

import httpx
import trafilatura
from pydantic import BaseModel, Field

_TIMEOUT = 10.0
_MAX_BYTES = 1000 * 1024  # 1MB cap
_HEADERS = {"User-Agent": "consulting-agent/0.1 (research tool; contact: gt@gmail.com)"}


class FetchArgs(BaseModel):
    url: str = Field(..., description="Absolute URL of the page to fetch.")


class FetchResult(BaseModel):
    url: str
    text: str


class FetchError(RuntimeError):
    """Raised when a page can't be fetched or turned into clean text."""


def fetch(args: FetchArgs) -> FetchResult:
    """Download a page and return its main text (boilerplate stripped)."""
    html = _download(args.url)
    return FetchResult(url=args.url, text=_extract(html, args.url))


def _download(url: str, client: httpx.Client | None = None) -> str:
    """Stream the body, aborting once we cross the byte cap."""
    owns = client is None

    client = client or httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers=_HEADERS,
    )
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise FetchError(f"{url} exceeds the {_MAX_BYTES}-byte cap")
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        if owns:
            client.close()


def _extract(html: str, url: str) -> str:
    """Pure step: HTML in, clean article text out."""
    text = trafilatura.extract(html, url=url)
    if not text or not text.strip():
        raise FetchError(f"No extractable text at {url}")
    return text.strip()
