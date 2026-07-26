import httpx
import pytest
from agentic_analyst.tools.fetch import _download, _extract, FetchError, _MAX_BYTES

_HTML = """
<html><body><article>
<h1>LangGraph agents</h1>
<p>This is a real paragraph of body text long enough for trafilatura
to treat it as the main content of the page and extract it cleanly.</p>
</article></body></html>
"""


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler),
                        follow_redirects=True)


def test_extract_strips_markup():
    text = _extract(_HTML, "https://example.com")
    assert "real paragraph" in text
    assert "<p>" not in text


def test_download_reads_body():
    def handler(_req):
        return httpx.Response(200, text="hello")
    with _mock_client(handler) as client:
        assert _download("https://example.com", client=client) == "hello"


def test_download_respects_cap():
    def handler(_req):
        return httpx.Response(200, content=b"x" * (_MAX_BYTES + 1))
    with _mock_client(handler) as client:
        with pytest.raises(FetchError):
            _download("https://example.com/big", client=client)