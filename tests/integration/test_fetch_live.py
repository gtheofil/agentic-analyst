import pytest

from agentic_analyst.tools.fetch import FetchArgs, fetch


@pytest.mark.integration
def test_live_fetch_real_article() -> None:
    result = fetch(FetchArgs(url="https://en.wikipedia.org/wiki/Las_Meninas"))
    print(result.text[:500])
    assert len(result.text) > 200
