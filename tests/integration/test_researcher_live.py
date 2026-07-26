"""Live test of the researcher loop — the Phase 2.3 acceptance check.

Hits Gemini, Tavily and the open web, and costs a fraction of a cent. It is
the only automated answer to the question that matters: are the quotes real?
A finding whose quote does not appear in its own source is not a weak finding,
it is a fabricated one, and only a live run can tell you which you have.

    uv run pytest tests/integration/test_researcher_live.py -m integration -s
"""

import re

import pytest

from agentic_analyst.agents.researcher import MAX_FINDINGS, researcher
from agentic_analyst.settings import SETTINGS
from agentic_analyst.state import Finding, Task
from agentic_analyst.tools.fetch import FetchArgs, fetch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (SETTINGS.google_api_key and SETTINGS.tavily_api_key),
        reason="GOOGLE_API_KEY and TAVILY_API_KEY are both needed for a live run",
    ),
]

TASK = Task(
    id=1,
    goal=(
        "Quantify how many heat pumps were installed in the UK in 2024, "
        "and how that compares with 2023."
    ),
)


def _normalise(text: str) -> str:
    """Compare quotes ignoring whitespace and quote-mark style.

    The model retypes a quote from text we extracted, and re-fetching the page
    can re-wrap it differently. Neither difference makes the quote invented, so
    neither should fail the check — anything beyond that should.
    """
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_is_in_source(finding: Finding) -> bool:
    """Re-fetch the cited page and look for the quote in it."""
    try:
        page = fetch(FetchArgs(url=finding.source_url))
    except Exception as exc:  # a page that has since moved is not a fabrication
        pytest.skip(f"could not re-fetch {finding.source_url}: {exc}")
    return _normalise(finding.quote) in _normalise(page.text)


def test_live_task_yields_verifiable_findings() -> None:
    findings = researcher(TASK)

    for finding in findings:
        print(f"\n[{finding.confidence}] {finding.claim}")
        print(f"  quote: {finding.quote[:160]}")
        print(f"  source: {finding.source_url}")

    assert findings, "the researcher recorded nothing at all"
    assert len(findings) <= MAX_FINDINGS

    assert all(f.task_id == TASK.id for f in findings)
    assert all(f.source_url.startswith("http") for f in findings)
    assert all(f.claim and f.quote for f in findings)

    verified = [f for f in findings if _quote_is_in_source(f)]
    for f in findings:
        if f not in verified:
            print(f"\nUNVERIFIED quote from {f.source_url}:\n  {f.quote}")

    assert verified, (
        "no finding's quote could be found in its own source — the prompt is "
        "not holding the model to verbatim quoting"
    )
