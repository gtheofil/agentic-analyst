"""The golden set itself.

An eval set is a test fixture that decides what "better" means, so a typo in it
is a silently wrong measurement rather than a crash. These checks are cheap and
run every time.
"""

from collections import Counter

from evals.golden_set import GOLDEN_DIR, load_golden_set

SMOKE_SUBSET_SIZE = 3
MIN_BRIEFS = 10
MIN_DOMAINS = 4


def test_every_brief_loads_and_validates() -> None:
    briefs = load_golden_set()

    assert len(briefs) >= MIN_BRIEFS
    assert len(briefs) == len(list(GOLDEN_DIR.glob("*.yaml")))


def test_ids_are_unique_and_match_their_filenames() -> None:
    """The id is what appears in the results table, so it has to be findable."""
    for path in GOLDEN_DIR.glob("*.yaml"):
        brief = next(b for b in load_golden_set() if b.id == path.stem)
        assert brief.id == path.stem

    ids = [b.id for b in load_golden_set()]
    assert len(set(ids)) == len(ids)


def test_the_set_spans_at_least_four_domains() -> None:
    domains = Counter(b.domain for b in load_golden_set())

    assert len(domains) >= MIN_DOMAINS, domains


def test_the_smoke_subset_is_three_and_none_of_them_are_hard() -> None:
    """CI runs these on every PR.

    Three because ten is fifteen minutes and a dollar. None of them `hard`
    because a gate that fails on the briefs *designed* to be borderline is a
    gate that teaches people to ignore it.
    """
    smoke = load_golden_set(smoke_only=True)

    assert len(smoke) == SMOKE_SUBSET_SIZE
    assert all(b.difficulty != "hard" for b in smoke)


def test_the_set_has_an_easy_anchor() -> None:
    """Without one, a broken pipeline and a hard brief produce the same score."""
    assert any(b.difficulty == "easy" for b in load_golden_set())


def test_the_prompt_is_a_single_line() -> None:
    """YAML folded scalars keep a trailing newline and can keep interior ones.

    The brief is a user message; leaking the file's line breaks into it makes
    the input depend on how the YAML happened to wrap.
    """
    assert all("\n" not in b.prompt for b in load_golden_set())
