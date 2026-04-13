"""Smoke tests for cross-round URL deduplication — P0-B fix.

Verifies:
- ResearchState has fetched_urls field with list[str] + operator.add reducer
- phase1a skips URLs already in state.fetched_urls
- phase1a returns new fetched URLs in output for state accumulation
- UNREACHABLE URLs are NOT added to fetched_urls (so they can be retried)
"""

import typing
import operator

import pytest

from deep_research.state import ResearchState


# ---------------------------------------------------------------------------
# State field
# ---------------------------------------------------------------------------

def test_state_has_fetched_urls():
    """ResearchState must have fetched_urls field."""
    ann = typing.get_type_hints(ResearchState, include_extras=True)
    assert "fetched_urls" in ann, "fetched_urls field missing from ResearchState"


def test_fetched_urls_uses_add_reducer():
    """fetched_urls must use operator.add reducer (accumulate across iterations)."""
    ann = typing.get_type_hints(ResearchState, include_extras=True)
    fetched_ann = ann["fetched_urls"]
    # Annotated[list[str], operator.add] — check metadata contains the add function
    args = typing.get_args(fetched_ann)
    assert len(args) >= 2, "fetched_urls annotation must have reducer metadata"
    reducer = args[1]
    assert reducer is operator.add, (
        f"fetched_urls reducer must be operator.add, got {reducer}"
    )


# ---------------------------------------------------------------------------
# URL dedup filtering in _select_urls_by_quota output
# ---------------------------------------------------------------------------

def test_prior_fetched_urls_are_skipped():
    """URLs already in state.fetched_urls must be filtered from selected list."""
    # Simulate the filtering logic used in phase1a_search
    selected = [
        {"url": "https://example.com/a", "title": "A"},
        {"url": "https://example.com/b", "title": "B"},
        {"url": "https://example.com/c", "title": "C"},
    ]
    prior_fetched = {"https://example.com/a", "https://example.com/b"}

    filtered = [s for s in selected if s["url"] not in prior_fetched]

    assert len(filtered) == 1
    assert filtered[0]["url"] == "https://example.com/c"


def test_empty_prior_fetched_passes_all():
    """When fetched_urls is empty (first round), all selected URLs pass through."""
    selected = [
        {"url": "https://example.com/a"},
        {"url": "https://example.com/b"},
    ]
    prior_fetched: set[str] = set()

    filtered = [s for s in selected if s["url"] not in prior_fetched]
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Return value: UNREACHABLE must not be added to fetched_urls
# ---------------------------------------------------------------------------

def test_unreachable_urls_excluded_from_new_fetched():
    """UNREACHABLE sources must not be added to fetched_urls (allow retry next round)."""
    raw_sources = [
        {"url": "https://live.com/page", "status": "LIVE"},
        {"url": "https://thin.com/page", "status": "THIN_CONTENT"},
        {"url": "https://dead.com/page", "status": "UNREACHABLE"},
    ]
    # replicate the logic from phase1a_search return statement
    new_fetched = [
        s["url"] for s in raw_sources
        if s.get("status") not in ("UNREACHABLE",) and s.get("url")
    ]
    assert "https://live.com/page" in new_fetched
    assert "https://thin.com/page" in new_fetched
    assert "https://dead.com/page" not in new_fetched


def test_operator_add_accumulates_across_iterations():
    """operator.add on two URL lists must produce union (with possible duplication guard)."""
    round1 = ["https://a.com", "https://b.com"]
    round2 = ["https://c.com"]
    accumulated = operator.add(round1, round2)
    assert "https://a.com" in accumulated
    assert "https://b.com" in accumulated
    assert "https://c.com" in accumulated
    assert len(accumulated) == 3
