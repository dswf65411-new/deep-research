"""Verify phase0→phase1a seed-URL bypass.

Plan context: the original audit found that phase0 correctly listed arxiv URLs
and paper links in the plan, but phase1a never fetched them — it only searched.
Search engines then ranked marketing blogs above those arxiv abstracts and the
plan's specific citations were effectively invisible.

This test suite verifies `_extract_seed_urls` correctly:
  1. Extracts explicit URLs from the plan.
  2. Synthesizes arxiv URLs from bare IDs.
  3. Tags each seed with the nearest preceding `## Q{n}:` heading so SQ coverage
     is attributed correctly.
  4. Skips URLs inside `[REMOVED: ...]` markers written by phase0's validator.
  5. Dedupes across raw URLs and arxiv-ID synthesis.
"""

from deep_research.nodes.phase1a import _extract_seed_urls


def test_extracts_plain_urls_and_tags_with_sq():
    plan = """# Research plan

## Q1: background
See https://langchain-ai.github.io/langgraph/ for framework overview.

## Q3: supervisor pattern failure handling
The Supervisor approach in https://arxiv.org/abs/2310.05193 is canonical.
"""
    seeds = _extract_seed_urls(plan)
    urls = {s["url"] for s in seeds}
    assert "https://langchain-ai.github.io/langgraph/" in urls
    assert "https://arxiv.org/abs/2310.05193" in urls

    sq_by_url = {s["url"]: s["subquestion"] for s in seeds}
    assert sq_by_url["https://langchain-ai.github.io/langgraph/"] == "Q1"
    assert sq_by_url["https://arxiv.org/abs/2310.05193"] == "Q3"


def test_synthesizes_arxiv_from_bare_id():
    """`## Q2: ... see 2402.04942 for details.` should synthesize
    https://arxiv.org/abs/2402.04942 as a seed."""
    plan = "## Q2: Supervisors\nSee 2402.04942 for prior work.\n"
    seeds = _extract_seed_urls(plan)
    urls = [s["url"] for s in seeds]
    assert "https://arxiv.org/abs/2402.04942" in urls


def test_skips_removed_marker_urls():
    """Phase0's validator marks hallucinated URLs with `[REMOVED: ...](url)`.
    Seed extraction must skip those — the URL inside the marker is known bad.
    `2604.05550` is a failure-workspace example (future month, doesn't exist)."""
    plan = """## Q1: prior work
- [REMOVED: hallucinated arxiv](https://arxiv.org/abs/2604.05550)
- Real paper: https://arxiv.org/abs/2310.05193
"""
    seeds = _extract_seed_urls(plan)
    urls = [s["url"] for s in seeds]
    assert "https://arxiv.org/abs/2310.05193" in urls
    assert "https://arxiv.org/abs/2604.05550" not in urls


def test_dedupes_url_and_arxiv_id_pointing_to_same_paper():
    """If plan has both `https://arxiv.org/abs/2310.05193` and bare `2310.05193`,
    they resolve to the same URL — dedupe."""
    plan = """## Q1: refs
https://arxiv.org/abs/2310.05193
See also 2310.05193 for the source."""
    seeds = _extract_seed_urls(plan)
    urls = [s["url"] for s in seeds]
    assert urls.count("https://arxiv.org/abs/2310.05193") == 1


def test_defaults_to_q1_when_no_preceding_heading():
    """URL appearing before any `## Qn:` heading (e.g. in an abstract paragraph)
    defaults to Q1 rather than being dropped."""
    plan = """# Overview

This research cites https://example.com/paper.

## Q1: main question
Content here.
"""
    seeds = _extract_seed_urls(plan)
    [seed] = [s for s in seeds if s["url"] == "https://example.com/paper"]
    assert seed["subquestion"] == "Q1"


def test_empty_plan_returns_empty():
    assert _extract_seed_urls("") == []
    assert _extract_seed_urls("## Q1: no URLs here at all") == []


def test_seed_has_expected_shape():
    """Seeds must carry enough fields to slot into `selected` alongside search hits."""
    plan = "## Q1: x\nhttps://example.com\n"
    [seed] = _extract_seed_urls(plan)
    assert seed["role"] == "seed"
    assert seed["engines"] == ["seed"]
    assert seed["subquestion"] == "Q1"
    assert "title" in seed
    assert "description" in seed
