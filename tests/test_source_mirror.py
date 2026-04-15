"""Tests for Whisper X-1 — cross-source mirror detector."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from deep_research.harness.source_mirror import (
    MirrorGroup,
    detect_mirror_groups,
    format_mirror_warnings,
    normalise_title,
    title_similarity,
)


@dataclass
class _Src:
    source_id: str
    url: str
    title: str = ""


# ---------------------------------------------------------------------------
# normalise_title / title_similarity
# ---------------------------------------------------------------------------


def test_normalise_title_strips_punct_and_stopwords():
    assert normalise_title("The Art of Research, Vol. 2!") == "art research vol 2"


def test_normalise_title_handles_empty():
    assert normalise_title("") == ""
    assert normalise_title(None) == ""  # type: ignore[arg-type]


def test_title_similarity_identical_is_one():
    assert title_similarity("LangGraph Supervisor", "LangGraph Supervisor") == 1.0


def test_title_similarity_disjoint_is_zero():
    assert title_similarity("apples and oranges", "quantum chromodynamics") == 0.0


def test_title_similarity_partial():
    sim = title_similarity(
        "Introduction to LangGraph Supervisor Pattern",
        "LangGraph Supervisor Pattern: A Practical Introduction",
    )
    assert 0.6 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Arxiv detection
# ---------------------------------------------------------------------------


def test_same_arxiv_id_different_formats_is_mirror():
    sources = [
        _Src("S1", "https://arxiv.org/abs/2310.05193", "ResearchAgent paper"),
        _Src("S2", "https://arxiv.org/html/2310.05193v2", "ResearchAgent paper (HTML)"),
        _Src("S3", "https://arxiv.org/pdf/2310.05193.pdf", "ResearchAgent paper (PDF)"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1
    assert groups[0].key == "arxiv:2310.05193"
    assert len(groups[0].sources) == 3


def test_different_arxiv_ids_are_not_grouped():
    sources = [
        _Src("S1", "https://arxiv.org/abs/2310.05193", "ResearchAgent"),
        _Src("S2", "https://arxiv.org/abs/2402.04942", "MLE-Agent"),
    ]
    assert detect_mirror_groups(sources) == []


def test_arxiv_id_with_version_suffix_grouped():
    sources = [
        _Src("S1", "https://arxiv.org/abs/2310.05193v1", "t1"),
        _Src("S2", "https://arxiv.org/abs/2310.05193v3", "t2"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# DOI detection
# ---------------------------------------------------------------------------


def test_same_doi_is_mirror():
    sources = [
        _Src("S1", "https://doi.org/10.1000/abc.123", "Title A"),
        _Src("S2", "https://doi.org/10.1000/abc.123", "Title B"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1
    assert groups[0].key.startswith("doi:")


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------


def test_same_domain_same_title_clusters():
    sources = [
        _Src("S1", "https://blog.foo.com/a", "Ten Tips for LangGraph Supervisors"),
        _Src("S2", "https://blog.foo.com/mirror-page", "Ten Tips for LangGraph Supervisors"),
        _Src("S3", "https://blog.foo.com/unrelated", "How to deploy a k8s cluster"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1
    assert len(groups[0].sources) == 2


def test_cross_domain_same_title_does_not_cluster():
    """False positives avoidance — two blogs with the same viral title should
    NOT be treated as mirrors unless other evidence (arxiv/doi) supports it."""
    sources = [
        _Src("S1", "https://medium.com/@x/post-a", "Understanding LangGraph Supervisor"),
        _Src("S2", "https://dev.to/y/post-b", "Understanding LangGraph Supervisor"),
    ]
    assert detect_mirror_groups(sources) == []


def test_arxiv_cross_mirror_title_grouped():
    """Two arxiv URLs that both miss the ID regex but share a title still
    cluster — we trust the arxiv domain hint."""
    sources = [
        _Src("S1", "https://arxiv.org/html/garbled-path", "A Totally Legitimate Paper Title"),
        _Src("S2", "https://arxiv.org/html/another-garbled-path", "A Totally Legitimate Paper Title"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1


def test_mixed_title_and_arxiv_coexists():
    """arxiv clusters + unrelated domain-only clusters coexist."""
    sources = [
        _Src("S1", "https://arxiv.org/abs/2310.05193", "Paper A"),
        _Src("S2", "https://arxiv.org/html/2310.05193", "Paper A"),
        _Src("S3", "https://blog.foo.com/a", "Ten Ways to X"),
        _Src("S4", "https://blog.foo.com/b", "Ten Ways to X"),
        _Src("S5", "https://unrelated.com", "Completely Different Title"),
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 2
    assert {g.key for g in groups if g.key.startswith("arxiv:")}
    assert any(len(g.sources) == 2 for g in groups)


# ---------------------------------------------------------------------------
# Input shape tolerance
# ---------------------------------------------------------------------------


def test_accepts_dicts():
    sources = [
        {"source_id": "S1", "url": "https://arxiv.org/abs/1234.5678", "title": "t"},
        {"source_id": "S2", "url": "https://arxiv.org/pdf/1234.5678", "title": "t"},
    ]
    groups = detect_mirror_groups(sources)
    assert len(groups) == 1


def test_empty_input_returns_empty():
    assert detect_mirror_groups([]) == []


def test_single_source_no_group():
    sources = [_Src("S1", "https://arxiv.org/abs/1234.5678", "solo")]
    assert detect_mirror_groups(sources) == []


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_format_empty_is_empty():
    assert format_mirror_warnings([]) == ""


def test_format_includes_group_members():
    groups = [
        MirrorGroup(
            key="arxiv:2310.05193",
            sources=[
                _Src("S1", "https://arxiv.org/abs/2310.05193", "P"),
                _Src("S2", "https://arxiv.org/html/2310.05193", "P"),
            ],
        )
    ]
    out = format_mirror_warnings(groups)
    assert "arxiv:2310.05193" in out
    assert "S1" in out and "S2" in out
    assert "independent verification" in out
