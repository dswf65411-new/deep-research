"""Tests for Whisper P3-3 — advocate/critic collision detector."""

from __future__ import annotations

from dataclasses import dataclass

from deep_research.harness.stakeholder_collision import (
    CollisionPair,
    collect_collisions,
    format_collision_section,
)


@dataclass
class _Src:
    source_id: str
    role: str


@dataclass
class _Claim:
    claim_id: str
    subquestion: str
    source_ids: list[str]


def test_no_claims_no_collision():
    assert collect_collisions([], []) == []


def test_only_advocate_no_collision():
    claims = [_Claim("Q1-C1", "Q1", ["S001"])]
    sources = [_Src("S001", "advocate")]
    assert collect_collisions(claims, sources) == []


def test_only_critic_no_collision():
    claims = [_Claim("Q1-C1", "Q1", ["S001"])]
    sources = [_Src("S001", "critic")]
    assert collect_collisions(claims, sources) == []


def test_same_sq_both_sides_is_collision():
    claims = [
        _Claim("Q1-C1", "Q1", ["S001"]),
        _Claim("Q1-C2", "Q1", ["S002"]),
    ]
    sources = [
        _Src("S001", "advocate"),
        _Src("S002", "critic"),
    ]
    pairs = collect_collisions(claims, sources)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.subquestion == "Q1"
    assert p.advocate_claims == ["Q1-C1"]
    assert p.critic_claims == ["Q1-C2"]
    assert p.advocate_sources == ["S001"]
    assert p.critic_sources == ["S002"]


def test_multiple_subquestions_independent():
    claims = [
        _Claim("Q1-C1", "Q1", ["S001"]),
        _Claim("Q1-C2", "Q1", ["S002"]),
        _Claim("Q2-C1", "Q2", ["S003"]),  # only advocate, no collision
    ]
    sources = [
        _Src("S001", "advocate"),
        _Src("S002", "critic"),
        _Src("S003", "advocate"),
    ]
    pairs = collect_collisions(claims, sources)
    assert len(pairs) == 1
    assert pairs[0].subquestion == "Q1"


def test_claim_with_multiple_sources_on_both_sides():
    """A single claim cited from BOTH advocate and critic sources counts as its
    own collision (the claim is internally contested)."""
    claims = [_Claim("Q1-C1", "Q1", ["S001", "S002"])]
    sources = [
        _Src("S001", "advocate"),
        _Src("S002", "critic"),
    ]
    pairs = collect_collisions(claims, sources)
    assert len(pairs) == 1
    assert pairs[0].advocate_claims == ["Q1-C1"]
    assert pairs[0].critic_claims == ["Q1-C1"]


def test_dict_claims_and_dict_sources_accepted():
    """Phase2 sometimes passes plain dicts instead of Pydantic models."""
    claims = [
        {"claim_id": "Q1-C1", "subquestion": "Q1", "source_ids": ["S001"]},
        {"claim_id": "Q1-C2", "subquestion": "Q1", "source_ids": ["S002"]},
    ]
    sources = [
        {"source_id": "S001", "role": "advocate"},
        {"source_id": "S002", "role": "critic"},
    ]
    pairs = collect_collisions(claims, sources)
    assert len(pairs) == 1


def test_missing_subquestion_is_skipped():
    claims = [
        _Claim("C1", "", ["S001"]),  # no sq
        _Claim("Q1-C2", "Q1", ["S002"]),
    ]
    sources = [
        _Src("S001", "advocate"),
        _Src("S002", "critic"),
    ]
    pairs = collect_collisions(claims, sources)
    assert pairs == []


def test_perspective_role_ignored_for_collision():
    """A 'perspective' source on its own doesn't make a collision."""
    claims = [
        _Claim("Q1-C1", "Q1", ["S001"]),
        _Claim("Q1-C2", "Q1", ["S002"]),
    ]
    sources = [
        _Src("S001", "advocate"),
        _Src("S002", "perspective"),
    ]
    assert collect_collisions(claims, sources) == []


# ---------------------------------------------------------------------------
# Gap-log section formatter
# ---------------------------------------------------------------------------


def test_format_empty_returns_empty_string():
    assert format_collision_section([]) == ""


def test_format_section_mentions_both_sides():
    pairs = [
        CollisionPair(
            subquestion="Q3",
            advocate_claims=["Q3-C1"],
            critic_claims=["Q3-C5"],
            advocate_sources=["S010"],
            critic_sources=["S020"],
        )
    ]
    out = format_collision_section(pairs)
    assert "## 未解矛盾" in out
    assert "Q3" in out
    assert "Q3-C1" in out
    assert "Q3-C5" in out
    assert "S010" in out
    assert "S020" in out


def test_format_section_sorted_by_sq():
    pairs = [
        CollisionPair("Q10", ["c"], ["d"], ["s10"], ["s20"]),
        CollisionPair("Q2", ["a"], ["b"], ["s1"], ["s2"]),
    ]
    out = format_collision_section(pairs)
    # "Q10" sorts before "Q2" lexicographically — that's intentional, we
    # just verify the order is stable.
    assert out.index("### Q10") < out.index("### Q2")
