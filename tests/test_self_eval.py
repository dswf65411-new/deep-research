"""Tests for Whisper P3-4 — pipeline self-eval + follow-up suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deep_research.harness.self_eval import (
    SqOutcome,
    format_self_eval_section,
    generate_follow_ups,
    score_subquestions,
)


@dataclass
class _Claim:
    claim_id: str
    subquestion: str
    status: str
    source_ids: list[str]


# ---------------------------------------------------------------------------
# score_subquestions
# ---------------------------------------------------------------------------


def test_empty_inputs_empty_outcomes():
    assert score_subquestions([], []) == []


def test_single_covered_sq():
    claims = [
        _Claim("Q1-C1", "Q1", "approved", ["S001"]),
        _Claim("Q1-C2", "Q1", "approved", ["S001"]),
        _Claim("Q1-C3", "Q1", "approved", ["S002"]),
    ]
    outcomes = score_subquestions(claims)
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.sq_id == "Q1"
    assert o.status == "covered"
    assert o.approved_count == 3
    assert o.rejected_count == 0
    # S001 has 2 claims → should be first top source
    assert o.top_sources[0] == "S001"


def test_thin_sq_under_threshold():
    claims = [_Claim("Q1-C1", "Q1", "approved", ["S001"])]
    outcomes = score_subquestions(claims)
    assert len(outcomes) == 1
    assert outcomes[0].status == "thin"


def test_sq_with_only_rejected_is_thin():
    claims = [
        _Claim("Q5-C1", "Q5", "rejected", ["S001"]),
        _Claim("Q5-C2", "Q5", "rejected", ["S002"]),
    ]
    outcomes = score_subquestions(claims)
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.status == "thin"
    assert o.approved_count == 0
    assert o.rejected_count == 2


def test_blocker_marks_sq_blocked():
    claims = [_Claim("Q3-C1", "Q3", "approved", ["S001"])]
    blockers = [
        "[BLOCKER] Q3 relevance dim failed three consecutive rounds",
    ]
    outcomes = score_subquestions(claims, blockers=blockers)
    q3 = next(o for o in outcomes if o.sq_id == "Q3")
    assert q3.status == "blocked"


def test_coverage_list_surfaces_zero_coverage_sqs():
    """A planned SQ with no claims at all still appears as ``thin``."""
    claims: list = []
    outcomes = score_subquestions(claims, coverage_sqs=["Q1", "Q2"])
    assert len(outcomes) == 2
    assert all(o.status == "thin" for o in outcomes)
    assert all(o.approved_count == 0 for o in outcomes)


def test_sort_order_is_numeric():
    """Q2 before Q10, not lexicographic."""
    claims = [
        _Claim("Q10-C1", "Q10", "approved", ["S1"]),
        _Claim("Q10-C2", "Q10", "approved", ["S1"]),
        _Claim("Q10-C3", "Q10", "approved", ["S1"]),
        _Claim("Q2-C1", "Q2", "approved", ["S1"]),
        _Claim("Q2-C2", "Q2", "approved", ["S1"]),
        _Claim("Q2-C3", "Q2", "approved", ["S1"]),
    ]
    outcomes = score_subquestions(claims)
    assert [o.sq_id for o in outcomes] == ["Q2", "Q10"]


def test_dict_inputs_accepted():
    claims = [
        {"claim_id": "Q1-C1", "subquestion": "Q1", "status": "approved", "source_ids": ["S1"]},
        {"claim_id": "Q1-C2", "subquestion": "Q1", "status": "approved", "source_ids": ["S1"]},
        {"claim_id": "Q1-C3", "subquestion": "Q1", "status": "approved", "source_ids": ["S2"]},
    ]
    outcomes = score_subquestions(claims)
    assert len(outcomes) == 1
    assert outcomes[0].status == "covered"


# ---------------------------------------------------------------------------
# format_self_eval_section
# ---------------------------------------------------------------------------


def test_format_empty_outcomes_note_only():
    out = format_self_eval_section([])
    assert "no sub-questions" in out.lower()
    assert out.startswith("## 本次研究自評")


def test_format_includes_tally_and_table():
    outcomes = [
        SqOutcome("Q1", "covered", approved_count=5, rejected_count=1, notes="ok"),
        SqOutcome("Q2", "thin", approved_count=1, rejected_count=0, notes="low"),
        SqOutcome("Q3", "blocked", approved_count=0, rejected_count=0, notes="blocker"),
    ]
    out = format_self_eval_section(outcomes, follow_ups=["Follow path X", "Check Y", "Probe Z"])
    assert "covered=1" in out
    assert "thin=1" in out
    assert "blocked=1" in out
    # Table header
    assert "| SQ |" in out
    # Follow-ups
    assert "1. Follow path X" in out
    assert "3. Probe Z" in out


def test_format_caps_follow_ups_at_five():
    outcomes = [SqOutcome("Q1", "covered", 3, 0, notes="ok")]
    follow = [f"idea {i}" for i in range(1, 12)]
    out = format_self_eval_section(outcomes, follow_ups=follow)
    assert "5. idea 5" in out
    assert "6. idea 6" not in out


# ---------------------------------------------------------------------------
# generate_follow_ups
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_generate_follow_ups_happy_path():
    async def fake_invoker(messages):
        return SimpleNamespace(content='{"follow_ups": ["a", "b", "c"]}')

    result = await generate_follow_ups(
        brief_text="some brief",
        outcomes=[SqOutcome("Q1", "covered", 5)],
        invoker=fake_invoker,
    )
    assert result == ["a", "b", "c"]


@pytest.mark.anyio
async def test_generate_follow_ups_strips_markdown_fence():
    async def fake_invoker(messages):
        return SimpleNamespace(
            content='```json\n{"follow_ups": ["x", "y"]}\n```'
        )

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert result == ["x", "y"]


@pytest.mark.anyio
async def test_generate_follow_ups_caps_at_five():
    async def fake_invoker(messages):
        return SimpleNamespace(content='{"follow_ups": ["1","2","3","4","5","6","7"]}')

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert len(result) == 5


@pytest.mark.anyio
async def test_generate_follow_ups_handles_malformed_json():
    async def fake_invoker(messages):
        return SimpleNamespace(content='not json at all')

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert result == []


@pytest.mark.anyio
async def test_generate_follow_ups_handles_exception():
    async def fake_invoker(messages):
        raise RuntimeError("llm unavailable")

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert result == []


@pytest.mark.anyio
async def test_generate_follow_ups_empty_response():
    async def fake_invoker(messages):
        return SimpleNamespace(content="")

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert result == []


@pytest.mark.anyio
async def test_generate_follow_ups_ignores_non_string_items():
    async def fake_invoker(messages):
        return SimpleNamespace(
            content='{"follow_ups": ["ok", 42, null, "", "also-ok"]}'
        )

    result = await generate_follow_ups("", [], invoker=fake_invoker)
    assert result == ["ok", "also-ok"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
