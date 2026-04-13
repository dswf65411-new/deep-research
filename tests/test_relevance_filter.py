"""Tests for Phase 1b relevance check (P1-D LLM-based).

Verifies:
- _extract_sq_descriptions parses coverage.chk correctly
- _run_relevance_checks skips when no SQ descriptions available
- quality_eval_node dim_scores include "relevance" dimension
- irrelevant_ids set is correctly propagated

LLM calls (_batch_relevance_check) are NOT unit-tested here since they require
a live LLM. Integration is tested via quality_eval_node with a mock that skips
the actual LLM call.
"""

import asyncio
import pytest

from deep_research.nodes.phase1b import (
    _extract_sq_descriptions,
    _run_relevance_checks,
)
from deep_research.state import Claim


# ---------------------------------------------------------------------------
# _extract_sq_descriptions — pure function, no LLM
# ---------------------------------------------------------------------------

SAMPLE_CHECKLIST = """# Coverage Checklist


## Q1: 市場概覽與初步篩選
- [ ] advocate — not_started
- [ ] critic — not_started

## Q2: 中文（台灣）轉譯準確性與說話者分離能力評估
- [ ] advocate — not_started
- [ ] critic — not_started

## Q3: 使用者介面、操作體驗與編輯功能比較
- [x] advocate — done

## Q4: 定價模式與性價比分析
- [ ] advocate — not_started
"""


def test_extract_sq_descriptions_parses_correctly():
    desc = _extract_sq_descriptions(SAMPLE_CHECKLIST)
    assert desc["Q1"] == "市場概覽與初步篩選"
    assert desc["Q2"] == "中文（台灣）轉譯準確性與說話者分離能力評估"
    assert desc["Q3"] == "使用者介面、操作體驗與編輯功能比較"
    assert desc["Q4"] == "定價模式與性價比分析"


def test_extract_sq_descriptions_empty_text():
    assert _extract_sq_descriptions("") == {}
    assert _extract_sq_descriptions(None) == {}  # type: ignore[arg-type]


def test_extract_sq_descriptions_no_match():
    text = "# Just a header\n## Not a Q: something else"
    desc = _extract_sq_descriptions(text)
    assert desc == {}


def test_extract_sq_descriptions_deduplicates():
    """First occurrence wins if Q1 appears twice."""
    text = "## Q1: First\n## Q1: Second\n## Q2: Other"
    desc = _extract_sq_descriptions(text)
    assert desc["Q1"] == "First"
    assert desc["Q2"] == "Other"


def test_extract_sq_descriptions_handles_mixed_format():
    """Only ## Q{n}: format matched; inline Q1 text not captured."""
    text = "## Q1: 主題一\n\nSome text mentioning Q1 again.\n\n## Q2: 主題二"
    desc = _extract_sq_descriptions(text)
    assert set(desc.keys()) == {"Q1", "Q2"}


# ---------------------------------------------------------------------------
# _run_relevance_checks — no workspace / no coverage.chk → returns empty set
# ---------------------------------------------------------------------------

def test_run_relevance_checks_no_workspace():
    """Empty workspace path → skip relevance check → return empty set."""
    claims = [
        _approved_claim("Q1-C1", "Q1", "Whisper Large v3 WER 為 8.3%"),
    ]
    result = asyncio.run(
        _run_relevance_checks(claims, "")
    )
    assert result == set(), "No workspace → conservative, no rejections"


def test_run_relevance_checks_no_approved_claims():
    """All claims still pending → nothing to check → empty set."""
    claims = [
        _pending_claim("Q1-C1", "Q1", "some text"),
        _rejected_claim("Q1-C2", "Q1", "other text"),
    ]
    result = asyncio.run(
        _run_relevance_checks(claims, "")
    )
    assert result == set()


# ---------------------------------------------------------------------------
# quality_eval_node — relevance dimension in dim_scores
# Note: this test mocks _run_relevance_checks to avoid LLM calls.
# ---------------------------------------------------------------------------

def test_quality_eval_relevance_dim_in_scores(monkeypatch):
    """dim_scores must contain 'relevance' key after quality_eval_node runs."""
    from deep_research.nodes.phase1b import quality_eval_node
    from deep_research.state import VerifyState

    # Patch _run_relevance_checks to return empty set (no off-topic claims)
    async def mock_relevance_checks(claims, workspace):
        return set()

    monkeypatch.setattr(
        "deep_research.nodes.phase1b._run_relevance_checks",
        mock_relevance_checks,
    )

    claims = [
        _approved_claim("Q1-C1", "Q1", "Whisper WER 為 8.3%"),
        _approved_claim("Q1-C2", "Q1", "MacWhisper 支援本機端模型"),
    ]
    grounding = [
        {"claim_id": "Q1-C1", "score": 0.9, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C2", "score": 0.85, "verdict": "GROUNDED"},
    ]

    state: VerifyState = {
        "claims_to_verify": claims,
        "grounding_results": grounding,
        "workspace_path": "",
    }

    result = asyncio.run(quality_eval_node(state))
    scores = result["quality_scores"]
    assert "Q1" in scores
    assert "relevance" in scores["Q1"], "'relevance' must appear in dim_scores"
    assert scores["Q1"]["relevance"] is True  # no off-topic claims


def test_quality_eval_relevance_false_when_offtopic(monkeypatch):
    """If relevance check rejects a claim, dim_scores['relevance'] must be False."""
    from deep_research.nodes.phase1b import quality_eval_node
    from deep_research.state import VerifyState

    # Simulate one off-topic claim detected
    async def mock_relevance_checks(claims, workspace):
        return {"Q1-C2"}  # Q1-C2 is off-topic

    monkeypatch.setattr(
        "deep_research.nodes.phase1b._run_relevance_checks",
        mock_relevance_checks,
    )

    claims = [
        _approved_claim("Q1-C1", "Q1", "Whisper WER 為 8.3%"),
        _approved_claim("Q1-C2", "Q1", "公司地址: 台北市信義路 1 號"),
    ]
    grounding = [
        {"claim_id": "Q1-C1", "score": 0.9, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C2", "score": 0.8, "verdict": "GROUNDED"},
    ]

    state: VerifyState = {
        "claims_to_verify": claims,
        "grounding_results": grounding,
        "workspace_path": "",
    }

    result = asyncio.run(quality_eval_node(state))
    scores = result["quality_scores"]
    updated_claims = result["claims_to_verify"]

    # Q1-C2 should be rejected
    c2 = next(c for c in updated_claims if c.claim_id == "Q1-C2")
    assert c2.status == "rejected", "Off-topic claim must be rejected"

    # Q1-C1 should remain approved
    c1 = next(c for c in updated_claims if c.claim_id == "Q1-C1")
    assert c1.status == "approved"

    # relevance dimension should be False (had off-topic claims)
    assert scores["Q1"]["relevance"] is False


def test_quality_eval_offtopic_reduces_actionability(monkeypatch):
    """If all claims for a SQ are off-topic → actionability=False → needs_attack."""
    from deep_research.nodes.phase1b import quality_eval_node
    from deep_research.state import VerifyState

    async def mock_relevance_checks(claims, workspace):
        return {"Q1-C1", "Q1-C2"}  # both off-topic

    monkeypatch.setattr(
        "deep_research.nodes.phase1b._run_relevance_checks",
        mock_relevance_checks,
    )

    claims = [
        _approved_claim("Q1-C1", "Q1", "地址資訊 A"),
        _approved_claim("Q1-C2", "Q1", "地址資訊 B"),
    ]
    grounding = [
        {"claim_id": "Q1-C1", "score": 0.9, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C2", "score": 0.85, "verdict": "GROUNDED"},
    ]

    state: VerifyState = {
        "claims_to_verify": claims,
        "grounding_results": grounding,
        "workspace_path": "",
    }

    result = asyncio.run(quality_eval_node(state))
    scores = result["quality_scores"]
    failed = result["failed_dimensions"]

    # All claims rejected → actionability=False
    assert scores["Q1"]["actionability"] is False
    assert "actionability" in failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved_claim(claim_id: str, subq: str, text: str) -> Claim:
    c = Claim(claim_id=claim_id, claim_text=text, source_ids=["S001"])
    c.status = "approved"
    c.subquestion = subq
    c.bedrock_score = 0.8
    c.quote_ids = ["S001-Q1"]
    return c


def _pending_claim(claim_id: str, subq: str, text: str) -> Claim:
    c = Claim(claim_id=claim_id, claim_text=text, source_ids=["S001"])
    c.status = "pending"
    c.subquestion = subq
    return c


def _rejected_claim(claim_id: str, subq: str, text: str) -> Claim:
    c = Claim(claim_id=claim_id, claim_text=text, source_ids=["S001"])
    c.status = "rejected"
    c.subquestion = subq
    return c
