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
# _extract_sq_descriptions - pure function, no LLM
# ---------------------------------------------------------------------------

SAMPLE_CHECKLIST = """# Coverage Checklist


## Q1: market overview and initial screening
- [ ] advocate - not_started
- [ ] critic - not_started

## Q2: Chinese (Taiwan) transcription accuracy and speaker diarization assessment
- [ ] advocate - not_started
- [ ] critic - not_started

## Q3: UI, operating experience, and editing feature comparison
- [x] advocate - done

## Q4: pricing model and value analysis
- [ ] advocate - not_started
"""


def test_extract_sq_descriptions_parses_correctly():
    desc = _extract_sq_descriptions(SAMPLE_CHECKLIST)
    assert desc["Q1"] == "market overview and initial screening"
    assert desc["Q2"] == "Chinese (Taiwan) transcription accuracy and speaker diarization assessment"
    assert desc["Q3"] == "UI, operating experience, and editing feature comparison"
    assert desc["Q4"] == "pricing model and value analysis"


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
    text = "## Q1: topic one\n\nSome text mentioning Q1 again.\n\n## Q2: topic two"
    desc = _extract_sq_descriptions(text)
    assert set(desc.keys()) == {"Q1", "Q2"}


# ---------------------------------------------------------------------------
# _run_relevance_checks - no workspace / no coverage.chk -> returns empty set
# ---------------------------------------------------------------------------

def test_run_relevance_checks_no_workspace():
    """Empty workspace path -> skip relevance check -> return empty set."""
    claims = [
        _approved_claim("Q1-C1", "Q1", "Whisper Large v3 WER is 8.3%"),
    ]
    result = asyncio.run(
        _run_relevance_checks(claims, "")
    )
    assert result == set(), "No workspace -> conservative, no rejections"


def test_run_relevance_checks_no_approved_claims():
    """All claims still pending -> nothing to check -> empty set."""
    claims = [
        _pending_claim("Q1-C1", "Q1", "some text"),
        _rejected_claim("Q1-C2", "Q1", "other text"),
    ]
    result = asyncio.run(
        _run_relevance_checks(claims, "")
    )
    assert result == set()


# ---------------------------------------------------------------------------
# quality_eval_node - relevance dimension in dim_scores
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
        _approved_claim("Q1-C1", "Q1", "Whisper WER is 8.3%"),
        _approved_claim("Q1-C2", "Q1", "MacWhisper supports local models"),
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
    """If relevance check rejects a claim, dim_scores['relevance'] must be False.

    P1-2 three-stage gate requires ≥3 approved and ≥50% flagged before
    rejection kicks in; use 4 claims with 2 flagged to cross both thresholds.
    """
    from deep_research.nodes.phase1b import quality_eval_node
    from deep_research.state import VerifyState

    async def mock_relevance_checks(claims, workspace):
        return {"Q1-C3", "Q1-C4"}  # 2 of 4 off-topic → ratio=0.5 ≥ threshold

    monkeypatch.setattr(
        "deep_research.nodes.phase1b._run_relevance_checks",
        mock_relevance_checks,
    )

    claims = [
        _approved_claim("Q1-C1", "Q1", "Whisper WER is 8.3%"),
        _approved_claim("Q1-C2", "Q1", "MacWhisper supports local models"),
        _approved_claim("Q1-C3", "Q1", "company address: 1 Xinyi Road, Taipei"),
        _approved_claim("Q1-C4", "Q1", "CEO bio says founded in 1998"),
    ]
    grounding = [
        {"claim_id": "Q1-C1", "score": 0.9, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C2", "score": 0.85, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C3", "score": 0.8, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C4", "score": 0.8, "verdict": "GROUNDED"},
    ]

    state: VerifyState = {
        "claims_to_verify": claims,
        "grounding_results": grounding,
        "workspace_path": "",
    }

    result = asyncio.run(quality_eval_node(state))
    scores = result["quality_scores"]
    updated_claims = result["claims_to_verify"]

    # Q1-C3 / Q1-C4 should be rejected
    c3 = next(c for c in updated_claims if c.claim_id == "Q1-C3")
    c4 = next(c for c in updated_claims if c.claim_id == "Q1-C4")
    assert c3.status == "rejected", "Off-topic claim must be rejected"
    assert c4.status == "rejected", "Off-topic claim must be rejected"

    # Q1-C1 / Q1-C2 should remain approved
    c1 = next(c for c in updated_claims if c.claim_id == "Q1-C1")
    c2 = next(c for c in updated_claims if c.claim_id == "Q1-C2")
    assert c1.status == "approved"
    assert c2.status == "approved"

    # relevance dimension should be False (had off-topic claims)
    assert scores["Q1"]["relevance"] is False


def test_quality_eval_offtopic_reduces_actionability(monkeypatch):
    """If all claims for a SQ are off-topic -> actionability=False -> needs_attack.

    P1-2 three-stage gate requires ≥3 approved claims before rejection kicks
    in; use 3 claims all flagged so gate triggers and all get rejected.
    """
    from deep_research.nodes.phase1b import quality_eval_node
    from deep_research.state import VerifyState

    async def mock_relevance_checks(claims, workspace):
        return {"Q1-C1", "Q1-C2", "Q1-C3"}  # all 3 off-topic

    monkeypatch.setattr(
        "deep_research.nodes.phase1b._run_relevance_checks",
        mock_relevance_checks,
    )

    claims = [
        _approved_claim("Q1-C1", "Q1", "address information A"),
        _approved_claim("Q1-C2", "Q1", "address information B"),
        _approved_claim("Q1-C3", "Q1", "address information C"),
    ]
    grounding = [
        {"claim_id": "Q1-C1", "score": 0.9, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C2", "score": 0.85, "verdict": "GROUNDED"},
        {"claim_id": "Q1-C3", "score": 0.82, "verdict": "GROUNDED"},
    ]

    state: VerifyState = {
        "claims_to_verify": claims,
        "grounding_results": grounding,
        "workspace_path": "",
    }

    result = asyncio.run(quality_eval_node(state))
    scores = result["quality_scores"]
    failed = result["failed_dimensions"]

    # All claims rejected -> actionability=False
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
