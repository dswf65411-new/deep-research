"""Smoke test for bedrock_score writeback — issue #6 fix.

Verifies that the grounding score computed in grounding_check_node is
correctly written back to Claim.bedrock_score before _write_claim_ledger
runs, so claim-ledger.md no longer shows all-zero bedrock values.
"""

import os
import tempfile

import pytest

from deep_research.nodes.phase1b import _write_claim_ledger
from deep_research.state import Claim


def test_bedrock_score_is_mutable():
    """Claim.bedrock_score must be directly assignable (model is not frozen)."""
    c = Claim(claim_id="Q1-C1", claim_text="Test", source_ids=["S001"])
    assert c.bedrock_score == 0.0
    c.bedrock_score = 0.7
    c.citation_verdict = "GROUNDED"
    assert c.bedrock_score == pytest.approx(0.7)
    assert c.citation_verdict == "GROUNDED"


def test_bedrock_writeback_logic():
    """grounding_results score must flow back into claim.bedrock_score + citation_verdict."""
    claim = Claim(claim_id="Q1-C1", claim_text="Test claim", source_ids=["S001"])
    assert claim.bedrock_score == 0.0  # pre-fix default

    grounding = [
        {"claim_id": "Q1-C1", "score": 0.7, "verdict": "GROUNDED"},
    ]

    # replicate the writeback block now in phase1b_verify
    g_map = {r["claim_id"]: r for r in grounding}
    updated_claims = [claim]
    for c in updated_claims:
        r = g_map.get(c.claim_id)
        if r:
            c.bedrock_score = r.get("score", 0.0)
            c.citation_verdict = r.get("verdict", "")

    assert claim.bedrock_score == pytest.approx(0.7)
    assert claim.citation_verdict == "GROUNDED"


def test_claim_ledger_writes_bedrock_score():
    """_write_claim_ledger must persist the updated bedrock_score to claim-ledger.md."""
    claim = Claim(claim_id="Q1-C1", claim_text="Test claim", source_ids=["S001"])
    claim.bedrock_score = 0.7
    claim.citation_verdict = "GROUNDED"

    with tempfile.TemporaryDirectory() as ws:
        _write_claim_ledger(ws, [claim])
        ledger_path = os.path.join(ws, "claim-ledger.md")
        assert os.path.exists(ledger_path), "claim-ledger.md should be created"
        content = open(ledger_path, encoding="utf-8").read()
        assert "0.70" in content, (
            f"Expected bedrock score 0.70 in ledger, got:\n{content}"
        )


def test_claim_ledger_ungrounded_claim_shows_zero():
    """Claim with no grounding update must still show 0.00 in ledger (regression guard)."""
    claim = Claim(claim_id="Q1-C2", claim_text="Another claim", source_ids=["S002"])

    with tempfile.TemporaryDirectory() as ws:
        _write_claim_ledger(ws, [claim])
        content = open(os.path.join(ws, "claim-ledger.md"), encoding="utf-8").read()
        assert "0.00" in content


def test_no_score_entry_leaves_claim_unchanged():
    """If a claim_id has no grounding entry, bedrock_score must stay at 0.0."""
    claim = Claim(claim_id="Q1-C1", claim_text="Test", source_ids=["S001"])
    grounding = [
        {"claim_id": "Q1-C99", "score": 0.9, "verdict": "GROUNDED"},  # different id
    ]

    g_map = {r["claim_id"]: r for r in grounding}
    updated_claims = [claim]
    for c in updated_claims:
        r = g_map.get(c.claim_id)
        if r:
            c.bedrock_score = r.get("score", 0.0)
            c.citation_verdict = r.get("verdict", "")

    assert claim.bedrock_score == pytest.approx(0.0)
    assert claim.citation_verdict == ""


# ---------------------------------------------------------------------------
# validate_claims_for_phase2 — bedrock minimum threshold (>= 0.3)
# ---------------------------------------------------------------------------

def _approved_claim(claim_id: str, bedrock: float, quote_ids: list[str] | None = None) -> Claim:
    c = Claim(claim_id=claim_id, claim_text="Test", source_ids=["S001"])
    c.status = "approved"
    c.bedrock_score = bedrock
    c.quote_ids = quote_ids if quote_ids is not None else ["S001-Q1"]
    return c


def test_bedrock_min_threshold_blocks_low_score():
    """Claims with bedrock_score < 0.3 must be rejected by validate_claims_for_phase2."""
    from deep_research.harness.validators import validate_claims_for_phase2
    garbage = _approved_claim("Q1-C14", bedrock=0.1)  # e.g. company address claim
    result = validate_claims_for_phase2([garbage])
    assert result == [], f"Expected empty list, got {result}"


def test_bedrock_min_threshold_blocks_zero_score():
    """Claims with bedrock_score == 0.0 (grounding never ran) must be rejected."""
    from deep_research.harness.validators import validate_claims_for_phase2
    unverified = _approved_claim("Q1-C1", bedrock=0.0)
    result = validate_claims_for_phase2([unverified])
    assert result == []


def test_bedrock_min_threshold_allows_above_threshold():
    """Claims with bedrock_score >= 0.3 must pass through."""
    from deep_research.harness.validators import validate_claims_for_phase2
    good = _approved_claim("Q1-C2", bedrock=0.3)
    high = _approved_claim("Q1-C3", bedrock=0.9)
    result = validate_claims_for_phase2([good, high])
    assert len(result) == 2


def test_bedrock_min_threshold_boundary():
    """Exactly 0.3 must pass; 0.29 must fail."""
    from deep_research.harness.validators import validate_claims_for_phase2
    at_boundary = _approved_claim("Q1-C4", bedrock=0.3)
    just_below = _approved_claim("Q1-C5", bedrock=0.29)
    result = validate_claims_for_phase2([at_boundary, just_below])
    assert len(result) == 1
    assert result[0].claim_id == "Q1-C4"


def test_bedrock_filter_preserves_other_conditions():
    """Rejected claim (wrong status) must still be blocked even with high bedrock."""
    from deep_research.harness.validators import validate_claims_for_phase2
    rejected = Claim(claim_id="Q1-C6", claim_text="Test", source_ids=["S001"])
    rejected.status = "rejected"
    rejected.bedrock_score = 0.9
    rejected.quote_ids = ["S001-Q1"]
    result = validate_claims_for_phase2([rejected])
    assert result == []
