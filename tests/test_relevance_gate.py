"""Tests for Whisper plan P1-2 — relevance three-stage gate.

The relevance filter in ``quality_eval_node`` used to reject every claim the
LLM judge flagged as off-topic. On the failed-workspace run (2026-04-14),
this had two failure modes:

1. Low-sample SQs (1–2 approved claims) would lose everything if the judge
   made one bad call, then trip ``actionability=False`` and ``completeness=
   False`` → false_dims ≥ 2 → trigger_fallback cascades.
2. High-sample SQs where the judge flagged a few edge cases would still
   pass quality, but wasted grounded claims the integrator actually needed.

The three-stage gate (Whisper plan P1-2) relaxes the first case without
changing behaviour on clear-reject cases. Per-SQ:
  - ``approved_count < RELEVANCE_MIN_APPROVED`` (3): skip rejection entirely.
  - ``rejected_ratio < RELEVANCE_REJECT_RATIO`` (0.5): skip rejection.
  - Otherwise: reject the flagged claims (unchanged).

Also verified: ``trigger_fallback`` already excludes the ``relevance``
dimension from its false_dims count (line 480/487), so relaxing the
rejection here is sufficient to stop the cascade.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_research.nodes.phase1b import (
    RELEVANCE_MIN_APPROVED,
    RELEVANCE_REJECT_RATIO,
    quality_eval_node,
)
from deep_research.state import Claim
from deep_research.tools.workspace import create_workspace


def _ws(tmp_path) -> str:
    return create_workspace("relevance-gate-test", base_dir=str(tmp_path))


def _approved(claim_id: str, subq: str, text: str, source_ids=None) -> Claim:
    c = Claim(
        claim_id=claim_id,
        claim_text=text,
        subquestion=subq,
        source_ids=source_ids or ["S001"],
        quote_ids=["QT001"],
    )
    c.status = "approved"
    c.bedrock_score = 0.9
    return c


def _grounding(claim: Claim, verdict: str = "GROUNDED") -> dict:
    return {
        "claim_id": claim.claim_id,
        "score": 0.9,
        "verdict": verdict,
        "tool": "bedrock",
    }


def _run_quality_eval(claims, workspace, irrelevant_ids):
    """Drive quality_eval_node with a canned relevance verdict."""
    import deep_research.nodes.phase1b as p1b

    async def fake_relevance(_claims, _workspace):
        return set(irrelevant_ids)

    orig = p1b._run_relevance_checks
    p1b._run_relevance_checks = fake_relevance
    try:
        state = {
            "claims_to_verify": claims,
            "grounding_results": [_grounding(c) for c in claims],
            "workspace_path": workspace,
        }
        return asyncio.run(quality_eval_node(state))
    finally:
        p1b._run_relevance_checks = orig


class TestThreeStageGate:
    def test_small_sample_skip_rejection(self, tmp_path):
        """< 3 approved claims: relevance flag must not rejct the claim."""
        ws = _ws(tmp_path)
        claims = [
            _approved("Q1-C1", "Q1", "fact one"),
            _approved("Q1-C2", "Q1", "fact two"),
        ]
        _run_quality_eval(claims, ws, irrelevant_ids={"Q1-C1"})
        # Flagged claim stays approved because sample was too small.
        assert claims[0].status == "approved"
        assert claims[1].status == "approved"

    def test_low_reject_ratio_skips(self, tmp_path):
        """>= 3 approved, but < 50% flagged: rejections are skipped."""
        ws = _ws(tmp_path)
        claims = [_approved(f"Q1-C{i}", "Q1", f"fact {i}") for i in range(1, 6)]
        # 1 out of 5 flagged = 20% → below threshold, don't reject.
        _run_quality_eval(claims, ws, irrelevant_ids={"Q1-C1"})
        assert all(c.status == "approved" for c in claims)

    def test_high_reject_ratio_rejects(self, tmp_path):
        """>= 3 approved AND >= 50% flagged: rejections apply."""
        ws = _ws(tmp_path)
        claims = [_approved(f"Q1-C{i}", "Q1", f"fact {i}") for i in range(1, 5)]
        flagged = {"Q1-C1", "Q1-C2", "Q1-C3"}  # 3/4 = 75% → reject
        _run_quality_eval(claims, ws, irrelevant_ids=flagged)
        assert claims[0].status == "rejected"
        assert claims[1].status == "rejected"
        assert claims[2].status == "rejected"
        assert claims[3].status == "approved"

    def test_per_sq_gating_independent(self, tmp_path):
        """Gate is applied per-subquestion: low-sample SQ protected even when
        another SQ has enough samples to trigger rejection."""
        ws = _ws(tmp_path)
        # Q1 has 2 approved (below min) — flagging must not reject.
        # Q2 has 4 approved, 3 flagged (75%) — must reject.
        claims = [
            _approved("Q1-C1", "Q1", "q1 fact one"),
            _approved("Q1-C2", "Q1", "q1 fact two"),
            _approved("Q2-C1", "Q2", "q2 fact one"),
            _approved("Q2-C2", "Q2", "q2 fact two"),
            _approved("Q2-C3", "Q2", "q2 fact three"),
            _approved("Q2-C4", "Q2", "q2 fact four"),
        ]
        flagged = {"Q1-C1", "Q2-C1", "Q2-C2", "Q2-C3"}
        _run_quality_eval(claims, ws, irrelevant_ids=flagged)

        # Q1 flagged claim protected by min-approved gate.
        assert claims[0].status == "approved"
        assert claims[1].status == "approved"
        # Q2 rejections applied (75% ratio).
        assert claims[2].status == "rejected"
        assert claims[3].status == "rejected"
        assert claims[4].status == "rejected"
        assert claims[5].status == "approved"

    def test_thresholds_are_module_level(self):
        """The two thresholds must be module-level constants so production
        code can tune them without editing the function body."""
        assert RELEVANCE_MIN_APPROVED == 3
        assert RELEVANCE_REJECT_RATIO == 0.5


class TestQualityGateStillIgnoresRelevance:
    def test_quality_gate_excludes_relevance_dim(self):
        """Sanity check: the gate must still exclude ``relevance`` from the
        4 primary dims, otherwise relaxing per-SQ rejection above doesn't
        help — a single flagged claim would still make false_dims cascade."""
        import inspect
        import deep_research.nodes.phase1b as p1b

        src = inspect.getsource(p1b.quality_eval_node)
        # The quality_gate call filters the dims dict, dropping "relevance".
        assert 'k != "relevance"' in src
