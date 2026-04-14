"""Verify phase1b emits a CRITICAL gap-log entry when grounding coverage drops.

In the 2026-04-14 failure workspace, only 17/208 claims got grounding entries but
the final report still claimed "all claims verified". We want a loud, auditable
warning the next time the invariant breaks, not silent success.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from deep_research.nodes import phase1b
from deep_research.state import Claim


def _make_claim(cid: str) -> Claim:
    return Claim(
        claim_id=cid,
        claim_text=f"claim {cid}",
        claim_type="qualitative",
        quote_ids=[f"Q{cid}"],
        source_id="S001",
        subquestion="Q1",
    )


async def _run_phase1b(claims, grounding_results, workspace):
    """Invoke phase1b_verify with the subgraph stubbed out — we only care about
    the post-subgraph bookkeeping (gap-log assertion, ledger write, etc.)."""
    async def fake_ainvoke(state):
        return {
            "claims_to_verify": claims,
            "grounding_results": grounding_results,
            "quality_scores": {},
            "failed_dimensions": [],
        }

    with patch.object(phase1b, "build_verify_subgraph") as fake_build:
        fake_build.return_value.ainvoke = AsyncMock(side_effect=fake_ainvoke)
        with patch.object(phase1b, "_write_claim_ledger"):
            return await phase1b.phase1b_verify({
                "claims": claims,
                "workspace_path": str(workspace),
            })


@pytest.mark.anyio
async def test_critical_written_when_coverage_below_90_percent(tmp_path: Path):
    claims = [_make_claim(str(i)) for i in range(10)]
    grounding = [{"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"} for c in claims[:3]]

    await _run_phase1b(claims, grounding, tmp_path)

    gap = (tmp_path / "gap-log.md").read_text()
    assert "CRITICAL — Phase 1b grounding coverage degraded" in gap
    assert "3/10" in gap
    assert "below 90% floor" in gap


@pytest.mark.anyio
async def test_no_critical_when_coverage_at_or_above_90_percent(tmp_path: Path):
    claims = [_make_claim(str(i)) for i in range(10)]
    # 9/10 = 90% exactly — must not trip the floor
    grounding = [{"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"} for c in claims[:9]]

    await _run_phase1b(claims, grounding, tmp_path)

    gap_path = tmp_path / "gap-log.md"
    if gap_path.exists():
        assert "grounding coverage degraded" not in gap_path.read_text()


@pytest.mark.anyio
async def test_empty_claims_no_assertion(tmp_path: Path):
    """No claims means no work to verify — the floor rule must not trip on 0/0."""
    await _run_phase1b([], [], tmp_path)
    gap_path = tmp_path / "gap-log.md"
    if gap_path.exists():
        assert "grounding coverage degraded" not in gap_path.read_text()


@pytest.fixture
def anyio_backend():
    return "asyncio"
