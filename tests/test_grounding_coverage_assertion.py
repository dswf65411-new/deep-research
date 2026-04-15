"""Verify phase1b enforces the ≥90% grounding-coverage invariant (Whisper P2-3).

In the 2026-04-14 failure workspace, only 17/208 claims got grounding entries
but the final report still claimed "all claims verified". The fix has two
parts:
    1. Backfill a ``MISSING`` stub for every claim ``to_verify`` that never
       landed a grounding row, so downstream audit can never again mistake
       an absent row for a silent pass.
    2. If coverage is *still* below 90% after backfill (shouldn't happen,
       but kept as defense-in-depth), write a CRITICAL entry into gap-log.
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
async def test_missing_claim_ids_get_backfilled(tmp_path: Path):
    """The core P2-3 invariant: every ``to_verify`` claim gets a row in the
    final grounding results, even if the subgraph forgot it."""
    claims = [_make_claim(str(i)) for i in range(10)]
    grounding = [
        {"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"}
        for c in claims[:3]
    ]

    await _run_phase1b(claims, grounding, tmp_path)

    # grounding-results/latest.md must now list all 10 claims
    md = (tmp_path / "grounding-results" / "latest.md").read_text()
    for c in claims:
        assert c.claim_id in md
    # The 7 missing ones must be marked MISSING, not silently "verified"
    missing_lines = [line for line in md.splitlines() if "verdict=MISSING" in line]
    assert len(missing_lines) == 7


@pytest.mark.anyio
async def test_backfill_restores_coverage_above_floor(tmp_path: Path):
    """After backfill, coverage should hit 100% → no CRITICAL banner needed."""
    claims = [_make_claim(str(i)) for i in range(10)]
    grounding = [
        {"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"}
        for c in claims[:3]
    ]

    await _run_phase1b(claims, grounding, tmp_path)

    gap_path = tmp_path / "gap-log.md"
    # Defense-in-depth banner should NOT fire because backfill restored 100%.
    if gap_path.exists():
        assert "grounding coverage degraded" not in gap_path.read_text()


@pytest.mark.anyio
async def test_no_critical_when_coverage_at_or_above_90_percent(tmp_path: Path):
    claims = [_make_claim(str(i)) for i in range(10)]
    # 9/10 already covered — only 1 backfill needed
    grounding = [
        {"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"}
        for c in claims[:9]
    ]

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


@pytest.mark.anyio
async def test_all_already_covered_no_backfill(tmp_path: Path):
    """When every claim already has a grounding entry, backfill is a no-op."""
    claims = [_make_claim(str(i)) for i in range(5)]
    grounding = [
        {"claim_id": c.claim_id, "score": 0.9, "verdict": "GROUNDED"}
        for c in claims
    ]

    await _run_phase1b(claims, grounding, tmp_path)

    md = (tmp_path / "grounding-results" / "latest.md").read_text()
    # No MISSING verdicts should exist
    assert "verdict=MISSING" not in md


@pytest.fixture
def anyio_backend():
    return "asyncio"
