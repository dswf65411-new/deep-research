"""Tests for Whisper X-4 — parallel grounding check concurrency.

Verifies grounding_check_node runs LLM grounding concurrently (up to the
configured semaphore limit) and respects the DEEP_RESEARCH_GROUNDING_CONCURRENCY
environment variable.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deep_research.nodes import phase1b


# ---------------------------------------------------------------------------
# _grounding_concurrency
# ---------------------------------------------------------------------------


def test_grounding_concurrency_default():
    """Default is 5 when env var is unset."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DEEP_RESEARCH_GROUNDING_CONCURRENCY", None)
        assert phase1b._grounding_concurrency() == 5


def test_grounding_concurrency_honours_env():
    with patch.dict(os.environ, {"DEEP_RESEARCH_GROUNDING_CONCURRENCY": "10"}):
        assert phase1b._grounding_concurrency() == 10


def test_grounding_concurrency_clamps_low():
    with patch.dict(os.environ, {"DEEP_RESEARCH_GROUNDING_CONCURRENCY": "0"}):
        assert phase1b._grounding_concurrency() == 1
    with patch.dict(os.environ, {"DEEP_RESEARCH_GROUNDING_CONCURRENCY": "-3"}):
        assert phase1b._grounding_concurrency() == 1


def test_grounding_concurrency_clamps_high():
    """Cap at 32 so a misconfigured env can't spawn 1000 concurrent LLM calls."""
    with patch.dict(os.environ, {"DEEP_RESEARCH_GROUNDING_CONCURRENCY": "9999"}):
        assert phase1b._grounding_concurrency() == 32


def test_grounding_concurrency_falls_back_on_garbage():
    with patch.dict(os.environ, {"DEEP_RESEARCH_GROUNDING_CONCURRENCY": "not-an-int"}):
        assert phase1b._grounding_concurrency() == 5


# ---------------------------------------------------------------------------
# grounding_check_node — concurrency behaviour
# ---------------------------------------------------------------------------


def _make_claim(claim_id: str, text: str = "test claim", kind: str = "qualitative"):
    return SimpleNamespace(
        claim_id=claim_id,
        claim_text=text,
        claim_type=kind,
    )


def test_grounding_runs_concurrently_under_semaphore(monkeypatch):
    """With 10 claims and concurrency=5, peak concurrency should be exactly 5."""
    monkeypatch.setenv("DEEP_RESEARCH_GROUNDING_CONCURRENCY", "5")

    claims = [_make_claim(f"Q1-C{i}") for i in range(10)]

    async def fake_gather_sources(claim, workspace):
        return ["some source text that is long enough to pass 200 char check. " * 10]

    monkeypatch.setattr(phase1b, "_gather_claim_sources", fake_gather_sources)

    concurrency_tracker = {"current": 0, "peak": 0}

    async def fake_llm_ground_one_claim(claim, sources, semaphore):
        async with semaphore:
            concurrency_tracker["current"] += 1
            concurrency_tracker["peak"] = max(concurrency_tracker["peak"], concurrency_tracker["current"])
            # Hold the slot so peers pile up at the semaphore
            await asyncio.sleep(0.05)
            concurrency_tracker["current"] -= 1
        return {
            "claim_id": claim.claim_id,
            "score": 0.9,
            "verdict": "GROUNDED",
            "tool": "llm:test",
        }

    monkeypatch.setattr(phase1b, "_llm_ground_one_claim", fake_llm_ground_one_claim)
    # Ensure MARCH recheck doesn't add extra work during this test
    monkeypatch.setattr(phase1b, "_list_march_alternates", lambda: [])

    state = {
        "claims_to_verify": claims,
        "workspace_path": "/tmp/nonexistent-test-ws",
    }
    result = asyncio.run(phase1b.grounding_check_node(state))

    assert len(result["grounding_results"]) == 10
    assert concurrency_tracker["peak"] == 5, (
        f"Expected peak concurrency of 5, got {concurrency_tracker['peak']}"
    )


def test_grounding_respects_env_concurrency_bump(monkeypatch):
    """Setting concurrency=8 should let 8 claims run in parallel."""
    monkeypatch.setenv("DEEP_RESEARCH_GROUNDING_CONCURRENCY", "8")

    claims = [_make_claim(f"Q1-C{i}") for i in range(12)]

    async def fake_gather_sources(claim, workspace):
        return ["source content. " * 50]

    monkeypatch.setattr(phase1b, "_gather_claim_sources", fake_gather_sources)

    tracker = {"current": 0, "peak": 0}

    async def fake_llm(claim, sources, semaphore):
        async with semaphore:
            tracker["current"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["current"])
            await asyncio.sleep(0.05)
            tracker["current"] -= 1
        return {
            "claim_id": claim.claim_id,
            "score": 0.9,
            "verdict": "GROUNDED",
            "tool": "llm:test",
        }

    monkeypatch.setattr(phase1b, "_llm_ground_one_claim", fake_llm)
    monkeypatch.setattr(phase1b, "_list_march_alternates", lambda: [])

    state = {"claims_to_verify": claims, "workspace_path": "/tmp/nonexistent"}
    asyncio.run(phase1b.grounding_check_node(state))

    assert tracker["peak"] == 8, f"Expected peak of 8 with env=8, got {tracker['peak']}"


def test_grounding_no_source_claims_return_no_source(monkeypatch):
    """Claims with no source should short-circuit to NO_SOURCE_TEXT without LLM calls."""
    monkeypatch.setenv("DEEP_RESEARCH_GROUNDING_CONCURRENCY", "3")

    claims = [_make_claim("Q1-C1"), _make_claim("Q1-C2")]

    async def fake_gather_sources(claim, workspace):
        return []  # no sources

    llm_calls = {"count": 0}

    async def fake_llm(claim, sources, semaphore):
        llm_calls["count"] += 1
        return {"claim_id": claim.claim_id, "score": 0.9, "verdict": "GROUNDED", "tool": "llm:test"}

    monkeypatch.setattr(phase1b, "_gather_claim_sources", fake_gather_sources)
    monkeypatch.setattr(phase1b, "_llm_ground_one_claim", fake_llm)
    monkeypatch.setattr(phase1b, "_list_march_alternates", lambda: [])

    state = {"claims_to_verify": claims, "workspace_path": "/tmp/x"}
    result = asyncio.run(phase1b.grounding_check_node(state))

    assert llm_calls["count"] == 0
    assert len(result["grounding_results"]) == 2
    for r in result["grounding_results"]:
        assert r["verdict"] == "NO_SOURCE_TEXT"
