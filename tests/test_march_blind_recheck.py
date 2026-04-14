"""Tests for MARCH blind recheck in phase1b.

MARCH (arxiv 2603.24579): a second-vendor LLM independently verifies each
GROUNDED claim. Disagreement downgrades the claim. These tests mock the
alternate-provider LLM call to exercise the agreement / disagreement /
exception branches without real API usage.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from deep_research.nodes.phase1b import _march_blind_recheck
from deep_research.state import Claim


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _mock_invoke_return(content: str):
    async def _inner(llm, messages, **kwargs):
        return _FakeResponse(content)
    return _inner


def _mock_invoke_raise(exc: Exception):
    async def _inner(llm, messages, **kwargs):
        raise exc
    return _inner


def test_march_recheck_agrees():
    claim = Claim(claim_id="Q1-C1", claim_text="Sky is blue", claim_type="qualitative", source_ids=["S1"])
    sem = asyncio.Semaphore(1)
    with patch("deep_research.nodes.phase1b.get_llm", return_value="dummy-llm"), \
         patch("deep_research.nodes.phase1b.safe_ainvoke", new=_mock_invoke_return('{"score": 0.9}')):
        result = asyncio.run(_march_blind_recheck(claim, ["Sources say the sky is blue."], "gemini", sem))
    assert result is not None
    assert result["verdict"] == "GROUNDED"
    assert result["tool"] == "march-gemini"
    assert result["score"] == pytest.approx(0.9)


def test_march_recheck_disagrees():
    claim = Claim(claim_id="Q1-C2", claim_text="Dogs fly naturally", claim_type="qualitative", source_ids=["S1"])
    sem = asyncio.Semaphore(1)
    with patch("deep_research.nodes.phase1b.get_llm", return_value="dummy-llm"), \
         patch("deep_research.nodes.phase1b.safe_ainvoke", new=_mock_invoke_return('{"score": 0.1}')):
        result = asyncio.run(_march_blind_recheck(claim, ["Dogs are mammals that walk on four legs."], "gemini", sem))
    assert result is not None
    assert result["verdict"] == "NOT_GROUNDED"
    assert result["score"] == pytest.approx(0.1)


def test_march_recheck_alt_provider_error_returns_none():
    """When alternate provider itself fails, return None so the primary verdict is preserved."""
    claim = Claim(claim_id="Q1-C3", claim_text="X", claim_type="qualitative", source_ids=["S1"])
    sem = asyncio.Semaphore(1)
    with patch("deep_research.nodes.phase1b.get_llm", return_value="dummy-llm"), \
         patch("deep_research.nodes.phase1b.safe_ainvoke", new=_mock_invoke_raise(RuntimeError("API down"))):
        result = asyncio.run(_march_blind_recheck(claim, ["source text"], "openai", sem))
    assert result is None


def test_march_recheck_malformed_json_returns_none():
    """Defensively handle LLM output that isn't valid JSON."""
    claim = Claim(claim_id="Q1-C4", claim_text="Y", claim_type="qualitative", source_ids=["S1"])
    sem = asyncio.Semaphore(1)
    with patch("deep_research.nodes.phase1b.get_llm", return_value="dummy-llm"), \
         patch("deep_research.nodes.phase1b.safe_ainvoke", new=_mock_invoke_return("no json here at all")):
        result = asyncio.run(_march_blind_recheck(claim, ["source"], "openai", sem))
    assert result is None


def test_march_recheck_uses_claim_type_threshold():
    """Numeric claims have a stricter threshold (0.8) than qualitative (0.7).

    Score 0.75 should be NOT_GROUNDED for numeric but GROUNDED for qualitative.
    """
    sem = asyncio.Semaphore(1)
    with patch("deep_research.nodes.phase1b.get_llm", return_value="dummy-llm"), \
         patch("deep_research.nodes.phase1b.safe_ainvoke", new=_mock_invoke_return('{"score": 0.75}')):
        numeric = Claim(claim_id="Q1-C5", claim_text="n", claim_type="numeric", source_ids=["S1"])
        qual = Claim(claim_id="Q1-C6", claim_text="q", claim_type="qualitative", source_ids=["S1"])
        r1 = asyncio.run(_march_blind_recheck(numeric, ["src"], "gemini", sem))
        r2 = asyncio.run(_march_blind_recheck(qual, ["src"], "gemini", sem))
    assert r1["verdict"] == "NOT_GROUNDED"  # 0.75 < 0.8
    assert r2["verdict"] == "GROUNDED"      # 0.75 >= 0.7
