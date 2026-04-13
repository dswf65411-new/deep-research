"""Tests for Phase 3 coverage sanity check (P2-E).

Covers:
- _find_uncovered_keywords: substring matching logic
- _format_keyword_coverage: output format
- _extract_brief_keywords: LLM call (mocked — no live API needed)
"""

import asyncio
import pytest

from deep_research.nodes.phase3 import (
    _extract_brief_keywords,
    _find_uncovered_keywords,
    _format_keyword_coverage,
)
from deep_research.state import Claim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved(claim_id: str, text: str) -> Claim:
    c = Claim(claim_id=claim_id, claim_text=text, source_ids=["S001"])
    c.status = "approved"
    return c


# ---------------------------------------------------------------------------
# _find_uncovered_keywords — pure function
# ---------------------------------------------------------------------------

class TestFindUncoveredKeywords:
    def test_missing_tools_detected(self):
        claims = [
            _approved("Q1-C1", "Otter.ai 免費版有每月 600 分鐘限制"),
            _approved("Q1-C2", "MacWhisper Pro 支援 Whisper Large v3"),
        ]
        keywords = ["Otter.ai", "MacWhisper", "Audio Hijack", "Plaud Note"]
        uncovered = _find_uncovered_keywords(keywords, claims)
        assert "Audio Hijack" in uncovered
        assert "Plaud Note" in uncovered
        assert "Otter.ai" not in uncovered
        assert "MacWhisper" not in uncovered

    def test_all_covered_returns_empty(self):
        claims = [
            _approved("Q1-C1", "Otter.ai 支援說話者分離"),
            _approved("Q1-C2", "MacWhisper 具備離線模式"),
        ]
        uncovered = _find_uncovered_keywords(["Otter.ai", "MacWhisper"], claims)
        assert uncovered == []

    def test_case_insensitive_matching(self):
        claims = [_approved("Q1-C1", "otter.ai 的功能...")]  # lowercase
        uncovered = _find_uncovered_keywords(["Otter.ai"], claims)
        assert uncovered == []  # should match case-insensitively

    def test_hyphen_normalized_matching(self):
        """'Plaud-Note' in keyword should match 'Plaud Note' in claim."""
        claims = [_approved("Q1-C1", "Plaud Note 是一款硬體錄音裝置")]
        uncovered = _find_uncovered_keywords(["Plaud-Note"], claims)
        assert uncovered == []  # hyphen normalized to space

    def test_empty_keywords_returns_empty(self):
        claims = [_approved("Q1-C1", "some claim")]
        assert _find_uncovered_keywords([], claims) == []

    def test_empty_claims_returns_all_keywords(self):
        uncovered = _find_uncovered_keywords(["Tool A", "Tool B"], [])
        assert uncovered == ["Tool A", "Tool B"]

    def test_partial_match_within_word(self):
        """'iOS' keyword should match claim containing 'iOS 26'."""
        claims = [_approved("Q1-C1", "iOS 26 本機截取支援離線轉錄")]
        uncovered = _find_uncovered_keywords(["iOS"], claims)
        assert uncovered == []

    def test_tool_version_specific(self):
        """'iOS 26' should match claim containing 'iOS 26' exactly."""
        claims = [_approved("Q1-C1", "iOS 26 本機截取支援離線轉錄")]
        uncovered = _find_uncovered_keywords(["iOS 26"], claims)
        assert uncovered == []

    def test_tool_not_in_any_claim(self):
        """'Audio Hijack' not mentioned anywhere → uncovered."""
        claims = [
            _approved("Q1-C1", "MacWhisper 支援多語言"),
            _approved("Q1-C2", "Whisper Large v3 WER 為 8.3%"),
        ]
        uncovered = _find_uncovered_keywords(["Audio Hijack"], claims)
        assert "Audio Hijack" in uncovered


# ---------------------------------------------------------------------------
# _format_keyword_coverage — pure function
# ---------------------------------------------------------------------------

class TestFormatKeywordCoverage:
    def test_no_keywords_returns_note(self):
        result = _format_keyword_coverage([], [])
        assert "未偵測到" in result

    def test_all_covered_no_warning(self):
        keywords = ["Otter.ai", "MacWhisper"]
        result = _format_keyword_coverage(keywords, uncovered=[])
        assert "✅" in result
        assert "⚠️" not in result
        assert "所有" in result

    def test_some_uncovered_shows_warning(self):
        keywords = ["Otter.ai", "Audio Hijack", "Plaud Note"]
        uncovered = ["Audio Hijack", "Plaud Note"]
        result = _format_keyword_coverage(keywords, uncovered)
        assert "⚠️" in result
        assert "Audio Hijack" in result
        assert "Plaud Note" in result
        assert "未找到有效來源" in result

    def test_uncovered_marked_with_x(self):
        result = _format_keyword_coverage(["Tool A"], uncovered=["Tool A"])
        assert "❌" in result

    def test_covered_marked_with_check(self):
        result = _format_keyword_coverage(["Tool A"], uncovered=[])
        assert "✅" in result


# ---------------------------------------------------------------------------
# _extract_brief_keywords — mock LLM, test fallback behavior
# ---------------------------------------------------------------------------

def test_extract_brief_keywords_empty_text():
    """Empty brief → return empty list without LLM call."""
    result = asyncio.run(_extract_brief_keywords(""))
    assert result == []


def test_extract_brief_keywords_short_text():
    """Very short text (< 50 chars) → return empty list without LLM call."""
    result = asyncio.run(_extract_brief_keywords("研究語音工具"))
    assert result == []


def test_extract_brief_keywords_mock_llm(monkeypatch):
    """Mock LLM response → keywords extracted correctly."""
    import json as _json

    class MockResponse:
        content = _json.dumps({"keywords": ["Otter.ai", "Audio Hijack", "Plaud Note", "iOS 26"]})

    async def mock_invoke(**kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "deep_research.nodes.phase3.safe_ainvoke_chain",
        mock_invoke,
    )

    brief = "研究任務：評估 Mac/iPhone 上的錄音轉文字工具，包含 Otter.ai、Audio Hijack、Plaud Note 等。iOS 26 本機截取也是重點。" * 5  # > 50 chars
    result = asyncio.run(_extract_brief_keywords(brief))

    assert "Otter.ai" in result
    assert "Audio Hijack" in result
    assert "Plaud Note" in result
    assert "iOS 26" in result


def test_extract_brief_keywords_llm_failure_fallback(monkeypatch):
    """LLM failure → conservative fallback returns empty list."""
    async def mock_invoke(**kwargs):
        raise RuntimeError("API error")

    monkeypatch.setattr(
        "deep_research.nodes.phase3.safe_ainvoke_chain",
        mock_invoke,
    )

    brief = "研究任務：評估 Mac 錄音工具，包含 Otter.ai 等。" * 5
    result = asyncio.run(_extract_brief_keywords(brief))
    assert result == []  # conservative: no false rejections on error


def test_extract_brief_keywords_malformed_json_fallback(monkeypatch):
    """Malformed LLM JSON → conservative fallback returns empty list."""
    class MockResponse:
        content = "Sorry, I cannot provide this information."

    async def mock_invoke(**kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "deep_research.nodes.phase3.safe_ainvoke_chain",
        mock_invoke,
    )

    brief = "研究任務：評估 Mac 錄音工具，包含 Otter.ai 等。" * 5
    result = asyncio.run(_extract_brief_keywords(brief))
    assert result == []


# ---------------------------------------------------------------------------
# Integration: end-to-end pipeline of keyword coverage
# ---------------------------------------------------------------------------

def test_keyword_coverage_end_to_end():
    """Full pipeline: keywords extracted → some uncovered → warning in output."""
    # Simulate: user mentioned Otter.ai and Audio Hijack in brief
    # Claims only cover Otter.ai, not Audio Hijack
    keywords = ["Otter.ai", "Audio Hijack"]
    claims = [
        _approved("Q1-C1", "Otter.ai 在台灣市場的中文語音識別表現優秀"),
        _approved("Q1-C2", "Otter.ai 提供說話者分離功能，最多支援 10 人"),
    ]

    uncovered = _find_uncovered_keywords(keywords, claims)
    assert uncovered == ["Audio Hijack"]

    note = _format_keyword_coverage(keywords, uncovered)
    assert "Audio Hijack" in note
    assert "❌" in note
    assert "Otter.ai" in note
    assert "✅" in note
    assert "⚠️ **注意" in note
