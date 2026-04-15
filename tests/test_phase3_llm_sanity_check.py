"""Tests for Whisper plan P2-1 — LLM-level report sanity check.

The rule-based ``_build_critical_banner`` catches the obvious carcasses
(empty body, no sections). This test covers the *semantic* layer on top:
a small LLM judge re-reads the final body against the brief and flags
defects like off-topic content, contradictions, or boilerplate filler.

We can't test the real LLM in unit tests; instead we monkey-patch
``get_llm`` / ``safe_ainvoke`` and exercise the parsing + banner-merging
logic that lives in the Python code.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import deep_research.nodes.phase3 as p3


# ---------------------------------------------------------------------------
# Helpers to fake the LLM round-trip
# ---------------------------------------------------------------------------


def _fake_llm_response(payload: dict | str) -> SimpleNamespace:
    """Emulate the ChatAnthropic response object — only ``content`` is read."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=content)


def _install_fake_llm(monkeypatch, response_payload):
    async def fake_ainvoke(_llm, _messages):
        return _fake_llm_response(response_payload)

    monkeypatch.setattr(p3, "safe_ainvoke", fake_ainvoke)
    monkeypatch.setattr(p3, "get_llm", lambda *_a, **_k: object())


# ---------------------------------------------------------------------------
# _llm_report_sanity_check — early-exit / short-circuit paths
# ---------------------------------------------------------------------------


def test_empty_body_returns_no_issues():
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="",
        approved_claim_count=5,
    ))
    assert issues == []


def test_empty_brief_returns_no_issues():
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="",
        final_body="x" * 2000,
        approved_claim_count=5,
    ))
    assert issues == []


def test_zero_claims_skips_check():
    """If no claim made it through, the rule-based banner already covers
    it — don't waste an LLM call."""
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief text",
        final_body="x" * 2000,
        approved_claim_count=0,
    ))
    assert issues == []


def test_short_body_skips_check():
    """Bodies shorter than the rule-based floor are already flagged by the
    rule layer; skip the LLM path so banners don't double up."""
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="short body",
        approved_claim_count=3,
    ))
    assert issues == []


# ---------------------------------------------------------------------------
# Happy path: JSON parse + cap + order
# ---------------------------------------------------------------------------


def test_healthy_response_returns_empty(monkeypatch):
    _install_fake_llm(monkeypatch, {"critical_issues": []})
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief about AIDE",
        final_body="AIDE details... " * 100,
        approved_claim_count=10,
    ))
    assert issues == []


def test_critical_issues_are_returned(monkeypatch):
    _install_fake_llm(monkeypatch, {
        "critical_issues": [
            "Body never actually discusses AIDE despite naming it",
            "Q6 (Supervisor failure arbitration) unanswered",
        ]
    })
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=10,
    ))
    assert len(issues) == 2
    assert "AIDE" in issues[0]
    assert "Q6" in issues[1]


def test_issues_capped_at_five(monkeypatch):
    _install_fake_llm(monkeypatch, {
        "critical_issues": [f"issue {i}" for i in range(10)]
    })
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=5,
    ))
    assert len(issues) == 5


def test_malformed_json_is_swallowed(monkeypatch):
    _install_fake_llm(monkeypatch, "this is not JSON at all")
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=5,
    ))
    assert issues == []


def test_json_with_markdown_fence_is_parsed(monkeypatch):
    """LLMs sometimes ignore 'no fences' instruction — strip them."""
    raw = "```json\n" + json.dumps({"critical_issues": ["drift"]}) + "\n```"
    _install_fake_llm(monkeypatch, raw)
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=5,
    ))
    assert issues == ["drift"]


def test_llm_exception_is_swallowed(monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("API 500")

    monkeypatch.setattr(p3, "safe_ainvoke", boom)
    monkeypatch.setattr(p3, "get_llm", lambda *_a, **_k: object())

    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=5,
    ))
    assert issues == []


def test_non_string_issue_items_filtered_out(monkeypatch):
    _install_fake_llm(monkeypatch, {
        "critical_issues": ["valid one", 42, None, "another valid"]
    })
    issues = asyncio.run(p3._llm_report_sanity_check(
        brief_text="brief",
        final_body="body " * 300,
        approved_claim_count=5,
    ))
    assert issues == ["valid one", "another valid"]


# ---------------------------------------------------------------------------
# Banner merging
# ---------------------------------------------------------------------------


def test_banner_merge_preserves_rule_findings(tmp_path):
    existing = (
        "\n> **CRITICAL — report integrity check failed**\n"
        "> - detailed analysis empty\n\n"
    )
    merged = p3._append_llm_issues_to_banner(
        existing, ["drift detected"], str(tmp_path)
    )
    # Both the rule finding and the LLM finding should be visible
    assert "detailed analysis empty" in merged
    assert "drift detected" in merged


def test_banner_merge_creates_header_when_rule_clean(tmp_path):
    merged = p3._append_llm_issues_to_banner("", ["off-topic body"], str(tmp_path))
    assert "CRITICAL" in merged
    assert "off-topic body" in merged


def test_banner_merge_writes_sanity_file(tmp_path):
    p3._append_llm_issues_to_banner(
        "", ["issue A", "issue B"], str(tmp_path)
    )
    out = tmp_path / "report-sanity-check.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "issue A" in text and "issue B" in text


def test_banner_merge_empty_issues_returns_unchanged():
    assert p3._append_llm_issues_to_banner("", [], "/tmp/nowhere") == ""
    assert p3._append_llm_issues_to_banner("some banner", [], "/tmp") == "some banner"
