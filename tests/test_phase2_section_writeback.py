"""Tests for Phase 2 fallback section writeback (P0-1).

Protects against the 2026-04-14 failure mode: 113 approved claims, zero
sections on disk, Phase 3 silently produces blank "detailed analysis" segment.

Covers:
- _build_fallback_section emits [Qx-Cy] inline tags (so Phase 3 ledger
  can still extract claim_ids)
- Biased-source tagging propagates into fallback
- phase2_integrate writes a fallback section to disk when iterative_refine
  raises
- phase2_integrate writes a fallback section when the writer returns empty
- Critical assertion fires when report-sections/ ends up empty despite
  having approved claims (writes gap-log CRITICAL entry)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_research.nodes.phase2 import (
    _build_fallback_section,
    phase2_integrate,
)
from deep_research.state import Claim
from deep_research.tools.workspace import create_workspace, init_gap_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved(claim_id: str, subq: str, text: str, source_ids: list[str] | None = None) -> Claim:
    c = Claim(
        claim_id=claim_id,
        claim_text=text,
        subquestion=subq,
        source_ids=source_ids or ["S001"],
        quote_ids=["QT001"],
    )
    c.status = "approved"
    c.bedrock_score = 0.95
    return c


def _make_workspace(tmp_path) -> str:
    ws = create_workspace("phase2-writeback-test", base_dir=str(tmp_path))
    init_gap_log(ws)
    return ws


# ---------------------------------------------------------------------------
# _build_fallback_section - unit
# ---------------------------------------------------------------------------

class TestFallbackSectionShape:
    def test_inline_claim_ids_present(self):
        claims = [
            _approved("Q1-C1", "Q1", "Whisper Large v3 WER on Chinese is 8.3%"),
            _approved("Q1-C2", "Q1", "MacWhisper Pro sells for 59 USD"),
        ]
        section = _build_fallback_section("Q1", claims, set(), {}, error="LLM timeout")
        # Each claim must carry an inline [Qx-Cy] tag at end of sentence so
        # Phase 3 can split statements from it.
        assert "[Q1-C1]" in section
        assert "[Q1-C2]" in section

    def test_header_marks_failure_loudly(self):
        claims = [_approved("Q2-C1", "Q2", "foo")]
        section = _build_fallback_section("Q2", claims, set(), {}, error="RuntimeError: network down")
        assert "section integration failed" in section or "Phase 2 writer failed" in section
        assert "LOW" in section  # confidence level
        assert "network down" in section  # error text must be transparent

    def test_biased_source_tag_propagates(self):
        claims = [_approved("Q3-C1", "Q3", "Murf AI is the #1 voice synthesis tool", ["S042"])]
        biased = {"murf.ai"}
        sid_to_domain = {"S042": "murf.ai"}
        section = _build_fallback_section("Q3", claims, biased, sid_to_domain, error="x")
        assert "BIASED_SOURCE" in section
        assert "murf.ai" in section

    def test_non_biased_source_no_tag(self):
        claims = [_approved("Q4-C1", "Q4", "x", ["S100"])]
        sid_to_domain = {"S100": "arxiv.org"}
        section = _build_fallback_section("Q4", claims, set(), sid_to_domain, error="x")
        assert "BIASED_SOURCE" not in section

    def test_empty_claims_still_returns_header(self):
        section = _build_fallback_section("Q5", [], set(), {}, error="no claims")
        assert "section integration failed" in section or "Phase 2 writer failed" in section

    def test_error_is_truncated_to_200_chars(self):
        """Prevent multi-MB LLM tracebacks leaking into the report body."""
        huge = "E" * 5000
        claims = [_approved("Q6-C1", "Q6", "x")]
        section = _build_fallback_section("Q6", claims, set(), {}, error=huge)
        # error should appear at most 200 Es, not 300
        assert "E" * 300 not in section


# ---------------------------------------------------------------------------
# phase2_integrate - integration: writer failure path lands on disk
# ---------------------------------------------------------------------------

def _run_phase2(state: dict) -> dict:
    return asyncio.run(phase2_integrate(state))


def _base_state(workspace: str, claims: list[Claim]) -> dict:
    return {
        "workspace_path": workspace,
        "claims": claims,
        "sources": [],
        "blockers": [],
    }


def test_phase2_writer_exception_falls_back_to_disk(tmp_path, monkeypatch):
    """When iterative_refine raises, fallback section must still reach disk."""
    ws = _make_workspace(tmp_path)
    claims = [
        _approved("Q1-C1", "Q1", "claim A"),
        _approved("Q1-C2", "Q1", "claim B"),
    ]

    async def boom_refine(**kwargs):
        raise RuntimeError("simulated LLM blow-up")

    async def boom_invoke(**kwargs):
        raise RuntimeError("simulated LLM blow-up")

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", boom_refine)
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", boom_invoke)

    result = _run_phase2(_base_state(ws, claims))

    # Section landed on disk
    on_disk = list((Path(ws) / "report-sections").glob("*.md"))
    assert len(on_disk) == 1
    body = on_disk[0].read_text(encoding="utf-8")
    assert "[Q1-C1]" in body
    assert "[Q1-C2]" in body
    assert "section integration failed" in body

    # Blocker recorded
    assert any("Phase 2 writer failed" in b or "writer failed" in b for b in result["blockers"])


def test_phase2_empty_response_triggers_fallback(tmp_path, monkeypatch):
    """When LLM returns empty string, fallback should fire - not silent empty file."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q2-C1", "Q2", "claim C")]

    async def empty_refine(**kwargs):
        return "   "  # whitespace only

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", empty_refine)

    # no sources -> routes through safe_ainvoke_chain branch
    class EmptyResp:
        content = ""

    async def empty_invoke(**kwargs):
        return EmptyResp()

    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", empty_invoke)

    result = _run_phase2(_base_state(ws, claims))
    on_disk = list((Path(ws) / "report-sections").glob("*.md"))
    assert len(on_disk) == 1
    body = on_disk[0].read_text(encoding="utf-8")
    assert "[Q2-C1]" in body
    assert "LLM returned empty section" in body


def test_phase2_critical_assertion_when_sections_cleared(tmp_path, monkeypatch):
    """Belt-and-suspenders: if the fallback write itself is sabotaged, gap-log must flag CRITICAL."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q3-C1", "Q3", "claim D")]

    # Make writer fail to enter fallback, then make write_workspace_file also fail
    async def boom(*args, **kwargs):
        raise RuntimeError("LLM blow-up")

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", boom)
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", boom)

    import deep_research.nodes.phase2 as p2
    real_write = p2.write_workspace_file

    def sabotaged_write(workspace_path: str, filename: str, content: str) -> str:
        if filename.startswith("report-sections/"):
            # Intentionally do not write, simulating disk or I/O failure
            return str(Path(workspace_path) / filename)
        return real_write(workspace_path, filename, content)

    monkeypatch.setattr(p2, "write_workspace_file", sabotaged_write)

    result = _run_phase2(_base_state(ws, claims))

    sections_on_disk = list((Path(ws) / "report-sections").glob("*.md"))
    assert sections_on_disk == []

    # CRITICAL must land in blockers
    assert any("[CRITICAL: phase2]" in b for b in result["blockers"])

    # gap-log must record for post-mortem
    gap = (Path(ws) / "gap-log.md").read_text(encoding="utf-8")
    assert "CRITICAL" in gap
    assert "Phase 2 empty output" in gap


def test_phase2_success_writes_normal_section(tmp_path, monkeypatch):
    """Sanity: when LLM works, section writes normally and no CRITICAL fires."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q4-C1", "Q4", "well-behaved claim")]

    async def ok_refine(**kwargs):
        return "Integrated paragraph content. [Q4-C1]\n\nConfidence: HIGH"

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", ok_refine)

    class OkResp:
        content = "Integrated paragraph content. [Q4-C1]\n\nConfidence: HIGH"

    async def ok_invoke(**kwargs):
        return OkResp()

    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", ok_invoke)

    result = _run_phase2(_base_state(ws, claims))

    on_disk = list((Path(ws) / "report-sections").glob("*.md"))
    assert len(on_disk) == 1
    body = on_disk[0].read_text(encoding="utf-8")
    assert "Integrated paragraph content" in body
    assert "section integration failed" not in body
    assert not any("CRITICAL: phase2" in b for b in result.get("blockers", []))
