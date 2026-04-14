"""Tests for Phase 3 sanity check + statement-ledger resilience (P0-1).

Protects against the 2026-04-14 failure mode where:
- Phase 2 wrote 0 sections to disk
- Phase 3 silently produced a blank "Detailed Analysis" segment
- final-report.md still claimed success in its header

The new `_build_critical_banner` in phase3 fires a CRITICAL banner at the
top of final-report whenever any of these conditions are detected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.nodes.phase3 import (
    _DETAILED_ANALYSIS_MIN_CHARS,
    _build_critical_banner,
    _build_statement_ledger,
    _format_statement_ledger,
)
from deep_research.state import Claim
from deep_research.tools.workspace import create_workspace, init_gap_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved(claim_id: str, subq: str, text: str) -> Claim:
    c = Claim(
        claim_id=claim_id,
        claim_text=text,
        subquestion=subq,
        source_ids=["S001"],
        quote_ids=["QT001"],
    )
    c.status = "approved"
    c.bedrock_score = 0.9
    return c


def _make_workspace(tmp_path) -> str:
    ws = create_workspace("phase3-sanity-test", base_dir=str(tmp_path))
    init_gap_log(ws)
    return ws


def _healthy_body() -> str:
    return "Fully integrated paragraph. " * 200  # >> 500 chars


# ---------------------------------------------------------------------------
# _build_critical_banner - healthy reports stay quiet
# ---------------------------------------------------------------------------

def test_banner_silent_for_healthy_report(tmp_path):
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "claim text")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body=_healthy_body(),
        section_files=["a.md"],
        statements=[{"statement_id": "ST-1", "type": "fact", "text": "...", "claim_ids": ["Q1-C1"]}],
        approved_claims=claims,
        brief_keywords=["Otter.ai"],
        uncovered_keywords=[],
    )
    assert banner == ""


def test_banner_silent_when_no_approved_claims(tmp_path):
    """No approved claims -> empty body is EXPECTED (research had nothing to report),
    so CRITICAL banner should NOT fire - different failure class."""
    ws = _make_workspace(tmp_path)
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body="",
        section_files=[],
        statements=[],
        approved_claims=[],
        brief_keywords=[],
        uncovered_keywords=[],
    )
    assert banner == ""


# ---------------------------------------------------------------------------
# Banner fires on concrete failure signatures
# ---------------------------------------------------------------------------

def test_banner_fires_on_short_detailed_analysis(tmp_path):
    """The exact 2026-04-14 symptom: approved claims exist but detail body is empty."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "claim text")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body="short",
        section_files=["a.md"],  # present, but body is still thin
        statements=[{"statement_id": "ST-1", "type": "fact", "text": "...", "claim_ids": ["Q1-C1"]}],
        approved_claims=claims,
        brief_keywords=[],
        uncovered_keywords=[],
    )
    assert banner != ""
    assert "CRITICAL" in banner
    assert "Detailed Analysis" in banner
    # Threshold number must be displayed (user can gauge severity)
    assert str(_DETAILED_ANALYSIS_MIN_CHARS) in banner


def test_banner_fires_on_empty_section_files(tmp_path):
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "x")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body=_healthy_body(),  # even if body somehow has content
        section_files=[],  # this is the smoking gun for a phase2 truncation
        statements=[],
        approved_claims=claims,
        brief_keywords=[],
        uncovered_keywords=[],
    )
    assert "report-sections/ is empty" in banner


def test_banner_fires_on_empty_statement_ledger(tmp_path):
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "x")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body=_healthy_body(),
        section_files=["a.md"],
        statements=[],  # ledger cannot be built
        approved_claims=claims,
        brief_keywords=[],
        uncovered_keywords=[],
    )
    assert "statement-ledger is empty" in banner


def test_banner_fires_on_zero_brief_coverage(tmp_path):
    """Every tool explicitly named in the brief is uncovered -> research provides zero help."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "some off-topic claim")]
    kws = ["AIDE", "MLE-Agent", "ResearchAgent"]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body=_healthy_body(),
        section_files=["a.md"],
        statements=[{"statement_id": "ST-1", "type": "fact", "text": "...", "claim_ids": ["Q1-C1"]}],
        approved_claims=claims,
        brief_keywords=kws,
        uncovered_keywords=kws,  # nothing is covered
    )
    assert "are covered by an approved claim" in banner or "None of the" in banner
    # The first few keywords should be surfaced so the user can see at a glance
    assert "AIDE" in banner


def test_banner_quiet_when_some_brief_keywords_covered(tmp_path):
    """Partial coverage != zero coverage; this rule must not fire."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "AIDE paper content...")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body=_healthy_body(),
        section_files=["a.md"],
        statements=[{"statement_id": "ST-1", "type": "fact", "text": "...", "claim_ids": ["Q1-C1"]}],
        approved_claims=claims,
        brief_keywords=["AIDE", "MLE-Agent"],
        uncovered_keywords=["MLE-Agent"],  # covered 1/2
    )
    # Other conditions healthy -> banner should stay quiet
    assert banner == ""


# ---------------------------------------------------------------------------
# Banner persists to gap-log for post-mortem
# ---------------------------------------------------------------------------

def test_banner_writes_to_gap_log(tmp_path):
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "x")]
    _build_critical_banner(
        workspace=ws,
        fixed_body="",
        section_files=[],
        statements=[],
        approved_claims=claims,
        brief_keywords=[],
        uncovered_keywords=[],
    )
    gap = (Path(ws) / "gap-log.md").read_text(encoding="utf-8")
    assert "CRITICAL" in gap
    assert "Phase 3 sanity check failed" in gap


def test_banner_survives_gap_log_failure(tmp_path, monkeypatch):
    """gap-log write failure must not block the banner from being returned (banner is what the user actually sees)."""
    ws = _make_workspace(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "deep_research.tools.workspace.append_workspace_file",
        boom,
    )

    claims = [_approved("Q1-C1", "Q1", "x")]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body="",
        section_files=[],
        statements=[],
        approved_claims=claims,
        brief_keywords=[],
        uncovered_keywords=[],
    )
    # Full banner still returned
    assert "CRITICAL" in banner


# ---------------------------------------------------------------------------
# Multiple issues surface together (bullet list format)
# ---------------------------------------------------------------------------

def test_banner_lists_all_fired_issues(tmp_path):
    """When multiple conditions fire, the banner must list them all rather than only the first."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "irrelevant content")]
    kws = ["AIDE"]
    banner = _build_critical_banner(
        workspace=ws,
        fixed_body="",  # condition 1
        section_files=[],  # condition 2
        statements=[],  # condition 3
        approved_claims=claims,
        brief_keywords=kws,
        uncovered_keywords=kws,  # condition 4
    )
    # At least four bullets must appear in the banner
    assert banner.count("\n> -") >= 4 or (
        banner.count("- \"Detailed Analysis\"")
        + banner.count("- report-sections")
        + banner.count("- statement-ledger")
        + banner.count("- research brief")
    ) >= 4


# ---------------------------------------------------------------------------
# _build_statement_ledger / _format_statement_ledger - resilience under failure
# ---------------------------------------------------------------------------

def test_empty_sections_input_yields_empty_ledger():
    """No sections -> empty list, must not raise."""
    import asyncio
    out = asyncio.run(_build_statement_ledger([], []))
    assert out == []


def test_format_statement_ledger_handles_empty_list():
    """Even with statements=[], a valid markdown header must be produced (no None, no errors)."""
    md = _format_statement_ledger([])
    assert "statement_id" in md
    assert "| claim_ids" in md  # table header
    # Must not raise KeyError / None


def test_format_statement_ledger_tolerates_missing_fields():
    """LLM may spit out dicts with missing fields; the full ledger must not crash."""
    statements = [
        {"statement_id": "ST-1", "text": "has text"},  # missing claim_ids, type, section
        {"claim_ids": ["Q1-C1"], "type": "fact"},  # missing statement_id, text
        {},  # missing everything
    ]
    md = _format_statement_ledger(statements)
    # All three rows should appear (not just the first)
    assert md.count("|") >= 3 * 7  # 7 pipes per row (8 columns)


def test_format_statement_ledger_escapes_pipe_in_text():
    """A | inside text breaks markdown tables; must be escaped."""
    statements = [
        {
            "statement_id": "ST-1",
            "section": "q1",
            "text": "a sentence containing | pipe",
            "claim_ids": ["Q1-C1"],
            "type": "fact",
        },
    ]
    md = _format_statement_ledger(statements)
    assert "\\|" in md  # | was escaped
