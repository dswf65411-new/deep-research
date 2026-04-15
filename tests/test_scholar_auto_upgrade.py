"""Tests for Whisper plan P1-6 — scholar-intent auto-upgrade of web queries.

Background: the failed-workspace analysis (2026-04-14) showed brief-named
SOTA papers (AIDE, MLE-bench, ResearchAgent) were buried under marketing
blogs when Brave / Serper ranked by backlinks. The fix is a small regex
that detects scholar-intent keywords and wraps the query with a
``site:arxiv.org OR site:aclanthology.org OR site:openreview.net``
clause before handing it to the generic web engine.

This test locks the heuristic's behaviour so:
- scholar queries get the filter (recall boost on the right domains);
- non-scholar queries are left alone (no false positives);
- dedicated scholar/arxiv/github engines are never touched (they already
  query authoritative sources natively);
- user-supplied ``site:`` filters always win.
"""

from __future__ import annotations

import deep_research.nodes.phase1a as p1a


# ---------------------------------------------------------------------------
# Positive cases: scholar keyword on generic web engine → upgrade
# ---------------------------------------------------------------------------


def test_arxiv_keyword_triggers_upgrade_on_brave():
    q = p1a._maybe_upgrade_to_scholar_query("AIDE arxiv 2024", "brave")
    assert "site:arxiv.org" in q
    assert "AIDE arxiv 2024" in q


def test_paper_keyword_triggers_upgrade_on_serper_en():
    q = p1a._maybe_upgrade_to_scholar_query("MLE-bench paper", "serper_en")
    assert "site:aclanthology.org" in q


def test_sota_keyword_triggers_upgrade():
    q = p1a._maybe_upgrade_to_scholar_query("auto-ML SOTA 2025", "brave")
    assert "site:openreview.net" in q


def test_benchmark_keyword_triggers_upgrade():
    q = p1a._maybe_upgrade_to_scholar_query(
        "llm agent benchmark leaderboard", "serper_en"
    )
    assert "site:" in q


def test_conference_acronym_triggers_upgrade():
    # ACL / NeurIPS / ICML etc. signal academic intent even without "paper"
    q = p1a._maybe_upgrade_to_scholar_query("ICLR 2024 auto-ml", "brave")
    assert "site:arxiv.org" in q


# ---------------------------------------------------------------------------
# Negative cases: no scholar signal → leave query unchanged
# ---------------------------------------------------------------------------


def test_no_scholar_signal_leaves_query_unchanged():
    original = "langgraph supervisor pattern tutorial"
    assert p1a._maybe_upgrade_to_scholar_query(original, "brave") == original


def test_marketing_query_unchanged():
    original = "best AI content marketing tools 2026"
    assert p1a._maybe_upgrade_to_scholar_query(original, "serper_en") == original


# ---------------------------------------------------------------------------
# Engine scoping: only brave / serper_en get the upgrade
# ---------------------------------------------------------------------------


def test_serper_scholar_engine_never_upgraded():
    # Scholar API already queries Google Scholar natively; wrapping with
    # site: filter would over-restrict.
    original = "AIDE agent arxiv"
    assert p1a._maybe_upgrade_to_scholar_query(original, "serper_scholar") == original


def test_arxiv_engine_never_upgraded():
    original = "MLE-bench paper SOTA"
    assert p1a._maybe_upgrade_to_scholar_query(original, "arxiv") == original


def test_github_engine_never_upgraded():
    original = "langgraph supervisor arxiv paper"
    assert p1a._maybe_upgrade_to_scholar_query(original, "github") == original


def test_serper_tw_engine_never_upgraded():
    # zh-TW search loses regional coverage if we pin it to english academic
    # domains; leave it alone.
    original = "LangGraph paper 繁體中文介紹"
    assert p1a._maybe_upgrade_to_scholar_query(original, "serper_tw") == original


def test_serper_cn_engine_never_upgraded():
    original = "大模型 benchmark 評估"
    assert p1a._maybe_upgrade_to_scholar_query(original, "serper_cn") == original


# ---------------------------------------------------------------------------
# User-specified site: filter wins
# ---------------------------------------------------------------------------


def test_user_site_filter_is_respected():
    # If the planner / user already added site:, don't stack another.
    original = "AIDE paper site:github.com"
    assert p1a._maybe_upgrade_to_scholar_query(original, "brave") == original
