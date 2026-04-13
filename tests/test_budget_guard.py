"""Smoke tests for Budget Guard — issue #4 fix.

Verifies:
- DEPTH_CONFIG has min_budget_per_sq for all depths
- _extract_sq_ids parses Q-IDs correctly from plan text
- _count_queries_per_sq counts from execution-log format
- _log_budget_gaps writes gap-log correctly when SQs are underfunded
- No gap-log write when all SQs are funded
"""

import os
import tempfile

import pytest

from deep_research.nodes.phase0 import DEPTH_CONFIG
from deep_research.nodes.phase1a import (
    _count_queries_per_sq,
    _extract_sq_ids,
    _log_budget_gaps,
)
from deep_research.tools.workspace import read_workspace_file


# ---------------------------------------------------------------------------
# DEPTH_CONFIG
# ---------------------------------------------------------------------------

def test_depth_config_has_min_budget_per_sq():
    for depth in ("quick", "standard", "deep"):
        assert "min_budget_per_sq" in DEPTH_CONFIG[depth], (
            f"DEPTH_CONFIG['{depth}'] missing min_budget_per_sq"
        )
    assert DEPTH_CONFIG["quick"]["min_budget_per_sq"] == 3
    assert DEPTH_CONFIG["standard"]["min_budget_per_sq"] == 6
    assert DEPTH_CONFIG["deep"]["min_budget_per_sq"] == 12


# ---------------------------------------------------------------------------
# _extract_sq_ids
# ---------------------------------------------------------------------------

def test_extract_sq_ids_basic():
    plan = "## Q1 是否好用？\n\n## Q2 競爭者？\n\n## Q3 ...\n\n## Q1 重複？"
    ids = _extract_sq_ids(plan)
    assert ids == ["Q1", "Q2", "Q3"]  # 有序、去重


def test_extract_sq_ids_empty():
    assert _extract_sq_ids("") == []
    assert _extract_sq_ids("No sub questions here") == []


def test_extract_sq_ids_preserves_order():
    plan = "Q3 first, Q1 second, Q2 third"
    ids = _extract_sq_ids(plan)
    assert ids == ["Q3", "Q1", "Q2"]


# ---------------------------------------------------------------------------
# _count_queries_per_sq
# ---------------------------------------------------------------------------

EXEC_LOG_SAMPLE = """\
- iOS 18 call recording VoIP support update 2025 [Q1/advocate/en]
- iOS 18.4 通話錄音 LINE VoIP 支援 更新 [Q1/advocate/zh-TW]
- iPhone LINE call recording impossible iOS sandbox limitation 2025 [Q1/critic/en]
- best transcription app Chinese speaker diarization 2025 [Q2/advocate/en]
- 中文 語音轉文字 說話者辨識 App 推薦 2025 [Q2/advocate/zh-TW]
"""


def test_count_queries_per_sq_basic():
    counts = _count_queries_per_sq(EXEC_LOG_SAMPLE)
    assert counts == {"Q1": 3, "Q2": 2}


def test_count_queries_per_sq_empty():
    assert _count_queries_per_sq("") == {}
    assert _count_queries_per_sq("no queries here") == {}


def test_count_queries_per_sq_missing_sq_returns_zero():
    counts = _count_queries_per_sq(EXEC_LOG_SAMPLE)
    assert counts.get("Q5", 0) == 0


# ---------------------------------------------------------------------------
# _log_budget_gaps
# ---------------------------------------------------------------------------

def test_log_budget_gaps_writes_when_underfunded():
    sq_ids = ["Q1", "Q2", "Q3"]
    sq_counts = {"Q1": 3, "Q2": 1, "Q3": 0}  # Q2, Q3 below min=3
    with tempfile.TemporaryDirectory() as ws:
        # gap-log.md must exist first (append_workspace_file expects it or creates it)
        open(os.path.join(ws, "gap-log.md"), "w").close()
        _log_budget_gaps(ws, iteration=2, sq_ids=sq_ids, sq_counts=sq_counts, min_per_sq=3)
        content = open(os.path.join(ws, "gap-log.md"), encoding="utf-8").read()
        assert "## 預算缺口（第 2 輪後）" in content
        assert "Q2" in content
        assert "Q3" in content
        assert "Q1" not in content  # Q1 is funded


def test_log_budget_gaps_no_write_when_all_funded():
    sq_ids = ["Q1", "Q2"]
    sq_counts = {"Q1": 5, "Q2": 8}
    with tempfile.TemporaryDirectory() as ws:
        gap_path = os.path.join(ws, "gap-log.md")
        open(gap_path, "w").write("# Gap Log\n")
        _log_budget_gaps(ws, iteration=3, sq_ids=sq_ids, sq_counts=sq_counts, min_per_sq=3)
        content = open(gap_path, encoding="utf-8").read()
        assert "預算缺口" not in content  # nothing appended


def test_log_budget_gaps_shows_shortage_count():
    sq_ids = ["Q5"]
    sq_counts = {"Q5": 2}
    with tempfile.TemporaryDirectory() as ws:
        open(os.path.join(ws, "gap-log.md"), "w").close()
        _log_budget_gaps(ws, iteration=1, sq_ids=sq_ids, sq_counts=sq_counts, min_per_sq=12)
        content = open(os.path.join(ws, "gap-log.md"), encoding="utf-8").read()
        assert "仍缺 10 次" in content  # 12 - 2 = 10


# ---------------------------------------------------------------------------
# Integration: underfunded logic
# ---------------------------------------------------------------------------

def test_underfunded_detection():
    """從 exec_log 算出 underfunded 清單的完整流程。"""
    plan = "Q1 子問題一\nQ2 子問題二\nQ3 子問題三\nQ4 子問題四"
    exec_log = EXEC_LOG_SAMPLE  # Q1=3, Q2=2, Q3/Q4 missing

    sq_ids = _extract_sq_ids(plan)
    sq_counts = _count_queries_per_sq(exec_log)
    min_per_sq = 6

    underfunded = [sq for sq in sq_ids if sq_counts.get(sq, 0) < min_per_sq]
    assert "Q1" in underfunded   # 3 < 6
    assert "Q2" in underfunded   # 2 < 6
    assert "Q3" in underfunded   # 0 < 6
    assert "Q4" in underfunded   # 0 < 6
