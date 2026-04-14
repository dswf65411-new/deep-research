"""Verify per-SQ evidence snapshot (sq-progress.md) — Tongyi-style evolving state.

Tech_Report §3.1 (Context Management): instead of feeding the full interaction
history to the next iteration, feed a compact `S_t` that summarises what was
found. Our pipeline writes that snapshot to `sq-progress.md` after each phase1a
iteration, and the next iteration's planner reads it to avoid re-asking queries
whose answers are already in hand.
"""

import asyncio
from pathlib import Path

from deep_research.nodes.phase1a import (
    _PLANNER_SYSTEM,
    _build_sq_evidence_snapshot,
)


def test_snapshot_lists_live_evidence_and_zero_live_sqs(tmp_path: Path):
    sq_texts = {
        "Q1": "background on LangGraph supervisor pattern",
        "Q3": "failure handling in supervisor architectures",
    }
    raw_sources = [
        {"subquestion": "Q1", "status": "LIVE",
         "title": "LangGraph Supervisor Pattern Overview",
         "url": "https://langchain-ai.github.io/langgraph/"},
        {"subquestion": "Q1", "status": "LIVE",
         "title": "Multi-agent Supervisor Guide",
         "url": "https://example.com/supervisor"},
        {"subquestion": "Q3", "status": "OFF_TOPIC",
         "title": "Brand Monitoring Trends 2026",
         "url": "https://sight.ai/trends"},
        {"subquestion": "Q3", "status": "UNREACHABLE",
         "title": "(dead link)",
         "url": "https://example.com/404"},
    ]

    _build_sq_evidence_snapshot(str(tmp_path), iteration=0, raw_sources=raw_sources, sq_texts=sq_texts)

    text = (tmp_path / "sq-progress.md").read_text()
    # Iteration header (iteration starts at 0 but display as round 1)
    assert "Iteration 1 — per-SQ evidence snapshot" in text
    # Q1 has LIVE evidence — titles should be present
    assert "LangGraph Supervisor Pattern Overview" in text
    assert "https://langchain-ai.github.io/langgraph/" in text
    # Q1 header includes the SQ text
    assert "background on LangGraph supervisor pattern" in text
    # Q3 has zero LIVE — must be flagged
    assert "no LIVE evidence yet" in text
    # Status counts appear
    assert "LIVE 2" in text
    assert "OFF_TOPIC 1" in text
    assert "UNREACHABLE 1" in text


def test_snapshot_sorts_sq_numerically(tmp_path: Path):
    """Q10 should sort AFTER Q2, not between Q1 and Q2."""
    raw_sources = [
        {"subquestion": f"Q{i}", "status": "LIVE", "title": f"t{i}", "url": f"https://e.com/{i}"}
        for i in [10, 2, 1]
    ]
    _build_sq_evidence_snapshot(str(tmp_path), iteration=0, raw_sources=raw_sources, sq_texts={})
    text = (tmp_path / "sq-progress.md").read_text()
    q1_pos = text.index("**Q1**")
    q2_pos = text.index("**Q2**")
    q10_pos = text.index("**Q10**")
    assert q1_pos < q2_pos < q10_pos


def test_snapshot_skipped_for_empty_sources(tmp_path: Path):
    _build_sq_evidence_snapshot(str(tmp_path), iteration=0, raw_sources=[], sq_texts={})
    assert not (tmp_path / "sq-progress.md").exists()


def test_snapshot_truncates_live_title_length(tmp_path: Path):
    """Over-long titles should be truncated to 100 chars so the snapshot stays compact."""
    raw_sources = [{
        "subquestion": "Q1",
        "status": "LIVE",
        "title": "x" * 300,
        "url": "https://e.com",
    }]
    _build_sq_evidence_snapshot(str(tmp_path), iteration=0, raw_sources=raw_sources, sq_texts={})
    text = (tmp_path / "sq-progress.md").read_text()
    # 100 x's present; 101+ x's not present on any single line
    for line in text.splitlines():
        assert "x" * 200 not in line


def test_snapshot_caps_live_entries_per_sq(tmp_path: Path):
    """Show at most 5 LIVE entries per SQ so the planner's context stays bounded."""
    raw_sources = [
        {"subquestion": "Q1", "status": "LIVE", "title": f"title-{i}", "url": f"https://e.com/{i}"}
        for i in range(10)
    ]
    _build_sq_evidence_snapshot(str(tmp_path), iteration=0, raw_sources=raw_sources, sq_texts={})
    text = (tmp_path / "sq-progress.md").read_text()
    for i in range(5):
        assert f"title-{i}" in text
    # Entries 5..9 omitted (beyond the 5-entry cap)
    assert "title-5" not in text
    # But the status line still reflects the true count (LIVE 10)
    assert "LIVE 10" in text


def test_planner_prompt_has_sq_progress_rule():
    """The planner's system prompt must reference sq-progress so the LLM knows
    to use it. Grep-anchor so a future rewrite cannot silently drop the rule."""
    assert "Per-SQ S_t rule" in _PLANNER_SYSTEM
    assert "no LIVE evidence yet" in _PLANNER_SYSTEM
    assert "Tongyi" in _PLANNER_SYSTEM  # surface the source of the design


def test_planner_user_msg_includes_sq_progress_section():
    """When sq_progress is passed, the planner's user_msg must include the section
    so the LLM actually sees S_t."""
    from deep_research.nodes.phase1a import _plan_queries
    from unittest.mock import AsyncMock, patch

    captured_user_msg = {}

    class FakeResponse:
        content = '{"queries": []}'

    async def fake_chain(*args, **kwargs):
        msgs = kwargs.get("messages", args[0] if args else [])
        captured_user_msg["msg"] = msgs[-1].content if msgs else ""
        return FakeResponse()

    with patch("deep_research.nodes.phase1a.safe_ainvoke_chain", new=AsyncMock(side_effect=fake_chain)):
        asyncio.run(_plan_queries(
            plan="# plan",
            coverage="## Q1: ...",
            gap_log="",
            sq_progress="## Iteration 1 — per-SQ evidence snapshot\n- **Q1**: found X",
            iteration=1,
            remaining_budget=10,
            already_searched=[],
            depth="deep",
        ))

    msg = captured_user_msg.get("msg", "")
    assert "Per-SQ evidence snapshot from last iteration (S_t)" in msg
    assert "found X" in msg


def test_planner_user_msg_omits_sq_progress_when_empty():
    """First iteration has no sq_progress — the section should be fully omitted
    rather than appearing empty."""
    from deep_research.nodes.phase1a import _plan_queries
    from unittest.mock import AsyncMock, patch

    captured_user_msg = {}

    class FakeResponse:
        content = '{"queries": []}'

    async def fake_chain(*args, **kwargs):
        msgs = kwargs.get("messages", args[0] if args else [])
        captured_user_msg["msg"] = msgs[-1].content if msgs else ""
        return FakeResponse()

    with patch("deep_research.nodes.phase1a.safe_ainvoke_chain", new=AsyncMock(side_effect=fake_chain)):
        asyncio.run(_plan_queries(
            plan="# plan",
            coverage="## Q1: ...",
            gap_log="",
            sq_progress="",
            iteration=0,
            remaining_budget=10,
            already_searched=[],
            depth="deep",
        ))

    msg = captured_user_msg.get("msg", "")
    assert "Per-SQ evidence snapshot from last iteration" not in msg
