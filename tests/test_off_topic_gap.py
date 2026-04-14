"""Verify that OFF_TOPIC sources are summarized into gap-log.md per SQ.

Without this signal, the next round's Planner keeps generating queries from the
same angle, and the goal-aware extractor keeps rejecting the resulting pages —
an invisible feedback loop that burns budget. The gap-log entry makes the
problem visible so the Planner can rewrite the SQ's angle.
"""

from pathlib import Path

from deep_research.nodes.phase1a import _log_off_topic_ratio


def test_high_ratio_written_to_gap_log(tmp_path: Path):
    raw = [
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "LIVE", "content": "x"},
        {"subquestion": "Q2", "status": "LIVE", "content": "x"},
        {"subquestion": "Q2", "status": "LIVE", "content": "x"},
    ]
    _log_off_topic_ratio(str(tmp_path), iteration=1, raw_sources=raw)

    gap = (tmp_path / "gap-log.md").read_text()
    assert "OFF_TOPIC_RATIO" in gap
    assert "Q1: [OFF_TOPIC_RATIO >= 0.5] 3/4" in gap
    assert "rewrite angle" in gap
    # Q2 has 0/2 off-topic: must not be flagged
    assert "Q2:" not in gap


def test_below_threshold_skipped(tmp_path: Path):
    """Only log when ratio >= threshold (default 0.5); below it is normal noise."""
    raw = [
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "LIVE", "content": "x"},
        {"subquestion": "Q1", "status": "LIVE", "content": "x"},
        {"subquestion": "Q1", "status": "LIVE", "content": "x"},
    ]
    _log_off_topic_ratio(str(tmp_path), iteration=1, raw_sources=raw)
    # No file produced (nothing to log)
    assert not (tmp_path / "gap-log.md").exists()


def test_ignores_sq_with_single_source(tmp_path: Path):
    """One source is a sampling floor — can't conclude angle failure from N=1.
    Only flag SQs with >= 2 sources."""
    raw = [
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
    ]
    _log_off_topic_ratio(str(tmp_path), iteration=1, raw_sources=raw)
    assert not (tmp_path / "gap-log.md").exists()


def test_ignores_empty_content_sources(tmp_path: Path):
    """Sources with no content never went through the extractor, so OFF_TOPIC
    cannot apply to them. Exclude from the denominator."""
    raw = [
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "OFF_TOPIC", "content": "x"},
        {"subquestion": "Q1", "status": "UNREACHABLE", "content": ""},
    ]
    _log_off_topic_ratio(str(tmp_path), iteration=1, raw_sources=raw)
    gap = (tmp_path / "gap-log.md").read_text()
    # 2/2 rather than 2/3 — the UNREACHABLE one has no content
    assert "Q1: [OFF_TOPIC_RATIO >= 0.5] 2/2" in gap
