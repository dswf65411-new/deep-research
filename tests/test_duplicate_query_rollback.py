"""Verify cross-round duplicate query rollback (MiroThinker-inspired).

The planner's prompt asks the LLM to avoid repeating already-searched queries,
but LLMs still paraphrase. We drop near-duplicate queries pre-execution so:
  1. Search budget is not wasted on restatements.
  2. Gap-log accumulates `[DUPLICATE ROLLBACK]` entries — forcing the next round's
     planner to switch angle instead of rewording synonyms.
  3. After 5 consecutive stuck rounds on an SQ, a `[STUCK]` escalation lands so
     downstream phases can treat the SQ as a BLOCKER candidate.
"""

from pathlib import Path

from deep_research.nodes.phase1a import (
    _count_consecutive_stuck_rounds,
    _detect_duplicate_queries,
    _log_duplicate_rollback,
)


def test_detect_exact_normalized_match():
    planned = [
        {"subquestion": "Q1", "query": "LangGraph supervisor pattern", "role": "advocate"},
        {"subquestion": "Q1", "query": "AutoML-Agent arxiv", "role": "advocate"},
    ]
    # "langgraph supervisor pattern" matches after NFKC + lowercase + whitespace strip
    already = ["LangGraph Supervisor Pattern!"]
    kept, dropped = _detect_duplicate_queries(planned, already)
    assert len(kept) == 1
    assert kept[0]["query"] == "AutoML-Agent arxiv"
    assert len(dropped) == 1
    assert dropped[0]["subquestion"] == "Q1"


def test_detect_fuzzy_paraphrase():
    """Minor reword (>= 0.9 ratio) must also be caught as duplicate."""
    planned = [
        {"subquestion": "Q2", "query": "LangGraph supervisor pattern overview 2026", "role": "advocate"},
    ]
    already = ["LangGraph supervisor pattern overview year 2026"]  # one-word insert
    kept, dropped = _detect_duplicate_queries(planned, already)
    assert len(dropped) == 1
    assert len(kept) == 0


def test_detect_empty_already_searched():
    """First round has nothing searched — all planned queries pass."""
    planned = [
        {"subquestion": "Q1", "query": "test query one", "role": "advocate"},
        {"subquestion": "Q2", "query": "test query two", "role": "critic"},
    ]
    kept, dropped = _detect_duplicate_queries(planned, [])
    assert len(kept) == 2
    assert dropped == []


def test_detect_empty_query_text_skipped():
    """Blank query should be silently dropped (not counted as duplicate)."""
    planned = [
        {"subquestion": "Q1", "query": "", "role": "advocate"},
        {"subquestion": "Q1", "query": "good query", "role": "advocate"},
    ]
    kept, dropped = _detect_duplicate_queries(planned, ["good query"])
    assert len(kept) == 0
    # Only "good query" counted as duplicate; blank silently dropped from kept/dropped
    assert len(dropped) == 1


def test_detect_within_batch_duplicate():
    """A planned query that duplicates an earlier kept query in the same batch
    should also be dropped — otherwise the within-round seen-set in the planner
    would be the only defense, and fuzzy matches slip through."""
    planned = [
        {"subquestion": "Q1", "query": "multi-agent evaluation framework 2026", "role": "advocate"},
        {"subquestion": "Q1", "query": "multi-agent evaluation framework year 2026", "role": "advocate"},
    ]
    kept, dropped = _detect_duplicate_queries(planned, [])
    # First passes; second is a near-duplicate of the first now in prior_norm set
    assert len(kept) == 1
    assert len(dropped) == 1


def test_log_duplicate_rollback_writes_gap_log(tmp_path: Path):
    dropped = [
        {"subquestion": "Q1", "query": "LangGraph supervisor pattern overview", "role": "advocate"},
        {"subquestion": "Q1", "query": "LangGraph supervisor pattern details", "role": "advocate"},
        {"subquestion": "Q3", "query": "multi-agent critic failure mode", "role": "critic"},
    ]
    _log_duplicate_rollback(str(tmp_path), iteration=2, dropped=dropped)

    text = (tmp_path / "gap-log.md").read_text()
    assert "[DUPLICATE ROLLBACK] round 3" in text
    # Q1 had 2 drops, Q3 had 1 — both listed
    assert "**Q1**: 2 duplicate queries dropped" in text
    assert "**Q3**: 1 duplicate queries dropped" in text
    # Next-round instruction present
    assert "MUST switch angle" in text


def test_log_duplicate_rollback_skipped_for_empty_list(tmp_path: Path):
    _log_duplicate_rollback(str(tmp_path), iteration=0, dropped=[])
    assert not (tmp_path / "gap-log.md").exists()


def test_count_consecutive_stuck_rounds_on_fresh_log():
    assert _count_consecutive_stuck_rounds("", "Q1") == 0
    assert _count_consecutive_stuck_rounds("random text without rollback", "Q1") == 0


def test_count_consecutive_stuck_rounds_tracks_latest_streak():
    gap_log = """
## [DUPLICATE ROLLBACK] round 2
- **Q1**: 1 duplicate queries dropped

## [DUPLICATE ROLLBACK] round 3
- **Q1**: 2 duplicate queries dropped

## [DUPLICATE ROLLBACK] round 4
- **Q1**: 1 duplicate queries dropped
- **Q2**: 3 duplicate queries dropped

## [DUPLICATE ROLLBACK] round 5
- **Q1**: 1 duplicate queries dropped
"""
    # Q1 rolled back in rounds 2,3,4,5 — all consecutive → 4
    assert _count_consecutive_stuck_rounds(gap_log, "Q1") == 4
    # Q2 only rolled back in round 4 → 1 (latest round is 5, Q2 missing there = 0)
    assert _count_consecutive_stuck_rounds(gap_log, "Q2") == 0


def test_count_consecutive_stuck_rounds_breaks_on_gap():
    """A round without the target SQ in the rollback list breaks the streak."""
    gap_log = """
## [DUPLICATE ROLLBACK] round 1
- **Q1**: 1 duplicate queries dropped

## [DUPLICATE ROLLBACK] round 2
- **Q2**: 1 duplicate queries dropped

## [DUPLICATE ROLLBACK] round 3
- **Q1**: 1 duplicate queries dropped
"""
    # Latest round (3) has Q1 → start at 1; round 2 missing Q1 → streak stops.
    assert _count_consecutive_stuck_rounds(gap_log, "Q1") == 1
