"""Verify the source-pool curator (gpt-researcher `skills/curator.py` pattern).

LLM scores each fetched source on relevance/credibility/quant_value (0-5 each);
scores go to `source-curation.md`. Avg < 2.0 gets a `[CURATOR WARNING]` in
gap-log so phase1b/phase2 can tighten grounding thresholds for junk sources.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from deep_research.nodes.phase1a import (
    _CURATOR_SYSTEM,
    _curate_sources,
    _write_source_curation,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def test_curator_system_prompt_anchors():
    """Grep-anchor so a future rewrite does not silently drop key rules."""
    assert "relevance" in _CURATOR_SYSTEM
    assert "credibility" in _CURATOR_SYSTEM
    assert "quant_value" in _CURATOR_SYSTEM
    assert "STRICT JSON" in _CURATOR_SYSTEM


def test_curate_sources_returns_clipped_scores():
    raw_sources = [
        {"source_id": "S001", "url": "https://arxiv.org/abs/1", "title": "Paper 1",
         "content": "data paper body", "status": "LIVE", "subquestion": "Q1", "role": "advocate"},
        {"source_id": "S002", "url": "https://blog.example.com/x", "title": "Marketing blog",
         "content": "light body", "status": "LIVE", "subquestion": "Q2", "role": "critic"},
    ]
    extractions = [
        {"source_id": "S001", "claims": [{"claim_text": "x"}, {"claim_text": "y"}]},
        {"source_id": "S002", "claims": []},
    ]

    payload = {
        "scores": [
            {"source_id": "S001", "relevance": 5, "credibility": 4, "quant_value": 3, "note": "seminal paper"},
            {"source_id": "S002", "relevance": 1, "credibility": 1, "quant_value": 0, "note": "marketing blog"},
            # Clipping check: above 5 must clamp
            {"source_id": "S001", "relevance": 99, "credibility": 5, "quant_value": 5, "note": "dup"},
        ]
    }

    async def fake_chain(*args, **kwargs):
        return _FakeResponse(json.dumps(payload))

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        scores = asyncio.run(_curate_sources(raw_sources, extractions, plan="plan", coverage="coverage"))

    # Duplicate S001 ignored (first wins)
    assert len(scores) == 2
    assert scores[0]["source_id"] == "S001"
    assert scores[0]["relevance"] == 5
    # Clipping still applies; here S001's first score used
    assert scores[1]["source_id"] == "S002"
    assert scores[1]["relevance"] == 1


def test_curate_sources_empty_for_unreachable_only():
    """UNREACHABLE sources (no content, not thin) must not be sent to LLM."""
    raw_sources = [{"source_id": "S999", "content": "", "status": "UNREACHABLE", "url": "https://x", "title": "t"}]

    async def fake_chain(*args, **kwargs):
        raise AssertionError("LLM should not be called when nothing is live")

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        scores = asyncio.run(_curate_sources(raw_sources, [], plan="p", coverage="c"))
    assert scores == []


def test_curate_sources_swallows_invalid_json():
    raw_sources = [{"source_id": "S1", "content": "body", "url": "https://x", "title": "t",
                    "status": "LIVE", "subquestion": "Q1", "role": "advocate"}]

    async def fake_chain(*args, **kwargs):
        return _FakeResponse("not json at all")

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        scores = asyncio.run(_curate_sources(raw_sources, [], plan="p", coverage="c"))
    assert scores == []


def test_write_source_curation_writes_header_and_rows(tmp_path: Path):
    scores = [
        {"source_id": "S001", "relevance": 5, "credibility": 4, "quant_value": 3, "note": "paper"},
        {"source_id": "S002", "relevance": 1, "credibility": 1, "quant_value": 0, "note": "marketing"},
    ]
    _write_source_curation(str(tmp_path), iteration=0, scores=scores)

    text = (tmp_path / "source-curation.md").read_text()
    assert "| source_id | round | relevance | credibility | quant_value | avg | note |" in text
    assert "| S001 | 1 | 5 | 4 | 3 | 4.00 | paper |" in text
    assert "| S002 | 1 | 1 | 1 | 0 | 0.67 | marketing |" in text

    gap = (tmp_path / "gap-log.md").read_text()
    assert "[CURATOR WARNING] low-quality sources (round 1)" in gap
    # S002 (avg 0.67) < 2.0 → flagged
    assert "S002: avg 0.67" in gap
    # S001 (avg 4.00) ≥ 2.0 → not flagged
    assert "S001: avg" not in gap


def test_write_source_curation_header_not_duplicated(tmp_path: Path):
    """Round 2's append must not re-emit the header."""
    scores1 = [{"source_id": "S1", "relevance": 3, "credibility": 3, "quant_value": 3, "note": "ok"}]
    scores2 = [{"source_id": "S2", "relevance": 2, "credibility": 2, "quant_value": 2, "note": "ok2"}]
    _write_source_curation(str(tmp_path), iteration=0, scores=scores1)
    _write_source_curation(str(tmp_path), iteration=1, scores=scores2)

    text = (tmp_path / "source-curation.md").read_text()
    assert text.count("| source_id | round | relevance") == 1
    assert "| S2 | 2 | 2 | 2 | 2 |" in text


def test_write_source_curation_skipped_for_empty(tmp_path: Path):
    _write_source_curation(str(tmp_path), iteration=0, scores=[])
    assert not (tmp_path / "source-curation.md").exists()
