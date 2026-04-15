"""Verify `think_tool`-style planner reflection appended to sq-progress.md.

open_deep_research forces the planner to invoke a `think_tool` after every
batch of results and list (a) what is missing and (b) the next step. Without
this, the planner drifts and reruns near-duplicate queries. We emulate that
with one small LLM call that appends `## Iteration N — planner reflection`
to sq-progress.md so the next round's planner reads the explicit reflection.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from deep_research.nodes.phase1a import (
    _REFLECTION_SYSTEM,
    _write_planner_reflection,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def test_reflection_system_prompt_has_required_anchors():
    """Grep-anchor so a future rewrite cannot silently drop the rule."""
    assert "missing:" in _REFLECTION_SYSTEM
    assert "next step:" in _REFLECTION_SYSTEM
    assert "DUPLICATE ROLLBACK" in _REFLECTION_SYSTEM


def test_reflection_appended_to_sq_progress(tmp_path: Path):
    existing = "## Iteration 1 — per-SQ evidence snapshot\n- **Q1**: ...\n"
    (tmp_path / "sq-progress.md").write_text(existing)

    fake_llm_reply = (
        "- **Q1**:\n"
        "  - missing: Tongyi DeepResearch arxiv citation count\n"
        "  - next step: query `Tongyi DeepResearch arxiv` with serper_scholar\n"
    )

    async def fake_chain(*args, **kwargs):
        return _FakeResponse(fake_llm_reply)

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        asyncio.run(_write_planner_reflection(
            workspace=str(tmp_path),
            iteration=0,
            sq_progress_snapshot="## Iteration 1 snapshot body",
            coverage="## Q1: done [ ]",
            gap_log="(empty)",
        ))

    text = (tmp_path / "sq-progress.md").read_text()
    # Existing snapshot preserved
    assert existing in text
    # Reflection section appended
    assert "## Iteration 1 — planner reflection" in text
    # LLM-generated body present
    assert "Tongyi DeepResearch arxiv citation count" in text


def test_reflection_skipped_when_llm_returns_empty(tmp_path: Path):
    (tmp_path / "sq-progress.md").write_text("## Iteration 1 snapshot\n")

    async def fake_chain(*args, **kwargs):
        return _FakeResponse("")

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        asyncio.run(_write_planner_reflection(
            workspace=str(tmp_path),
            iteration=0,
            sq_progress_snapshot="snapshot",
            coverage="coverage",
            gap_log="gap",
        ))

    text = (tmp_path / "sq-progress.md").read_text()
    assert "planner reflection" not in text


def test_reflection_swallows_exceptions(tmp_path: Path):
    """The planner loop must not die if the reflection LLM errors."""
    async def fake_chain(*args, **kwargs):
        raise RuntimeError("upstream LLM timeout")

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        # No exception should bubble up
        asyncio.run(_write_planner_reflection(
            workspace=str(tmp_path),
            iteration=0,
            sq_progress_snapshot="snapshot",
            coverage="coverage",
            gap_log="gap",
        ))
    # File not created (we had no pre-existing content)
    assert not (tmp_path / "sq-progress.md").exists() or \
        "planner reflection" not in (tmp_path / "sq-progress.md").read_text()


def test_reflection_skipped_for_empty_inputs(tmp_path: Path):
    """If both snapshot and coverage are empty there's nothing to reflect on."""
    async def fake_chain(*args, **kwargs):
        raise AssertionError("LLM should NOT be invoked with empty context")

    with patch(
        "deep_research.nodes.phase1a.safe_ainvoke_chain",
        new=AsyncMock(side_effect=fake_chain),
    ):
        asyncio.run(_write_planner_reflection(
            workspace=str(tmp_path),
            iteration=0,
            sq_progress_snapshot="",
            coverage="",
            gap_log="",
        ))
