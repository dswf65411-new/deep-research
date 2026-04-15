"""Tests for Whisper P3-1 — staged clarification (round 1 scope, round N+ depth).

The failed workspace asked ten questions in a single shot, some of which (team
size, tech-stack preference) diluted attention and pushed the user toward
fatigue-filled one-word answers. The fix narrows round 1 to ≤3 *core scope*
questions; the Judge drives later rounds if needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deep_research.nodes import phase0
from deep_research.nodes.phase0 import FIRST_ROUND_CORE_CAP, generate_questions


def _fake_response(questions: list[str], reasoning: str = "mock") -> SimpleNamespace:
    import json
    payload = json.dumps({"questions": questions, "reasoning": reasoning})
    return SimpleNamespace(content=payload)


@pytest.mark.anyio
async def test_round1_prompt_caps_at_three_questions():
    """Even when caller allows 10, round 1 must ask ≤ FIRST_ROUND_CORE_CAP."""
    recorded_prompt = {}

    async def fake_chain(role, messages, **kwargs):
        # System message content = messages[0].content
        recorded_prompt["system"] = messages[0].content
        return _fake_response(["q1", "q2", "q3", "q4", "q5"])

    with patch.object(phase0, "safe_ainvoke_chain", side_effect=fake_chain):
        questions, _reasoning = await generate_questions(
            topic="langgraph supervisor pattern",
            existing_clarifications=[],
            max_questions=10,
            round_num=1,
        )

    # The slice happens on the return, so even if the mock returned 5 we
    # only keep FIRST_ROUND_CORE_CAP.
    assert len(questions) == FIRST_ROUND_CORE_CAP
    # The prompt itself should instruct the LLM to generate at most 3.
    assert f"Generate at most {FIRST_ROUND_CORE_CAP} questions" in recorded_prompt["system"]
    # And round 1 is labelled the scoping round.
    assert "scoping" in recorded_prompt["system"].lower()


@pytest.mark.anyio
async def test_round1_excludes_narrow_topics_from_prompt():
    recorded_prompt = {}

    async def fake_chain(role, messages, **kwargs):
        recorded_prompt["system"] = messages[0].content
        return _fake_response(["scope-q"])

    with patch.object(phase0, "safe_ainvoke_chain", side_effect=fake_chain):
        await generate_questions(
            topic="x",
            existing_clarifications=[],
            max_questions=10,
            round_num=1,
        )

    sys_text = recorded_prompt["system"].lower()
    # The round-1 note explicitly forbids narrow detail probes.
    assert "do not" in sys_text or "don't" in sys_text
    assert "tech-stack" in sys_text
    assert "budget" in sys_text


@pytest.mark.anyio
async def test_round2_uses_full_max_questions():
    """Round 2+ uses the caller-provided ``max_questions`` fully."""
    recorded_prompt = {}

    async def fake_chain(role, messages, **kwargs):
        recorded_prompt["system"] = messages[0].content
        return _fake_response(["d1", "d2", "d3", "d4", "d5", "d6"])

    with patch.object(phase0, "safe_ainvoke_chain", side_effect=fake_chain):
        questions, _ = await generate_questions(
            topic="x",
            existing_clarifications=[{"question": "scope?", "answer": "langgraph"}],
            max_questions=6,
            round_num=2,
        )

    assert len(questions) == 6
    assert "Generate at most 6 questions" in recorded_prompt["system"]
    # Round 2 prompt is the "depth" round.
    assert "depth" in recorded_prompt["system"].lower()


@pytest.mark.anyio
async def test_max_questions_smaller_than_cap_is_respected():
    """If caller passes max_questions=2 on round 1, we cap at 2, not 3."""

    async def fake_chain(role, messages, **kwargs):
        return _fake_response(["q1", "q2"])

    with patch.object(phase0, "safe_ainvoke_chain", side_effect=fake_chain):
        questions, _ = await generate_questions(
            topic="x",
            existing_clarifications=[],
            max_questions=2,
            round_num=1,
        )

    assert len(questions) == 2


@pytest.mark.anyio
async def test_first_round_core_cap_constant_is_small():
    """Guard against someone bumping the cap back to 10."""
    assert 1 <= FIRST_ROUND_CORE_CAP <= 4


@pytest.fixture
def anyio_backend():
    return "asyncio"
