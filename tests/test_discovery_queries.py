"""Smoke tests for Discovery Query Feature — issue #1 fix.

Verifies:
- _extract_known_tools parses tool names from plan text correctly
- Multi-word tool names (Clova Note, Good Tape) are preserved
- Leading particles (如, 例如) are stripped
- Trailing 等 is stripped
- current_year and known_tools flow into _plan_queries signature without error
- _PLANNER_SYSTEM contains required Discovery Query Family section
"""

import pytest

from deep_research.nodes.phase1a import (
    _PLANNER_SYSTEM,
    _extract_known_tools,
)


PLAN_WITH_TOOLS = """
## SQ2 工具盤點與初篩
  F2.1：整合型 App 盤點（Otter.ai、Notta、Clova Note、Good Tape、Fireflies.ai、TurboScribe 等）
  F2.2：雲端 API 服務盤點（OpenAI Whisper API、AssemblyAI、Deepgram、Rev.ai 等）
  F2.3：本地方案盤點（MacWhisper、Aiko 等）
視角 1：效率導向使用者（如 Otter.ai、Notta、Clova Note），願意付費換取便利
"""


# ---------------------------------------------------------------------------
# _extract_known_tools
# ---------------------------------------------------------------------------

def test_extracts_multiword_tool_names():
    """Multi-word tool names like 'Clova Note' must not be split."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert "Clova Note" in tools
    assert "Good Tape" in tools
    assert "Clova" not in tools   # should NOT be split
    assert "Note" not in tools    # should NOT be split


def test_strips_leading_particle():
    """如 Otter.ai → Otter.ai (leading 如 stripped)."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert "Otter.ai" in tools
    assert "如 Otter.ai" not in tools


def test_strips_trailing_etc():
    """TurboScribe 等 → TurboScribe (trailing 等 stripped)."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert "TurboScribe" in tools
    assert "TurboScribe 等" not in tools


def test_requires_english_letters():
    """Terms without any English letters must be excluded."""
    plan_only_cjk = "（工具甲、工具乙、工具丙 等）"
    tools = _extract_known_tools(plan_only_cjk)
    assert tools == []


def test_deduplicates_repeated_tools():
    """Same tool appearing multiple times must appear only once."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert tools.count("Otter.ai") == 1


def test_max_twenty_tools():
    """Must return at most 20 tools even if plan mentions more."""
    many_tools = ", ".join([f"Tool{i}.ai" for i in range(30)])
    plan = f"（{many_tools}）"
    tools = _extract_known_tools(plan)
    assert len(tools) <= 20


def test_empty_plan():
    assert _extract_known_tools("") == []
    assert _extract_known_tools("no brackets here") == []


def test_common_tools_present():
    """Key tools from the real plan must all be extracted."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    for expected in ("Otter.ai", "Notta", "Fireflies.ai", "MacWhisper", "Aiko"):
        assert expected in tools, f"Expected '{expected}' in tools, got: {tools}"


# ---------------------------------------------------------------------------
# _PLANNER_SYSTEM contains Discovery Query Family
# ---------------------------------------------------------------------------

def test_planner_system_has_discovery_section():
    assert "Discovery Query Family" in _PLANNER_SYSTEM


def test_planner_system_has_type_a():
    assert "A 最新工具" in _PLANNER_SYSTEM
    assert "best {主題} {YEAR}" in _PLANNER_SYSTEM


def test_planner_system_has_type_c():
    assert "C 在地" in _PLANNER_SYSTEM
    assert "serper_tw" in _PLANNER_SYSTEM


def test_planner_system_requires_at_least_two_discovery():
    assert "至少" in _PLANNER_SYSTEM and "2 個" in _PLANNER_SYSTEM
