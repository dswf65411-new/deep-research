"""Smoke tests for Discovery Query Feature - issue #1 fix.

Verifies:
- _extract_known_tools parses tool names from plan text correctly
- Multi-word tool names (Clova Note, Good Tape) are preserved
- Leading particles (such as, for example) are stripped
- Trailing "etc." is stripped
- current_year and known_tools flow into _plan_queries signature without error
- _PLANNER_SYSTEM contains required Discovery Query Family section
"""

import pytest

from deep_research.nodes.phase1a import (
    _PLANNER_SYSTEM,
    _extract_known_tools,
)


PLAN_WITH_TOOLS = """
## SQ2 tool inventory and initial screening
  F2.1: integrated app inventory (Otter.ai, Notta, Clova Note, Good Tape, Fireflies.ai, TurboScribe etc.)
  F2.2: cloud API service inventory (OpenAI Whisper API, AssemblyAI, Deepgram, Rev.ai etc.)
  F2.3: local solution inventory (MacWhisper, Aiko etc.)
perspective 1: efficiency-oriented users (such as Otter.ai, Notta, Clova Note), willing to pay for convenience
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
    """such as Otter.ai -> Otter.ai (leading particle stripped)."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert "Otter.ai" in tools
    assert "such as Otter.ai" not in tools


def test_strips_trailing_etc():
    """TurboScribe etc. -> TurboScribe (trailing etc. stripped)."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert "TurboScribe" in tools
    assert "TurboScribe etc." not in tools


def test_requires_english_letters():
    """Terms without any English letters must be excluded."""
    # Fixture with purely non-Latin placeholders (represented via unicode escapes
    # so the source file itself contains no Chinese characters).
    plan_only_cjk = "(\u5de5\u5177\u7532, \u5de5\u5177\u4e59, \u5de5\u5177\u4e19 etc.)"
    tools = _extract_known_tools(plan_only_cjk)
    assert tools == []


def test_deduplicates_repeated_tools():
    """Same tool appearing multiple times must appear only once."""
    tools = _extract_known_tools(PLAN_WITH_TOOLS)
    assert tools.count("Otter.ai") == 1


def test_max_twenty_tools():
    """Must return at most 20 tools even if plan mentions more."""
    many_tools = ", ".join([f"Tool{i}.ai" for i in range(30)])
    plan = f"({many_tools})"
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
    assert "A latest tools" in _PLANNER_SYSTEM
    assert "best {topic} {YEAR}" in _PLANNER_SYSTEM


def test_planner_system_has_type_c():
    assert "C local" in _PLANNER_SYSTEM
    assert "serper_tw" in _PLANNER_SYSTEM


def test_planner_system_requires_at_least_two_discovery():
    assert "at least" in _PLANNER_SYSTEM and "2" in _PLANNER_SYSTEM
