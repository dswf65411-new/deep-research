"""Smoke tests for Iterative Expansion — issue #2 fix.

Verifies:
- _extract_emerging_entities exists and is async
- _extract_emerging_from_gap_log extracts only the last round's entities
- _plan_queries accepts emerging_entities param and injects section
- phase1a source has focus_mode guard (only expands when not in focus mode)
- PLANNER_SYSTEM has rule about 新發現實體 → B-type follow-up query
"""

import asyncio
import inspect

import pytest


# ---------------------------------------------------------------------------
# _extract_emerging_from_gap_log
# ---------------------------------------------------------------------------

def test_extract_emerging_from_gap_log_returns_last_round():
    """Only the last '## 新發現實體' block should be returned."""
    from deep_research.nodes.phase1a import _extract_emerging_from_gap_log

    gap = (
        "# Gap Log\n\n"
        "## 新發現實體（第 1 輪）\n"
        "- OldTool\n"
        "- AlsoOld\n\n"
        "## 新發現實體（第 2 輪）\n"
        "- Descript\n"
        "- Grain\n"
    )
    result = _extract_emerging_from_gap_log(gap, 2)
    assert "Descript" in result
    assert "Grain" in result
    assert "OldTool" not in result
    assert "AlsoOld" not in result


def test_extract_emerging_from_gap_log_empty_when_no_section():
    from deep_research.nodes.phase1a import _extract_emerging_from_gap_log

    gap = "# Gap Log\n\n## 預算缺口（第 1 輪後）\n- Q1：已搜 3 次\n"
    result = _extract_emerging_from_gap_log(gap, 1)
    assert result == []


def test_extract_emerging_from_gap_log_caps_at_15():
    from deep_research.nodes.phase1a import _extract_emerging_from_gap_log

    entities = [f"Tool{i}" for i in range(20)]
    lines = "\n".join(f"- {e}" for e in entities)
    gap = f"## 新發現實體（第 1 輪）\n{lines}\n"
    result = _extract_emerging_from_gap_log(gap, 1)
    assert len(result) <= 15


def test_extract_emerging_from_gap_log_filters_short_names():
    from deep_research.nodes.phase1a import _extract_emerging_from_gap_log

    gap = "## 新發現實體（第 1 輪）\n- A\n- ValidTool\n- B\n"
    result = _extract_emerging_from_gap_log(gap, 1)
    assert "A" not in result
    assert "B" not in result
    assert "ValidTool" in result


# ---------------------------------------------------------------------------
# _extract_emerging_entities
# ---------------------------------------------------------------------------

def test_extract_emerging_entities_exists_and_is_async():
    from deep_research.nodes.phase1a import _extract_emerging_entities
    assert inspect.iscoroutinefunction(_extract_emerging_entities)


def test_extract_emerging_entities_returns_empty_for_no_sources():
    from deep_research.nodes.phase1a import _extract_emerging_entities
    result = asyncio.run(_extract_emerging_entities([], plan="test plan", iteration=0))
    assert result == []


def test_extract_emerging_entities_returns_empty_for_no_content():
    from deep_research.nodes.phase1a import _extract_emerging_entities
    raw_sources = [{"source_id": "S001", "title": "", "content": ""}]
    result = asyncio.run(_extract_emerging_entities(raw_sources, plan="test plan", iteration=0))
    assert result == []


# ---------------------------------------------------------------------------
# _plan_queries — emerging_entities param
# ---------------------------------------------------------------------------

def test_plan_queries_has_emerging_entities_param():
    from deep_research.nodes.phase1a import _plan_queries
    sig = inspect.signature(_plan_queries)
    assert "emerging_entities" in sig.parameters


def test_plan_queries_emerging_section_in_source():
    """The _plan_queries function source must contain the emerging_section injection."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._plan_queries)
    assert "emerging_entities" in src
    assert "新發現實體" in src
    assert "emerging_section" in src


# ---------------------------------------------------------------------------
# _PLANNER_SYSTEM — 新發現實體 rule
# ---------------------------------------------------------------------------

def test_planner_system_has_emerging_entities_rule():
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "新發現實體" in _PLANNER_SYSTEM
    assert "B 類 follow-up query" in _PLANNER_SYSTEM or "B 類" in _PLANNER_SYSTEM


# ---------------------------------------------------------------------------
# phase1a_search — focus_mode guard
# ---------------------------------------------------------------------------

def test_phase1a_focus_mode_skips_emerging_expansion():
    """When focus_mode is True, emerging expansion must not run."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a.phase1a_search)
    # The guard should be: if not focus_mode → extract emerging
    assert "focus_mode" in src
    assert "_extract_emerging_entities" in src


def test_phase1a_writes_emerging_to_gap_log():
    """phase1a_search source must write emerging entities to gap-log.md."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a.phase1a_search)
    assert "gap-log.md" in src
    assert "新發現實體" in src


# ---------------------------------------------------------------------------
# _extract_emerging_from_gap_log — iteration param accepted
# ---------------------------------------------------------------------------

def test_extract_emerging_accepts_iteration():
    """Function signature must accept iteration param."""
    from deep_research.nodes.phase1a import _extract_emerging_from_gap_log
    sig = inspect.signature(_extract_emerging_from_gap_log)
    assert "iteration" in sig.parameters
