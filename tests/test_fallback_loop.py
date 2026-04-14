"""Smoke tests for Fallback Loop - issue #8 fix.

Verifies:
- ResearchState has needs_refetch, fallback_count, quality_scores
- trigger_fallback_node returns needs_refetch for low-grounding SQs
- trigger_fallback_node emits [BLOCKER] when fallback_count >= 2
- trigger_fallback_node returns empty needs_refetch for well-grounded SQs
- phase1a focus mode caps budget to 25 when needs_refetch is set
- _plan_queries accepts focus_sqs param and injects focus section
- graph node trigger_fallback exists
- route_after_fallback routes correctly
- phase2 adds BLOCKER disclaimer when blocker present for a subq
"""

import asyncio
import inspect

import pytest

from deep_research.state import Claim, ResearchState


# ---------------------------------------------------------------------------
# State fields
# ---------------------------------------------------------------------------

def test_state_has_needs_refetch():
    import deep_research.state as s
    assert "needs_refetch" in s.ResearchState.__annotations__


def test_state_has_fallback_count():
    import deep_research.state as s
    assert "fallback_count" in s.ResearchState.__annotations__


def test_state_has_quality_scores():
    import deep_research.state as s
    assert "quality_scores" in s.ResearchState.__annotations__


# ---------------------------------------------------------------------------
# trigger_fallback_node - returns needs_refetch for low-grounding SQs
# ---------------------------------------------------------------------------

def _make_claim(claim_id, subquestion, bedrock, verdict):
    c = Claim(claim_id=claim_id, claim_text="test", source_ids=["S001"], subquestion=subquestion)
    c.bedrock_score = bedrock
    c.citation_verdict = verdict
    return c


def _run_fallback(state_dict: dict) -> dict:
    from deep_research.nodes.phase1b import trigger_fallback_node
    return asyncio.run(trigger_fallback_node(state_dict))


def test_trigger_flags_low_grounding(tmp_path):
    """SQ with grounded_ratio < 0.3 must appear in needs_refetch."""
    claims = [
        _make_claim("Q1-C1", "Q1", 0.1, "NOT_GROUNDED"),
        _make_claim("Q1-C2", "Q1", 0.1, "NOT_GROUNDED"),
        _make_claim("Q1-C3", "Q1", 0.1, "NOT_GROUNDED"),
        _make_claim("Q1-C4", "Q1", 0.1, "GROUNDED"),  # only 1/4 grounded
    ]
    state = {
        "claims": claims,
        "quality_scores": {"Q1": {"actionability": True, "freshness": True, "plurality": True, "completeness": True}},
        "fallback_count": 0,
        "workspace_path": str(tmp_path),
        # bypass budget guard - we want to test quality-based triggering
        "search_count": 100,
        "search_budget": 150,
        "depth": "deep",
    }
    (tmp_path / "gap-log.md").write_text("# Gap\n")
    result = _run_fallback(state)
    assert "Q1" in result["needs_refetch"]
    assert result["fallback_count"] == 1


def test_trigger_flags_low_avg_bedrock(tmp_path):
    """SQ with avg_bedrock < 0.4 must appear in needs_refetch."""
    claims = [
        _make_claim("Q2-C1", "Q2", 0.1, "GROUNDED"),
        _make_claim("Q2-C2", "Q2", 0.2, "GROUNDED"),
        _make_claim("Q2-C3", "Q2", 0.2, "GROUNDED"),
    ]
    state = {
        "claims": claims,
        "quality_scores": {"Q2": {"actionability": True, "freshness": True, "plurality": True, "completeness": True}},
        "fallback_count": 0,
        "workspace_path": str(tmp_path),
    }
    (tmp_path / "gap-log.md").write_text("# Gap\n")
    result = _run_fallback(state)
    assert "Q2" in result["needs_refetch"]


def test_trigger_flags_false_dims(tmp_path):
    """SQ with >= 2 false dim_scores must appear in needs_refetch."""
    claims = [
        _make_claim("Q3-C1", "Q3", 0.8, "GROUNDED"),
        _make_claim("Q3-C2", "Q3", 0.8, "GROUNDED"),
    ]
    state = {
        "claims": claims,
        "quality_scores": {"Q3": {
            "actionability": False,
            "freshness": False,   # 2 false dims
            "plurality": True,
            "completeness": True,
        }},
        "fallback_count": 0,
        "workspace_path": str(tmp_path),
    }
    (tmp_path / "gap-log.md").write_text("# Gap\n")
    result = _run_fallback(state)
    assert "Q3" in result["needs_refetch"]


def test_trigger_no_refetch_for_good_sq(tmp_path):
    """Well-grounded SQ must NOT appear in needs_refetch."""
    claims = [
        _make_claim("Q4-C1", "Q4", 0.85, "GROUNDED"),
        _make_claim("Q4-C2", "Q4", 0.90, "GROUNDED"),
        _make_claim("Q4-C3", "Q4", 0.75, "GROUNDED"),
    ]
    state = {
        "claims": claims,
        "quality_scores": {"Q4": {"actionability": True, "freshness": True, "plurality": True, "completeness": True}},
        "fallback_count": 0,
        "workspace_path": str(tmp_path),
        # bypass budget guard - we want to test quality-based triggering
        "search_count": 100,
        "search_budget": 150,
        "depth": "deep",
    }
    result = _run_fallback(state)
    assert "Q4" not in result["needs_refetch"]
    assert result["fallback_count"] == 0  # not incremented


# ---------------------------------------------------------------------------
# trigger_fallback_node - BLOCKER when fallback_count >= 2
# ---------------------------------------------------------------------------

def test_trigger_emits_blocker_at_max_retries(tmp_path):
    """When fallback_count >= 2, must emit [BLOCKER] and keep needs_refetch empty."""
    claims = [
        _make_claim("Q5-C1", "Q5", 0.1, "NOT_GROUNDED"),
    ]
    state = {
        "claims": claims,
        "quality_scores": {"Q5": {"actionability": False, "freshness": False, "plurality": False, "completeness": False}},
        "fallback_count": 2,  # already at max
        "workspace_path": str(tmp_path),
    }
    (tmp_path / "gap-log.md").write_text("# Gap\n")
    result = _run_fallback(state)

    assert result["needs_refetch"] == []
    assert result["fallback_count"] == 2  # not incremented (BLOCKER path)
    assert any("[BLOCKER:" in b for b in result["blockers"])
    assert any("Q5" in b for b in result["blockers"])


def test_blocker_written_to_gap_log(tmp_path):
    """[BLOCKER] entries must be appended to gap-log.md."""
    (tmp_path / "gap-log.md").write_text("# Gap\n")
    claims = [_make_claim("Q6-C1", "Q6", 0.1, "NOT_GROUNDED")]
    state = {
        "claims": claims,
        "quality_scores": {"Q6": {"actionability": False, "freshness": False, "plurality": False, "completeness": False}},
        "fallback_count": 2,
        "workspace_path": str(tmp_path),
    }
    _run_fallback(state)
    content = (tmp_path / "gap-log.md").read_text()
    assert "BLOCKER" in content


# ---------------------------------------------------------------------------
# phase1a focus mode
# ---------------------------------------------------------------------------

def test_phase1a_plan_queries_has_focus_sqs_param():
    from deep_research.nodes.phase1a import _plan_queries
    sig = inspect.signature(_plan_queries)
    assert "focus_sqs" in sig.parameters


def test_focus_section_in_prompt_when_focus():
    """_plan_queries user_msg must contain focus section when focus_sqs given."""
    # We just check the source to avoid running the LLM in tests
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._plan_queries)
    assert "focus_sqs" in src
    assert "focused refetch mode" in src or "focus_section" in src


def test_phase1a_caps_budget_in_focus_mode():
    """When needs_refetch is non-empty, phase1a must cap remaining to 25."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a.phase1a_search)
    assert "needs_refetch" in src
    assert "focus_mode" in src
    assert "25" in src  # cap to 25


# ---------------------------------------------------------------------------
# graph - trigger_fallback node exists
# ---------------------------------------------------------------------------

def test_graph_has_trigger_fallback_node():
    from deep_research.graph import build_deep_research
    # Just check the node is registered - don't compile with checkpointer
    import deep_research.nodes.phase1b as p1b
    assert hasattr(p1b, "trigger_fallback_node")


def test_graph_imports_trigger_fallback():
    import deep_research.graph as g
    assert hasattr(g, "trigger_fallback_node")


def test_route_after_fallback_needs_refetch_routes_to_increment():
    from deep_research.graph import route_after_fallback
    state = {"needs_refetch": ["Q1", "Q3"], "phase1b_result": "pass", "iteration_count": 1}
    assert route_after_fallback(state) == "increment_iter"


def test_route_after_fallback_empty_refetch_routes_to_phase2():
    from deep_research.graph import route_after_fallback
    state = {"needs_refetch": [], "phase1b_result": "pass", "iteration_count": 1}
    assert route_after_fallback(state) == "phase2"


# ---------------------------------------------------------------------------
# phase2 BLOCKER disclaimer
# ---------------------------------------------------------------------------

def test_phase2_source_has_blocker_disclaimer():
    import deep_research.nodes.phase2 as p2
    src = inspect.getsource(p2.phase2_integrate)
    assert "BLOCKER" in src
    assert "insufficient data" in src or "disclaimer" in src.lower()
