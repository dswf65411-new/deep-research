"""Deterministic gate-check functions — no LLM involved."""

from __future__ import annotations

from deep_research.state import ResearchState


def gate_check(state: ResearchState) -> str:
    """Decide whether Phase 1 loop should continue or pass to Phase 2.

    Returns:
        "pass"        – all required coverage items checked, all claims resolved
        "fail"        – need more iteration (back to Phase 1a)
        "max_retries" – hard stop after 3 iterations (Fail-Fast)
    """
    iteration = state.get("iteration_count", 0)
    if iteration >= 3:
        return "max_retries"

    # Coverage: every required facet must be checked or searched_2x
    coverage = state.get("coverage_status", {})
    required_facets = [
        f for f in coverage
        if not f.endswith("(optional)")
    ]
    all_covered = all(
        coverage[f] in ("checked", "searched_2x_no_evidence")
        for f in required_facets
    ) if required_facets else True

    # Claims: every claim must be approved or rejected (no pending / needs_revision)
    claims = state.get("claims", [])
    all_resolved = all(
        c.status in ("approved", "rejected") if hasattr(c, "status") else
        c.get("status") in ("approved", "rejected")
        for c in claims
    ) if claims else False  # no claims at all → fail

    if all_covered and all_resolved:
        return "pass"
    return "fail"


def budget_check(state: ResearchState) -> str:
    """Check search budget status.

    Returns:
        "ok"         – plenty of budget left
        "beast_mode" – >=80% used, stop opening new facets
        "exhausted"  – 100% used
    """
    budget = state.get("search_budget", 150)
    used = state.get("search_count", 0)
    if budget <= 0:
        return "exhausted"
    ratio = used / budget
    if ratio >= 1.0:
        return "exhausted"
    if ratio >= 0.8:
        return "beast_mode"
    return "ok"


def quality_gate(quality_scores: dict) -> tuple[str, list[str]]:
    """Evaluate the 4-dimension quality check.

    Args:
        quality_scores: {dimension: True/False} for
            actionability, freshness, plurality, completeness

    Returns:
        ("all_pass", []) or ("needs_attack", [failed_dimensions])
    """
    dimensions = ["actionability", "freshness", "plurality", "completeness"]
    failed = [d for d in dimensions if not quality_scores.get(d, False)]
    if not failed:
        return "all_pass", []
    return "needs_attack", failed
