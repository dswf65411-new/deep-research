"""Tests for `_extract_brief_entities` (P0-3).

Reproduces the 2026-04-14 failure: the brief mentions LangGraph Supervisor
Pattern, AutoML-Agent, MLE-Agent, AIDE, ResearchAgent, MLR-Copilot,
DS-Agent, OpenAI deep research - but phase1a's query generator never picks
these up because it only reads parenthetical clauses from the plan, not
the brief's labelled list / bullet list / inline list.

This test asserts the new extractor catches at least 6 of them.
"""

from __future__ import annotations

from deep_research.nodes.phase1a import _extract_brief_entities


# ---------------------------------------------------------------------------
# Pattern A - inline comma list
# ---------------------------------------------------------------------------

class TestPatternAInlineList:
    def test_english_comma_list_of_three(self):
        text = "Mainstream tools include AutoML-Agent, MLE-Agent, AIDE (three packages)."
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        assert "MLE-Agent" in out
        assert "AIDE" in out

    def test_longer_comma_list(self):
        text = "This study mainly compares AutoML-Agent, MLE-Agent, AIDE, ResearchAgent, MLR-Copilot and similar tools."
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        assert "MLE-Agent" in out
        assert "AIDE" in out
        assert "ResearchAgent" in out
        assert "MLR-Copilot" in out

    def test_comma_list_four_items(self):
        text = "AIDE, MLE-Agent, ResearchAgent, AutoML-Agent are four comparable packages."
        out = _extract_brief_entities(text)
        assert {"AIDE", "MLE-Agent", "ResearchAgent", "AutoML-Agent"}.issubset(set(out))

    def test_two_items_not_enough(self):
        """Only 2 capitalized items -> Pattern A should not fire (avoid false positives)."""
        text = "One thing is Apple, another is Bar."
        # Apple is caps; Bar is caps; only 2 - should not trigger Pattern A
        out = _extract_brief_entities(text)
        # Not guaranteed to be empty (bullet/label patterns don't fire either),
        # but Apple/Bar should not both appear from Pattern A alone.
        assert not ({"Apple", "Bar"}.issubset(set(out)))

    def test_strips_leading_helpers(self):
        text = "e.g. AutoML-Agent, MLE-Agent, AIDE etc."
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        # "AIDE etc." -> strip "etc." -> "AIDE"
        assert "AIDE" in out


# ---------------------------------------------------------------------------
# Pattern B - markdown bullet list
# ---------------------------------------------------------------------------

class TestPatternBBullets:
    def test_dash_bullet(self):
        text = """
known tools:
- AIDE: 2024 SOTA paper
- MLE-Agent: Cursor-style tool
- ResearchAgent
"""
        out = _extract_brief_entities(text)
        assert "AIDE" in out
        assert "MLE-Agent" in out
        assert "ResearchAgent" in out

    def test_asterisk_bullet_with_paren_desc(self):
        text = """
* AutoML-Agent (Ma et al., 2024)
* MLE-Agent (MLR-Copilot follow-up)
"""
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        assert "MLE-Agent" in out

    def test_bullet_with_dash_separator(self):
        text = "- DS-Agent - data science copilot"
        out = _extract_brief_entities(text)
        assert "DS-Agent" in out

    def test_bullet_lowercase_prose_skipped(self):
        """Bullets that start with lowercase prose shouldn't be extracted."""
        text = "- this bullet is prose, not an entity"
        out = _extract_brief_entities(text)
        assert out == []


# ---------------------------------------------------------------------------
# Pattern C - labelled list ("facets: A, B, C", "known tools: X, Y")
# ---------------------------------------------------------------------------

class TestPatternCLabelledList:
    def test_facets_label(self):
        text = "facets: AutoML-Agent, MLE-Agent, AIDE, ResearchAgent"
        out = _extract_brief_entities(text)
        assert {"AutoML-Agent", "MLE-Agent", "AIDE", "ResearchAgent"}.issubset(set(out))

    def test_english_label_evaluation_targets(self):
        text = "evaluation targets: LangGraph, AutoML-Agent, MLE-Agent"
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        assert "MLE-Agent" in out
        assert "LangGraph" in out

    def test_known_entities_label(self):
        text = "known tools: AIDE, ResearchAgent, MLR-Copilot"
        out = _extract_brief_entities(text)
        assert "AIDE" in out
        assert "ResearchAgent" in out
        assert "MLR-Copilot" in out

    def test_strips_parenthetical_in_label_value(self):
        text = "facets: AutoML-Agent (Ma 2024), MLE-Agent (Zhang 2024), AIDE"
        out = _extract_brief_entities(text)
        assert "AutoML-Agent" in out
        assert "MLE-Agent" in out
        assert "AIDE" in out


# ---------------------------------------------------------------------------
# Parenthetical fallback
# ---------------------------------------------------------------------------

class TestParentheticalFallback:
    def test_paren_list(self):
        text = "We chose several representative agents (AIDE, ResearchAgent, AutoML-Agent) for comparison"
        out = _extract_brief_entities(text)
        assert "AIDE" in out
        assert "ResearchAgent" in out
        assert "AutoML-Agent" in out


# ---------------------------------------------------------------------------
# Dedup / normalization
# ---------------------------------------------------------------------------

class TestDedupAndNormalize:
    def test_dedup_case_insensitive(self):
        text = """
facets: aide, AIDE, AIDE
- AIDE
"""
        out = _extract_brief_entities(text)
        lower = [x.lower() for x in out]
        assert lower.count("aide") == 1

    def test_cap_at_20(self):
        entities = [f"Tool{i}" for i in range(30)]
        text = "facets: " + ", ".join(entities)
        out = _extract_brief_entities(text)
        assert len(out) <= 20

    def test_strips_trailing_meta_suffix(self):
        text = """
known tools:
- AIDE tool
- MLE-Agent service
- ResearchAgent framework
"""
        out = _extract_brief_entities(text)
        # "AIDE tool" -> strip "tool" -> "AIDE"
        assert "AIDE" in out
        assert "MLE-Agent" in out
        assert "ResearchAgent" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        assert _extract_brief_entities("") == []

    def test_no_entities(self):
        text = "This is a plain paragraph without any tool names."
        out = _extract_brief_entities(text)
        # `This`, `is`, etc are English but pattern A requires comma-list >=3,
        # pattern B requires bullet. Should be empty or near-empty.
        # We're lenient - just check obvious prose doesn't produce tool-looking junk.
        assert all(len(e) <= 60 for e in out)

    def test_too_short_skipped(self):
        # Single-letter tokens shouldn't be accepted
        text = "facets: A, B, C, D, E"
        out = _extract_brief_entities(text)
        # Each is 1 char -> filtered out
        assert all(len(e) >= 2 for e in out)


# ---------------------------------------------------------------------------
# Failure-workspace regression - >=6 of the 2026-04-14 SOTA list
# ---------------------------------------------------------------------------

def test_reproduces_failure_brief_extracts_six_plus_agents():
    """End-to-end smoke test modelled on the 2026-04-14 failure brief.

    The real brief listed these SOTA agents both inline and in a bullet list;
    phase1a's old extractor missed every one of them. New extractor must
    recover at least 6.
    """
    brief_like = """
# Research Brief

## Topic
How to design a LangGraph Supervisor Pattern auto-ML optimization agent.

## Core comparison targets
known tools: AutoML-Agent, MLE-Agent, AIDE, ResearchAgent, MLR-Copilot, DS-Agent, OpenAI deep research

## Characteristics of each target
- AutoML-Agent - 2024 automated AutoML paper
- MLE-Agent - Cursor-style ML engineering copilot
- AIDE - 2024 SOTA ML engineer agent
- ResearchAgent - research automation prototype
- MLR-Copilot - paper writing automation
- DS-Agent - data science copilot

## Further reading
Agents combined with LangGraph and Supervisor Pattern (AIDE, MLE-Agent, AutoML-Agent).
"""
    out = _extract_brief_entities(brief_like)
    sota = {
        "AutoML-Agent",
        "MLE-Agent",
        "AIDE",
        "ResearchAgent",
        "MLR-Copilot",
        "DS-Agent",
    }
    matched = sota.intersection(out)
    assert len(matched) >= 6, f"expected >=6 SOTA matches, got {matched} (all={out})"
