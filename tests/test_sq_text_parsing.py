"""Verify _parse_sq_texts recovers per-SQ goal text from coverage.chk.

The extractor needs the *content* of each SQ, not just its Q1 label, to run
the goal-aware relevance gate. coverage.chk already normalizes all phase0
plan formats into `## Q1: text`, so we parse that directly.
"""

from deep_research.nodes.phase1a import _parse_sq_texts


def test_parse_basic_entries():
    md = """# Coverage Checklist

## Q1: How does a LangGraph Supervisor route failures to the right worker?
- [ ] advocate — not_started
- [ ] critic — not_started

## Q2: What offline evaluation metrics do auto-ML agents use on tabular tasks?
- [ ] advocate — not_started
"""
    out = _parse_sq_texts(md)
    assert out == {
        "Q1": "How does a LangGraph Supervisor route failures to the right worker?",
        "Q2": "What offline evaluation metrics do auto-ML agents use on tabular tasks?",
    }


def test_skips_placeholder_lines():
    """Early coverage.chk entries that haven't been populated yet must resolve to empty
    so callers degrade back to label-only behavior rather than feeding the LLM junk."""
    md = """## Q1: (to be filled by Phase 1a)
## Q2: Actual text here
"""
    out = _parse_sq_texts(md)
    assert out == {"Q2": "Actual text here"}


def test_empty_coverage_returns_empty():
    assert _parse_sq_texts("") == {}
    assert _parse_sq_texts("# No SQ entries here at all\n") == {}


def test_handles_double_digit_qids():
    """Q10, Q11 etc. must still parse — early code used `\\d` which only matched Q1..Q9."""
    md = "## Q10: tenth subquestion\n## Q11: eleventh one\n"
    out = _parse_sq_texts(md)
    assert out["Q10"] == "tenth subquestion"
    assert out["Q11"] == "eleventh one"
