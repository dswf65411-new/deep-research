"""Verify the planner prompt contains the year-bias rule for academic topics.

Context from the 80-issue audit: the planner was appending `{YEAR}` to every
category of query. For academic topics that meant AIDE (2024) and MLE-bench (2024)
were silently excluded from search results. The fix is a prompt rule telling the
LLM that E-type (arxiv/github/paper pdf) queries MUST NOT append {YEAR}.
"""

from deep_research.nodes.phase1a import _PLANNER_SYSTEM


def test_prompt_has_year_bias_rule():
    """The exact phrases used in the rule should appear so downstream tooling
    can grep for them if the prompt ever gets rewritten."""
    # The policy block heading and the key imperative for academic topics.
    assert "Year-window policy" in _PLANNER_SYSTEM
    assert "do NOT append any year" in _PLANNER_SYSTEM


def test_prompt_cites_prior_year_examples():
    """The rationale (prior-year leading papers) must stay in the prompt so the
    LLM understands why, not just the what."""
    assert "AIDE 2024" in _PLANNER_SYSTEM or "MLE-bench 2024" in _PLANNER_SYSTEM


def test_prompt_off_topic_gap_rule_references_gap_log_marker():
    """The `[OFF_TOPIC_RATIO >= 0.5]` marker written by _log_off_topic_ratio
    must be the exact string the planner is told to look for — otherwise the
    closed-loop signal is invisible."""
    assert "[OFF_TOPIC_RATIO >= 0.5]" in _PLANNER_SYSTEM
    assert "rewrite" in _PLANNER_SYSTEM.lower()


def test_prompt_scholar_description_updated():
    """The old description mentioned the `site:arxiv.org hack`; after switching
    to the real /scholar endpoint, the description must reflect the real API
    features (citation counts, pdfUrl) so the LLM picks it for E-type queries."""
    # New language:
    assert "dedicated Google Scholar API" in _PLANNER_SYSTEM
    assert "citation counts" in _PLANNER_SYSTEM
    # Stale language gone:
    assert "Google + site:arxiv.org/semanticscholar.org" not in _PLANNER_SYSTEM
