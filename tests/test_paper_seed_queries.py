"""Tests for `_seed_paper_queries` (P0-3).

`_seed_paper_queries` is the pure-template companion to
`_extract_brief_entities` — no LLM, no network. For every brief-level
entity it must emit up to 2 queries (arxiv + github) tagged role="seed"
so downstream quota logic (_QUOTA_PER_ROLE["seed"]=6) treats them as a
separate deep-read bucket.

Covers:
  - basic 2-query-per-entity emission
  - budget cap truncation
  - round-robin SQ assignment
  - engine selection (arxiv → serper_scholar + brave; github → brave + serper_en)
  - role is always "seed"
  - language is always "en" (academic/code search is in English)
  - empty entities / zero budget → []
"""

from __future__ import annotations

from deep_research.nodes.phase1a import _seed_paper_queries


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestBasic:
    def test_two_queries_per_entity(self):
        """entities=[AIDE, MLE-Agent] → 4 queries (2 arxiv + 2 github)."""
        out = _seed_paper_queries(["AIDE", "MLE-Agent"], ["Q1"], budget_cap=10)
        assert len(out) == 4
        queries = [q["query"] for q in out]
        assert "AIDE arxiv" in queries
        assert "AIDE github" in queries
        assert "MLE-Agent arxiv" in queries
        assert "MLE-Agent github" in queries

    def test_all_role_seed(self):
        out = _seed_paper_queries(["A"], ["Q1"], budget_cap=10)
        assert all(q["role"] == "seed" for q in out)

    def test_all_lang_en(self):
        out = _seed_paper_queries(["AIDE"], ["Q1"], budget_cap=10)
        assert all(q["lang"] == "en" for q in out)

    def test_arxiv_engines(self):
        out = _seed_paper_queries(["AIDE"], ["Q1"], budget_cap=10)
        arxiv_q = next(q for q in out if q["query"] == "AIDE arxiv")
        assert set(arxiv_q["engines"]) == {"serper_scholar", "brave"}

    def test_github_engines(self):
        out = _seed_paper_queries(["AIDE"], ["Q1"], budget_cap=10)
        github_q = next(q for q in out if q["query"] == "AIDE github")
        assert set(github_q["engines"]) == {"brave", "serper_en"}


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------

class TestBudgetCap:
    def test_cap_truncates(self):
        """6 entities × 2 queries = 12; cap=5 → 5 queries."""
        entities = [f"Tool{i}" for i in range(6)]
        out = _seed_paper_queries(entities, ["Q1"], budget_cap=5)
        assert len(out) == 5

    def test_zero_cap_returns_empty(self):
        assert _seed_paper_queries(["A"], ["Q1"], budget_cap=0) == []

    def test_negative_cap_returns_empty(self):
        assert _seed_paper_queries(["A"], ["Q1"], budget_cap=-5) == []

    def test_cap_larger_than_need(self):
        out = _seed_paper_queries(["A", "B"], ["Q1"], budget_cap=1000)
        # 2 entities × 2 queries = 4
        assert len(out) == 4


# ---------------------------------------------------------------------------
# SQ round-robin
# ---------------------------------------------------------------------------

class TestSqRoundRobin:
    def test_round_robin_across_multiple_sqs(self):
        """6 queries, 3 SQs → each SQ gets exactly 2."""
        out = _seed_paper_queries(["A", "B", "C"], ["Q1", "Q2", "Q3"], budget_cap=10)
        assert len(out) == 6
        sq_counts: dict[str, int] = {}
        for q in out:
            sq_counts[q["subquestion"]] = sq_counts.get(q["subquestion"], 0) + 1
        assert sq_counts == {"Q1": 2, "Q2": 2, "Q3": 2}

    def test_none_sq_ids_defaults_q1(self):
        out = _seed_paper_queries(["A"], None, budget_cap=10)
        assert all(q["subquestion"] == "Q1" for q in out)

    def test_empty_sq_ids_defaults_q1(self):
        out = _seed_paper_queries(["A"], [], budget_cap=10)
        assert all(q["subquestion"] == "Q1" for q in out)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_entities(self):
        assert _seed_paper_queries([], ["Q1"], budget_cap=10) == []

    def test_single_entity(self):
        out = _seed_paper_queries(["AIDE"], ["Q1"], budget_cap=10)
        assert len(out) == 2
        assert {q["query"] for q in out} == {"AIDE arxiv", "AIDE github"}

    def test_entity_with_spaces(self):
        out = _seed_paper_queries(["OpenAI deep research"], ["Q1"], budget_cap=10)
        queries = [q["query"] for q in out]
        assert "OpenAI deep research arxiv" in queries
        assert "OpenAI deep research github" in queries

    def test_entity_with_dash(self):
        out = _seed_paper_queries(["MLR-Copilot"], ["Q1"], budget_cap=10)
        queries = [q["query"] for q in out]
        assert "MLR-Copilot arxiv" in queries

    def test_query_shape_valid(self):
        """Every output dict has the 5 required phase1a keys."""
        out = _seed_paper_queries(["A"], ["Q1"], budget_cap=10)
        for q in out:
            assert {"subquestion", "role", "query", "lang", "engines"}.issubset(q)
            assert isinstance(q["engines"], list)
            assert q["engines"]  # non-empty


# ---------------------------------------------------------------------------
# Arxiv-before-github ordering (helps hit arxiv first when budget is tight)
# ---------------------------------------------------------------------------

def test_arxiv_queries_come_first():
    """Tight budget should still prefer arxiv over github — easier to cite."""
    out = _seed_paper_queries(["A", "B", "C"], ["Q1"], budget_cap=3)
    kinds = [q["query"].split()[-1] for q in out]
    # First 3 should all be arxiv (since we have 3 entities and budget=3)
    assert kinds == ["arxiv", "arxiv", "arxiv"]
