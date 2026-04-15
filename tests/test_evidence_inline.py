"""Verify evidence_quotes flow (Tongyi-style three-part row).

Gap being closed: previously claim-ledger.md stored only `quote_ids` references
(e.g. `S001-Q3`). The phase1b MARCH blind checker had to chase those IDs
through a separate file to see the actual text the claim was grounded on — or
worse, the LLM could silently ignore unresolved IDs. Tongyi DeepResearch keeps
the verbatim evidence inline on the same row as the claim, so checkers see
`(claim_text, evidence_quotes)` as one self-contained unit.

Covers:
  1. `Claim` model accepts `evidence_quotes`, default empty.
  2. `_collect_claims` resolves quote_ids / number_ids → verbatim snippets.
  3. `_write_claim_ledger` renders the evidence column inline.
"""

from pathlib import Path

from deep_research.nodes.phase1a import _collect_claims
from deep_research.nodes.phase1b import _write_claim_ledger
from deep_research.state import Claim


def test_claim_has_evidence_quotes_default_empty():
    c = Claim(claim_id="Q1-C1", claim_text="x", subquestion="Q1")
    assert c.evidence_quotes == []


def test_claim_accepts_evidence_quotes():
    c = Claim(
        claim_id="Q1-C1",
        claim_text="x",
        subquestion="Q1",
        evidence_quotes=["raw quote one", "raw quote two"],
    )
    assert c.evidence_quotes == ["raw quote one", "raw quote two"]


def test_collect_claims_resolves_quote_ids_to_text():
    extractions = [
        {
            "source_id": "S001",
            "quotes": [
                {"quote_id": "S001-Q1", "text": "verbatim quote alpha"},
                {"quote_id": "S001-Q2", "text": "verbatim quote beta"},
            ],
            "numbers": [
                {"number_id": "S001-N1", "sentence": "67% of agents failed the eval."},
            ],
            "claims": [
                {
                    "subquestion": "Q1",
                    "claim_text": "Agents often fail the eval.",
                    "claim_type": "qualitative",
                    "quote_ids": ["S001-Q1", "S001-N1"],
                    "source_id": "S001",
                }
            ],
        }
    ]

    claims = _collect_claims(extractions)
    assert len(claims) == 1
    c = claims[0]
    assert c.claim_id == "Q1-C1"
    # evidence_quotes ordered per quote_ids sequence
    assert c.evidence_quotes == [
        "verbatim quote alpha",
        "67% of agents failed the eval.",
    ]
    # Unreferenced quote (S001-Q2) must NOT leak in
    assert "verbatim quote beta" not in c.evidence_quotes


def test_collect_claims_skips_unknown_quote_ids():
    """A claim referencing a quote_id not in this source's extraction should
    produce an evidence list only for the known IDs (no KeyError, no placeholder)."""
    extractions = [{
        "source_id": "S002",
        "quotes": [{"quote_id": "S002-Q1", "text": "valid quote"}],
        "numbers": [],
        "claims": [{
            "subquestion": "Q2",
            "claim_text": "Mixed evidence claim.",
            "claim_type": "qualitative",
            "quote_ids": ["S002-Q1", "S002-Q99"],
            "source_id": "S002",
        }],
    }]
    claims = _collect_claims(extractions)
    assert len(claims) == 1
    # S002-Q99 silently dropped — don't invent placeholders
    assert claims[0].evidence_quotes == ["valid quote"]


def test_collect_claims_empty_evidence_when_no_quotes():
    """Legacy path: claims with no quotes/numbers registered should just have
    an empty evidence list (don't crash, don't fall back)."""
    extractions = [{
        "source_id": "S003",
        "quotes": [],
        "numbers": [],
        "claims": [{
            "subquestion": "Q3",
            "claim_text": "Unsupported claim.",
            "claim_type": "qualitative",
            "quote_ids": [],
            "source_id": "S003",
        }],
    }]
    claims = _collect_claims(extractions)
    assert len(claims) == 1
    assert claims[0].evidence_quotes == []


def test_ledger_includes_evidence_column(tmp_path: Path):
    claims = [
        Claim(
            claim_id="Q1-C1",
            subquestion="Q1",
            claim_text="Supervisor pattern reduces coordination overhead.",
            claim_type="qualitative",
            source_ids=["S001"],
            quote_ids=["S001-Q1"],
            evidence_quotes=["Supervisor coordinates sub-agents via shared state."],
            bedrock_score=0.87,
            status="approved",
        ),
    ]
    _write_claim_ledger(str(tmp_path), claims)

    text = (tmp_path / "claim-ledger.md").read_text()
    # Header column exists
    assert "evidence" in text
    # Inline evidence snippet visible on the same row
    assert "Supervisor coordinates sub-agents via shared state." in text
    # Still carries the quote_id for lookup compatibility
    assert "S001-Q1" in text


def test_ledger_evidence_cell_escapes_pipes(tmp_path: Path):
    """Evidence containing a literal `|` must not break markdown table layout."""
    claims = [Claim(
        claim_id="Q1-C1",
        subquestion="Q1",
        claim_text="edge case",
        evidence_quotes=["a | b | c — pipe-heavy quote", "second quote"],
    )]
    _write_claim_ledger(str(tmp_path), claims)
    text = (tmp_path / "claim-ledger.md").read_text()
    assert "a \\| b \\| c" in text
    # Paragraph separator between evidence entries
    assert " ¶ " in text
    assert "second quote" in text


def test_ledger_empty_evidence_renders_none(tmp_path: Path):
    """A claim with no resolved evidence must still render legally (no empty
    cell that collapses the table)."""
    claims = [Claim(claim_id="Q1-C1", subquestion="Q1", claim_text="x")]
    _write_claim_ledger(str(tmp_path), claims)
    text = (tmp_path / "claim-ledger.md").read_text()
    assert "(none)" in text
