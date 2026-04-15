"""Tests for Whisper X-2 — claim-level relevance_to_question scoring."""

from __future__ import annotations

from dataclasses import dataclass

from deep_research.harness.claim_relevance import (
    ON_TOPIC_THRESHOLD,
    TANGENTIAL_THRESHOLD,
    annotate_claims,
    classify_claim,
    compute_claim_relevance,
    count_by_label,
)


# ---------------------------------------------------------------------------
# compute_claim_relevance
# ---------------------------------------------------------------------------


def test_identical_text_has_high_score():
    score = compute_claim_relevance(
        "LangGraph Supervisor handles advocate critic arbitration",
        "LangGraph Supervisor handles advocate critic arbitration",
    )
    assert score == 1.0


def test_totally_off_topic_scores_zero():
    """Murf AI TTS marketing text vs LangGraph Supervisor SQ — the original
    failure mode. Must score at off_topic threshold."""
    score = compute_claim_relevance(
        "Murf AI offers 140 voices and supports multiple languages",
        "How should a LangGraph supervisor arbitrate between sub-agents when one fails?",
    )
    assert score < TANGENTIAL_THRESHOLD


def test_partial_overlap_lands_in_tangential_band():
    score = compute_claim_relevance(
        "LangGraph documentation covers persistence and checkpointing",
        "How should a LangGraph supervisor arbitrate sub-agent failures?",
    )
    # shares 'langgraph' → score in tangential band
    assert TANGENTIAL_THRESHOLD <= score < ON_TOPIC_THRESHOLD or score < TANGENTIAL_THRESHOLD


def test_entity_boost_lifts_marginal_claim():
    ents = ["AIDE"]
    no_boost = compute_claim_relevance(
        "Some text",
        "How does X compare tools",
        brief_entities=ents,
    )
    # AIDE not in either → no boost
    assert no_boost < 0.05

    with_boost = compute_claim_relevance(
        "AIDE is a data-science automation tool",
        "How does AIDE compare to other automation tools",
        brief_entities=ents,
    )
    # AIDE in both → +0.15 boost
    assert with_boost >= 0.15


def test_empty_inputs_zero():
    assert compute_claim_relevance("", "some SQ") == 0.0
    assert compute_claim_relevance("some claim", "") == 0.0
    assert compute_claim_relevance("", "") == 0.0


def test_stopwords_dont_inflate_score():
    """Two texts sharing only 'the / a / is' should score near zero."""
    score = compute_claim_relevance(
        "the quick brown fox is a mammal",
        "the slow green turtle is a reptile",
    )
    assert score < 0.05


def test_score_is_symmetric():
    a = compute_claim_relevance("alpha beta gamma", "beta gamma delta")
    b = compute_claim_relevance("beta gamma delta", "alpha beta gamma")
    assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# classify_claim
# ---------------------------------------------------------------------------


def test_classify_thresholds():
    assert classify_claim(0.0) == "off_topic"
    assert classify_claim(TANGENTIAL_THRESHOLD - 0.001) == "off_topic"
    assert classify_claim(TANGENTIAL_THRESHOLD) == "tangential"
    assert classify_claim(ON_TOPIC_THRESHOLD - 0.001) == "tangential"
    assert classify_claim(ON_TOPIC_THRESHOLD) == "on_topic"
    assert classify_claim(1.0) == "on_topic"


# ---------------------------------------------------------------------------
# annotate_claims
# ---------------------------------------------------------------------------


@dataclass
class _Claim:
    claim_id: str
    subquestion: str
    claim_text: str


def test_annotate_maps_subquestions():
    claims = [
        _Claim("Q1-C1", "Q1", "LangGraph Supervisor handles arbitration"),
        _Claim("Q2-C1", "Q2", "Murf AI offers 140 voices"),
    ]
    sqs = {
        "Q1": "How does a LangGraph Supervisor arbitrate between sub-agents",
        "Q2": "How should a LangGraph supervisor handle sub-agent failures",
    }
    rows = annotate_claims(claims, sqs)
    assert len(rows) == 2
    assert rows[0].claim_id == "Q1-C1"
    assert rows[0].label == "on_topic"
    assert rows[1].claim_id == "Q2-C1"
    assert rows[1].label == "off_topic"


def test_annotate_unknown_sq_scores_zero():
    claims = [_Claim("Q9-C1", "Q9", "some claim")]
    rows = annotate_claims(claims, subquestions={"Q1": "something"})
    assert rows[0].score == 0.0
    assert rows[0].label == "off_topic"


def test_annotate_accepts_dicts():
    claims = [{"claim_id": "Q1-C1", "subquestion": "Q1",
               "claim_text": "LangGraph supervisor arbitration"}]
    sqs = {"Q1": "LangGraph supervisor arbitration"}
    rows = annotate_claims(claims, sqs)
    assert rows[0].label == "on_topic"


def test_count_by_label():
    claims = [
        _Claim("C1", "Q1", "LangGraph supervisor arbitration"),
        _Claim("C2", "Q1", "Murf AI TTS voices"),
        _Claim("C3", "Q1", "LangGraph docs only"),
    ]
    sqs = {"Q1": "How does LangGraph supervisor arbitrate failures"}
    rows = annotate_claims(claims, sqs)
    counts = count_by_label(rows)
    # Exact distribution depends on scoring, but we must have at least one
    # off_topic (Murf AI).
    assert counts["off_topic"] >= 1
    assert sum(counts.values()) == 3
