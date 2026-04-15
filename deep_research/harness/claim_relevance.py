"""Whisper X-2 — claim-level ``relevance_to_question`` scoring.

Bedrock grounding of ``1.00`` only says the claim *can be aligned to a source*.
It does not answer "does this claim actually address the sub-question?". In the
failed workspace, six marketing-blog claims about Murf AI text-to-speech
grounded at 1.00 despite the research being about LangGraph Supervisor
patterns.

This module computes a cheap, offline relevance score per claim so phase2/3
can highlight (or filter) off-topic claims without re-running grounding.

Scoring uses token overlap (Jaccard) between claim text, sub-question text,
and brief entities. Brief entities count double because the failure mode was
"grounded but named entity never appears". Nothing here talks to an LLM —
this stays fast and deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal

RelevanceLabel = Literal["on_topic", "tangential", "off_topic"]

# Tuned against the failure workspace: Murf AI claims had overlap < 0.05
# against SQs; AIDE / MLE-Agent entity-carrying claims had ≥ 0.25.
TANGENTIAL_THRESHOLD = 0.10
ON_TOPIC_THRESHOLD = 0.25

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+", re.UNICODE)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with",
    "to", "from", "by", "at", "as", "is", "are", "be", "it", "this",
    "that", "these", "those", "we", "our", "their", "into", "how",
    "what", "why", "which", "whose", "when", "where", "whom",
    "use", "using", "used", "vs", "via", "per",
})


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if tok and len(tok) > 1 and tok.lower() not in _STOPWORDS
    }


def compute_claim_relevance(
    claim_text: str,
    subquestion_text: str,
    brief_entities: Iterable[str] | None = None,
) -> float:
    """Return a 0..1 relevance score for ``claim_text`` vs the SQ.

    * Base: Jaccard over non-stopword tokens between claim and SQ.
    * Entity boost: any ``brief_entities`` token appearing in both claim and SQ
      adds ``+0.15`` (capped at 1.0). This catches the "named entity missing"
      failure mode where a claim grounds high but never mentions the entity
      the user actually cares about.
    """
    c_tokens = _tokenize(claim_text)
    q_tokens = _tokenize(subquestion_text)
    if not c_tokens or not q_tokens:
        return 0.0

    intersection = c_tokens & q_tokens
    union = c_tokens | q_tokens
    base = len(intersection) / len(union) if union else 0.0

    boost = 0.0
    for ent in brief_entities or []:
        ent_tokens = _tokenize(ent)
        if not ent_tokens:
            continue
        # All entity tokens need to appear in *both* claim and SQ for the
        # boost to fire — prevents "SQ mentions LangGraph but claim is about
        # a totally different LangGraph blog post" false boosts.
        if ent_tokens.issubset(c_tokens) and ent_tokens.issubset(q_tokens):
            boost += 0.15

    return min(1.0, base + boost)


def classify_claim(score: float) -> RelevanceLabel:
    if score >= ON_TOPIC_THRESHOLD:
        return "on_topic"
    if score >= TANGENTIAL_THRESHOLD:
        return "tangential"
    return "off_topic"


@dataclass
class RelevanceRow:
    claim_id: str
    score: float
    label: RelevanceLabel


def annotate_claims(
    claims: Iterable,
    subquestions: dict[str, str],
    brief_entities: Iterable[str] | None = None,
) -> list[RelevanceRow]:
    """Batch-score a list of claims.

    ``subquestions`` maps SQ id (``Q1``, ``Q2``, ...) → SQ text. Claims whose
    ``subquestion`` id isn't in the dict fall back to score=0.0 so they flag
    as off_topic (better than silently vanishing).
    """
    ents = list(brief_entities or [])
    rows: list[RelevanceRow] = []
    for c in claims or []:
        cid = _get(c, "claim_id") or ""
        sq_id = (_get(c, "subquestion") or "").strip()
        ctext = _get(c, "claim_text") or ""
        sq_text = subquestions.get(sq_id, "")
        score = compute_claim_relevance(ctext, sq_text, brief_entities=ents)
        rows.append(RelevanceRow(
            claim_id=cid, score=round(score, 3), label=classify_claim(score),
        ))
    return rows


def count_by_label(rows: Iterable[RelevanceRow]) -> dict[RelevanceLabel, int]:
    counts: dict[RelevanceLabel, int] = {"on_topic": 0, "tangential": 0, "off_topic": 0}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    return counts


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
