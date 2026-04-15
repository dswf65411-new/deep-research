"""Pydantic models and TypedDict state schemas for the research graph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic value objects (serialisable, validated)
# ---------------------------------------------------------------------------


class TextSpan(BaseModel):
    """A verified character-range pointer into some source string.

    Semantics align with the Anthropic Claude Citations API (start_char_index /
    end_char_index, 0-based, end-exclusive, Python slice semantics).

    Purpose: when quoting text, do not copy the text; only record
    (source_ref, start, end, text).
      - text: the verified value sliced from the source at quote-creation time
        (kept as a safeguard against broken citation chains)
      - resolve(raw): when rendering, extract via raw[start:end] to avoid
        transcription hallucination
      - source_ref: which raw string the pointer refers to — typically a
        source_id, but phase3's report_span may point to the report draft
        string, so source_ref accepts any free-form string

    Invariants (verified at phase1a creation time):
      - 0 <= start < end <= len(original_raw)
      - original_raw[start:end] == text
    """
    source_ref: str = Field(description="identifier of raw (source_id / report-draft / etc.)")
    start: int = Field(description="start char index, 0-based, Python slice")
    end: int = Field(description="end char index, exclusive")
    text: str = Field(description="text sliced from raw at verification time, used as fallback when citation chain breaks")

    def resolve(self, raw: str) -> str:
        """Retrieve this text from raw. Prefer slicing; if the slice no longer
        matches text (raw has been modified) fall back to the recorded text
        (prefer the old verified value over a wrong transcription)."""
        if 0 <= self.start < self.end <= len(raw):
            sliced = raw[self.start : self.end]
            if sliced == self.text:
                return sliced
        return self.text


class Source(BaseModel):
    """A single research source (one URL / document)."""
    source_id: str = Field(description="e.g. S001")
    url: str
    title: str
    fetched_title: str = ""
    tier: Literal["T1", "T2", "T3", "T4", "T5", "T6"] = "T4"
    url_status: Literal["LIVE", "STALE", "UNREACHABLE", "UNKNOWN", "THIN_CONTENT"] = "UNKNOWN"
    fetch_date: str = ""
    engines: list[str] = Field(default_factory=list)
    role: Literal["advocate", "critic", "perspective"] = "advocate"
    subquestion: str = ""


class Claim(BaseModel):
    """A single verifiable claim extracted from sources."""
    claim_id: str = Field(description="e.g. Q1-C1")
    subquestion: str = ""
    claim_text: str
    claim_type: Literal["numeric", "comparative", "causal", "forecast", "qualitative"] = "qualitative"
    source_ids: list[str] = Field(default_factory=list)
    quote_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim quote / number-sentence snippets resolved from quote_ids at "
            "extraction time. Populated by phase1a `_collect_claims`; phase1b's "
            "MARCH-style blind checker reads this directly so the verifier sees the "
            "same text the claim was grounded on, instead of chasing opaque IDs."
        ),
    )
    bedrock_score: float = 0.0
    citation_verdict: str = ""
    number_tag: Literal["ORIGINAL", "NORMALIZED", "DERIVED"] | None = None
    status: Literal["pending", "approved", "rejected", "needs_revision"] = "pending"


class SubagentResult(BaseModel):
    """Result from an attack sub-agent check."""
    claim_id: str
    verdict: Literal["SUPPORTED", "PARTIAL", "NOT_SUPPORTED"]
    quote_id: str = "NONE"
    issue: str = ""


class StatementCheck(BaseModel):
    """Result from the final statement-level audit."""
    statement_id: str
    issue: Literal[
        "NONE", "BROKEN_CHAIN", "NUMBER_MISMATCH",
        "TONE_MISMATCH", "COMPOSITE_HALLUCINATION",
        "OVER_INFERENCE", "NO_SOURCE",
    ] = "NONE"
    detail: str = ""
    fix: str = ""


# ---------------------------------------------------------------------------
# Reducer helpers – LangGraph uses Annotated[list, operator.add] to append
# ---------------------------------------------------------------------------

import operator

def _replace(a, b):
    """Default reducer: latest value wins."""
    return b


def _upsert_by_id(id_attr: str):
    """Factory: return a reducer that upserts a list of objects by their id_attr field."""
    def _upsert(existing: list, new: list) -> list:
        if not existing:
            return list(new)
        merged: dict[str, object] = {}
        for item in existing:
            item_id = getattr(item, id_attr, None) if hasattr(item, id_attr) else (
                item.get(id_attr) if isinstance(item, dict) else None
            )
            if item_id:
                merged[item_id] = item
        for item in new:
            item_id = getattr(item, id_attr, None) if hasattr(item, id_attr) else (
                item.get(id_attr) if isinstance(item, dict) else None
            )
            if item_id:
                merged[item_id] = item
        return list(merged.values())
    return _upsert


_upsert_claims = _upsert_by_id("claim_id")
_upsert_sources = _upsert_by_id("source_id")


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # Phase 0
    topic: str
    full_research_topic: str  # integrated research brief (topic + refs + clarifications → LLM synthesis)
    plan: str
    depth: Literal["quick", "standard", "deep"]
    search_budget: int
    search_count: Annotated[int, _replace]
    ask_mode: bool  # True = interactive (--ask), False = autonomous (--noask)
    clarifications: Annotated[list[dict], operator.add]  # Q&A pairs from clarify
    refs: list[dict]  # reference files (text/image/pdf); format defined in context.py

    # Phase 1a
    sources: Annotated[list[Source], _upsert_sources]
    workspace_path: str
    fetched_urls: Annotated[list[str], operator.add]  # URLs already fetched across rounds, to prevent re-fetching

    # Phase 1b
    claims: Annotated[list[Claim], _upsert_claims]
    iteration_count: Annotated[int, _replace]
    coverage_status: Annotated[dict, _replace]

    # Phase 2
    # _replace (not operator.add) so the Critic-revise loop can re-run phase2
    # without the second draft being *appended* to the first. Each revision
    # replaces the whole section list; phase3 reads from disk anyway.
    report_sections: Annotated[list[str], _replace]
    # gpt-researcher-style Critic-revise loop (see multi_agents/agents/editor.py:129).
    # review_verdict holds the latest critic output {accept, issues, per_sq_issues};
    # revision_count is incremented each time phase2 is re-entered and capped at 2
    # so a persistently-failing writer cannot loop forever.
    review_verdict: Annotated[dict, _replace]
    revision_count: Annotated[int, _replace]
    # Tongyi ParallelMuse "Heavy Mode": when the revise loop hits max revisions
    # with an unresolved critic verdict, fan out the rejected sections into n=3
    # rollouts at staggered temperatures and let a selector LLM pick the best
    # candidate per section (no merging, per Tongyi INTEGRATE_PROMPT:58-66).
    # heavy_mode_triggered is a one-shot latch so the route does not re-enter.
    heavy_mode_triggered: Annotated[bool, _replace]

    # Phase 3
    final_report: str

    # Harness metadata
    execution_log: Annotated[list[str], operator.add]
    blockers: Annotated[list[str], operator.add]

    # Internal routing
    phase1b_result: str  # "pass" | "fail" | "max_retries"
    quality_scores: Annotated[dict, _replace]        # per-SQ quality scores from phase1b subgraph
    needs_refetch: Annotated[list[str], _replace]    # SQ IDs for focused fallback re-search
    fallback_count: Annotated[int, _replace]         # number of fallback loops triggered (max 2)


# ---------------------------------------------------------------------------
# Phase 1b private state (subgraph)
# ---------------------------------------------------------------------------

class VerifyState(TypedDict, total=False):
    claims_to_verify: list[Claim]
    grounding_results: list[dict]
    quality_scores: dict
    attack_results: list[SubagentResult]
    iteration: int
    workspace_path: str
    failed_dimensions: list[str]
