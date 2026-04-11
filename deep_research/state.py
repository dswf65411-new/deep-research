"""Pydantic models and TypedDict state schemas for the research graph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic value objects (serialisable, validated)
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A single research source (one URL / document)."""
    source_id: str = Field(description="e.g. S001")
    url: str
    title: str
    fetched_title: str = ""
    tier: Literal["T1", "T2", "T3", "T4", "T5", "T6"] = "T4"
    url_status: Literal["LIVE", "STALE", "UNREACHABLE", "UNKNOWN"] = "UNKNOWN"
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


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # Phase 0
    topic: str
    full_research_topic: str  # 統整後的研究任務書（topic + refs + clarifications → LLM synthesis）
    plan: str
    depth: Literal["quick", "standard", "deep"]
    search_budget: int
    search_count: Annotated[int, _replace]
    ask_mode: bool  # True = interactive (--ask), False = autonomous (--noask)
    clarifications: Annotated[list[dict], operator.add]  # Q&A pairs from clarify
    refs: list[dict]  # 參考文件（text/image/pdf），格式見 context.py

    # Phase 1a
    sources: Annotated[list[Source], operator.add]
    workspace_path: str

    # Phase 1b
    claims: Annotated[list[Claim], operator.add]
    iteration_count: Annotated[int, _replace]
    coverage_status: Annotated[dict, _replace]

    # Phase 2
    report_sections: Annotated[list[str], operator.add]

    # Phase 3
    final_report: str

    # Harness metadata
    execution_log: Annotated[list[str], operator.add]
    blockers: Annotated[list[str], operator.add]

    # Internal routing
    phase1b_result: str  # "pass" | "fail" | "max_retries"


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
