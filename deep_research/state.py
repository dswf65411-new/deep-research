"""Pydantic models and TypedDict state schemas for the research graph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic value objects (serialisable, validated)
# ---------------------------------------------------------------------------


class TextSpan(BaseModel):
    """A verified character-range pointer into some source string.

    語意對齊 Anthropic Claude Citations API（start_char_index/end_char_index，
    0-based, end-exclusive, Python slice 語意）。

    用途：引用某段文字時，不複製文字，只記 (source_ref, start, end, text)。
      - text：引用建立當下從 source 切出的真值（保留防斷鏈）
      - resolve(raw)：需要渲染時用 raw[start:end] 取出，避免抄字幻覺
      - source_ref：pointer 指到哪個 raw — 通常是 source_id，但 phase3 的
        report_span 可能指向 report 草稿字串，所以 source_ref 接受自由字串

    Invariants（phase1a 產生時已驗證）：
      - 0 <= start < end <= len(original_raw)
      - original_raw[start:end] == text
    """
    source_ref: str = Field(description="raw 的識別符（source_id / report-draft 等）")
    start: int = Field(description="start char index, 0-based, Python slice")
    end: int = Field(description="end char index, exclusive")
    text: str = Field(description="驗證當下從 raw 切出的文字，作為斷鏈備援")

    def resolve(self, raw: str) -> str:
        """從 raw 取回這段文字。優先走 slice；若 slice 與 text 不符（raw 被動過）
        則退回記錄的 text（寧可用舊真值也不要抄錯）。"""
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
    full_research_topic: str  # 統整後的研究任務書（topic + refs + clarifications → LLM synthesis）
    plan: str
    depth: Literal["quick", "standard", "deep"]
    search_budget: int
    search_count: Annotated[int, _replace]
    ask_mode: bool  # True = interactive (--ask), False = autonomous (--noask)
    clarifications: Annotated[list[dict], operator.add]  # Q&A pairs from clarify
    refs: list[dict]  # 參考文件（text/image/pdf），格式見 context.py

    # Phase 1a
    sources: Annotated[list[Source], _upsert_sources]
    workspace_path: str
    fetched_urls: Annotated[list[str], operator.add]  # 跨輪已抓 URL，防止重複 fetch

    # Phase 1b
    claims: Annotated[list[Claim], _upsert_claims]
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
