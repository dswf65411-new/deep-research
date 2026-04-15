"""Phase 1a: Planner-Executor-Extractor.

Replaces the old ReAct agent design. The original create_react_agent accumulated
every round's tool_result (full page text) into the next LLM request, easily
blowing past Anthropic's 30K ITPM on a single request.

The new architecture splits into 4 stages so each LLM call has a fixed, small context:
  1. Planner (LLM x 1)    reads plan/coverage/gap, produces a query list
  2. Executor (code)      parallel search, urlhealth validation, concurrent WebFetch/Serper, writes raw
  3. Extractor (LLM x N)  each page uses a fresh context; extracts QUOTE / NUMBER / pending claim
  4. Registry (code)      updates source-registry, execution-log, gap-log

The Extractor is throttled by the rate_limiter in deep_research.llm to avoid 429s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from deep_research.config import get_prompt
from deep_research.nodes.phase0 import DEPTH_CONFIG
from deep_research.harness.validators import (
    resolve_quote_index,
    validate_quote_ids_in_ledger,
    validate_quotes_exist,
    validate_quotes_indexed,
    verify_indexed_items,
)
from deep_research.harness.claim_dedup import is_near_duplicate, normalize_for_dedup
from deep_research.harness.source_tier import classify_tier
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
from deep_research.prompts_shared import FOCUSED_EXEC_PROMPT
from deep_research.state import Claim, ResearchState, Source
from deep_research.tools.arxiv_retriever import arxiv_search
from deep_research.tools.github_search import github_repo_search
from deep_research.tools.search import (
    BRAVE_API_KEY,
    SERPER_API_KEY,
    brave_search,
    serper_scholar,
    serper_search,
    serper_scrape,
    web_fetch,
)
from deep_research.harness.url_validator import extract_arxiv_ids, extract_urls
from deep_research.tools.workspace import (
    append_workspace_file,
    init_source_registry,
    read_workspace_file,
    write_workspace_file,
)


_URLHEALTH_PY = "/Users/yao.chu/.pyenv/versions/3.13.12/bin/python3"
_URLHEALTH_CLI = "/Users/yao.chu/.claude/mcp-servers/urlhealth.py"

# Deep-read quota: per subquestion, per round
# seed: brief-named entities' arxiv/github queries — quota relaxed so the 2-cap
#       doesn't block SOTA papers from entering source-registry.
_QUOTA_PER_ROLE = {"advocate": 2, "critic": 2, "perspective": 1, "seed": 6}

# Full-text truncation cap (avoids exceeding ~10K tokens per page; the Extractor
# sees ~6-8K tokens per page).
# Reduced from 45000: a 45K-char page ~ 15K tokens — feeding the full thing to the
# LLM makes middle quotes/numbers hit Lost-in-the-Middle and get ignored, breaking
# hard rule 4 (citation chain integrity). Above _CHUNK_SIZE we switch to chunked mode.
_RAW_CHAR_LIMIT = 25000

# Taiwan authoritative domain whitelist: used by source_tier.py — a hit auto-promotes to T3
_TAIWAN_DOMAIN_WHITELIST: frozenset[str] = frozenset({
    "ithome.com.tw",
    "ithelp.ithome.com.tw",
    "techbang.com",
    "kocpc.com.tw",
    "mobile01.com",
    "eprice.com.tw",
    "inside.com.tw",
    "bnext.com.tw",
})

# Chunked extraction: sliding-window slices; each chunk is an independent concurrent
# LLM call, then merged and deduplicated.
_CHUNK_SIZE = 8000
_CHUNK_OVERLAP = 1000


# ---------------------------------------------------------------------------
# Scholar-intent auto-upgrade (Whisper P1-6)
# ---------------------------------------------------------------------------
# In the failed 2026-04-14 workspace, brief-named SOTA papers (AIDE, MLE-bench,
# ResearchAgent, AutoML-Agent) were buried under marketing blogs because
# generic web engines (Brave / Serper) rank by backlinks, not by venue. Even
# with dedicated ``arxiv`` / ``serper_scholar`` engines, the web ones still
# burn budget on noise.
#
# This heuristic catches scholar-intent web queries (words like ``arxiv``,
# ``paper``, ``benchmark``, ``SOTA`` …) and appends a ``site:`` filter so
# Brave/Serper only return hits from arxiv / ACL Anthology / OpenReview.
#
# Intentionally narrow: only applies to ``brave`` and ``serper_en``. The
# zh/cn variants would lose regional coverage with site: filters, and the
# dedicated ``serper_scholar`` / ``arxiv`` / ``github`` engines already
# query authoritative corpora directly.
_SCHOLAR_INTENT_RE: re.Pattern = re.compile(
    r"\b("
    r"arxiv|"
    r"papers?|preprint|"
    r"sota|state[- ]of[- ]the[- ]art|"
    r"benchmark|leaderboard|"
    r"peer[- ]review(?:ed)?|proceedings|"
    r"thesis|dissertation|"
    r"acl|emnlp|neurips|nips|icml|iclr|aaai|cvpr|iccv"
    r")\b",
    re.IGNORECASE,
)

_SCHOLAR_UPGRADE_ENGINES: frozenset[str] = frozenset({"brave", "serper_en"})

_SCHOLAR_SITE_CLAUSE = (
    " (site:arxiv.org OR site:aclanthology.org OR site:openreview.net)"
)


def _maybe_upgrade_to_scholar_query(query: str, engine: str) -> str:
    """Wrap scholar-intent queries with a ``site:`` filter on generic web engines.

    Returns the query unchanged if:
      - engine is not a generic web engine (dedicated scholar/arxiv/github
        engines handle authoritative sources natively);
      - the query already contains ``site:`` (user-specified filter wins);
      - no scholar-intent keyword matched.

    Keeping this pure + deterministic so the test suite can exercise it
    without touching HTTP.
    """
    if engine not in _SCHOLAR_UPGRADE_ENGINES:
        return query
    if "site:" in query.lower():
        return query
    if not _SCHOLAR_INTENT_RE.search(query):
        return query
    return query + _SCHOLAR_SITE_CLAUSE


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

async def phase1a_search(state: ResearchState) -> dict:
    """Phase 1a entry: plan → search → deep-read → extract."""
    workspace = state["workspace_path"]
    plan = state.get("plan", "")
    depth = state.get("depth", "deep")
    budget = state.get("search_budget", 150)
    used = state.get("search_count", 0)
    iteration = state.get("iteration_count", 0)
    topic = state.get("topic", "")

    coverage = read_workspace_file(workspace, "coverage.chk") or ""
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""
    exec_log = read_workspace_file(workspace, "execution-log.md") or ""
    sq_progress = read_workspace_file(workspace, "sq-progress.md") or ""

    # gap_log accumulates round over round (by round 5 it holds every UNREACHABLE /
    # MISSING / CONFLICT from rounds 1-4). For the Planner, only recent gaps carry
    # useful signal; older content is just a distractor. Keep the last ~2000 chars
    # (~650 tokens) to avoid context amplifier effects.
    if len(gap_log) > 2000:
        gap_log = "...[earlier content omitted, keeping only recent gaps]...\n\n" + gap_log[-2000:]
    # sq-progress.md is the Tongyi-style S_t state. Only the latest iteration's
    # section carries signal; older ones are stale, so slice to the last
    # "## Iteration N — per-SQ evidence snapshot" section.
    if sq_progress:
        last_idx = sq_progress.rfind("## Iteration")
        if last_idx > 0:
            sq_progress = sq_progress[last_idx:]
    # If plan exceeds 8000 chars (rare but happens), keep head and tail so the
    # Planner can still see both ends.
    if len(plan) > 8000:
        plan = plan[:4000] + "\n\n...[middle omitted]...\n\n" + plan[-4000:]

    # For cross-round incremental numbering: existing sources / claims
    existing_sources = state.get("sources", [])
    existing_claims = state.get("claims", [])

    # Ensure source-registry exists
    if not read_workspace_file(workspace, "source-registry.md"):
        init_source_registry(workspace)

    remaining = budget - used
    if remaining <= 0:
        return {
            "execution_log": [f"Phase 1a round {iteration + 1}: budget exhausted, skipping"],
        }

    # Focus mode: trigger_fallback_node specifies SQs needing refetch
    needs_refetch: list[str] = state.get("needs_refetch", [])
    focus_mode = bool(needs_refetch)
    if focus_mode:
        # Budget-guard refetch (large needs_refetch) vs quality-failure refetch (small needs_refetch)
        # - Quality failure (<= 5 SQ) → focused refetch, budget cap 25
        # - Budget guard (> 5 SQ)   → broad continued search, cap is half of remaining (at least 40)
        if len(needs_refetch) > 5:
            budget_cap = max(40, remaining // 2)
            remaining = min(remaining, budget_cap)
            logger.info(f"Phase 1a budget guard refetch: SQ={len(needs_refetch)}, budget={remaining}")
        else:
            remaining = min(remaining, 25)
            logger.info(f"Phase 1a focused refetch: SQ={needs_refetch}, budget={remaining}")

    # ── Stage 1: Planner ─────────────────────────────────────────────
    already_searched = _extract_searched_queries(exec_log)

    # Budget guard: tally queries used per SQ, find SQs below the minimum
    min_per_sq = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["deep"])["min_budget_per_sq"]
    sq_ids = _extract_sq_ids(plan)
    sq_counts = _count_queries_per_sq(exec_log)
    underfunded_sqs = [sq for sq in sq_ids if sq_counts.get(sq, 0) < min_per_sq]

    # Discovery queries: current year + known-tools seed
    current_year = str(datetime.now().year)
    known_tools = _extract_known_tools(plan)

    # Parse newly discovered entities from the prior round's gap-log
    # (only when iteration > 0 and not in focus mode)
    prior_emerging = _extract_emerging_from_gap_log(gap_log, iteration) if iteration > 0 and not focus_mode else []

    # ── Brief entity seeds (injected on round 1, when not in focus mode) ───
    # Read research-brief.md + phase0-plan.md to extract brief-named entities →
    # produce hard-rule arxiv/github queries per entity; guarantees SOTA tools
    # named in the brief are actually searched and land in the seed URL set.
    seed_queries: list[dict] = []
    brief_entities: list[str] = []
    if iteration == 0 and not focus_mode:
        brief_text = read_workspace_file(workspace, "research-brief.md") or ""
        combined_source = (brief_text + "\n\n" + plan) if brief_text else plan
        brief_entities = _extract_brief_entities(combined_source)
        # Merge with _extract_known_tools results so the LLM planner also sees brief entities
        combined_known = list(dict.fromkeys(list(brief_entities) + list(known_tools)))[:20]
        # 2 queries per entity (arxiv+github); ~13% of remaining budget, capped at 20
        seed_budget_cap = min(20, max(0, remaining // 4))
        seed_queries = _seed_paper_queries(brief_entities, sq_ids, seed_budget_cap)
        known_tools = combined_known
    remaining_for_llm = max(0, remaining - len(seed_queries))

    query_plan = await _plan_queries(
        plan=plan,
        coverage=coverage,
        gap_log=gap_log,
        sq_progress=sq_progress,
        iteration=iteration,
        remaining_budget=remaining_for_llm,
        already_searched=already_searched,
        depth=depth,
        underfunded_sqs=underfunded_sqs,
        sq_counts=sq_counts,
        min_per_sq=min_per_sq,
        current_year=current_year,
        known_tools=known_tools,
        focus_sqs=needs_refetch if focus_mode else None,
        emerging_entities=prior_emerging if prior_emerging else None,
    )

    llm_queries = query_plan.get("queries", [])

    # Cross-round duplicate rollback (MiroThinker-inspired):
    # planner's prompt asks it to dedupe, but LLM paraphrases still slip through.
    # Dropping pre-execution saves search budget and forces the next round's
    # planner to change angle when the gap-log shows the rollback.
    llm_queries, dup_dropped = _detect_duplicate_queries(llm_queries, already_searched)
    if dup_dropped:
        _log_duplicate_rollback(workspace, iteration, dup_dropped)
        # STUCK escalation: 5+ consecutive rounds of rollback for the same SQ
        # gets a louder warning so phase1b / phase2 can note it.
        stuck_sqs = sorted({
            q.get("subquestion", "Q?") for q in dup_dropped
            if _count_consecutive_stuck_rounds(gap_log, q.get("subquestion", "")) + 1 >= 5
        })
        if stuck_sqs:
            append_workspace_file(
                workspace,
                "gap-log.md",
                f"\n\n## [STUCK] subquestions with 5+ consecutive rollback rounds\n"
                + "\n".join(f"- {s}: planner looks trapped — consider BLOCKER" for s in stuck_sqs)
                + "\n",
            )

    # Merge seed + LLM queries, seeds first to ensure priority; dedupe by query string
    seen_q: set[str] = set()
    queries: list[dict] = []
    for q in seed_queries + llm_queries:
        key = (q.get("query") or "").strip().lower()
        if not key or key in seen_q:
            continue
        seen_q.add(key)
        queries.append(q)
    if not queries:
        return {
            "execution_log": [f"Phase 1a round {iteration + 1}: planner produced no queries, skipping"],
        }

    # ── Stage 2: Executor ────────────────────────────────────────────
    search_hits, searches_used = await _execute_searches(queries, remaining)
    url_health = await _verify_urls(search_hits)
    selected = _select_urls_by_quota(search_hits, url_health, queries)

    # ── Stage 2.5: Seed URLs from phase0 plan (round 1 only, non-focus) ─
    # URLs/arxiv IDs phase0 already validated stream straight into fetch —
    # zero search-budget cost and no chance of search engines ranking
    # marketing blogs above them. Closes the Phase0 → Phase1a context
    # break that caused the 2026-04-14 failure workspace.
    seed_urls_taken: list[str] = []
    if iteration == 0 and not focus_mode:
        seeds = _extract_seed_urls(plan)
        if seeds:
            seen_selected = {s["url"] for s in selected}
            for seed in seeds:
                if seed["url"] in seen_selected:
                    continue
                selected.append(seed)
                seen_selected.add(seed["url"])
                seed_urls_taken.append(seed["url"])
            if seed_urls_taken:
                logger.info(
                    "phase1a: seeded %d URLs/arxiv IDs from phase0 plan (budget-free)",
                    len(seed_urls_taken),
                )

    # Cross-round URL dedup: skip URLs already fetched in prior rounds to avoid
    # wasting budget and producing duplicate claims.
    prior_fetched: set[str] = set(state.get("fetched_urls", []))
    if prior_fetched:
        before = len(selected)
        selected = [s for s in selected if s["url"] not in prior_fetched]
        skipped = before - len(selected)
        if skipped:
            logger.info("phase1a: cross-round dedup skipped %d already-fetched URLs", skipped)

    source_id_start = _next_source_id_index(existing_sources)
    raw_sources = await _fetch_pages(selected, workspace, source_id_start)

    # ── Stage 3: Extractor (each page is an independent LLM call with fresh context) ──
    # Parse the SQ text index once per round — extractor uses it for the goal-aware relevance gate
    # (Tongyi DeepResearch style: page summary is conditioned on the goal, not just a label).
    sq_texts = _parse_sq_texts(coverage)
    extractions = await _extract_all_sources(raw_sources, workspace, sq_texts)

    # ── Stage 3.5: Source-pool curator (gpt-researcher style) ─────────
    # LLM scores every fetched source for relevance/credibility/quant_value.
    # Low-scoring sources emit a [CURATOR WARNING] in gap-log so phase1b and
    # phase2 can apply stricter grounding thresholds / reject their claims.
    try:
        curator_scores = await _curate_sources(raw_sources, extractions, plan, coverage)
        _write_source_curation(workspace, iteration, curator_scores)
    except Exception as e:
        logger.warning("source curator step failed (%s) — continuing", e)

    # ── Stage 4: Registry ────────────────────────────────────────────
    _update_source_registry(workspace, raw_sources)
    _append_execution_log(workspace, iteration, queries, searches_used)
    _log_unreachable(workspace, url_health, raw_sources)
    _log_domain_bias(workspace, iteration, existing_sources, raw_sources)
    _log_off_topic_ratio(workspace, iteration, raw_sources)
    _build_sq_evidence_snapshot(workspace, iteration, raw_sources, sq_texts)

    # think_tool-style planner reflection appended to sq-progress.md for the
    # next round's planner. Re-reads the snapshot that was just written.
    refreshed_snapshot = read_workspace_file(workspace, "sq-progress.md") or ""
    refreshed_gap_log = read_workspace_file(workspace, "gap-log.md") or gap_log
    await _write_planner_reflection(
        workspace,
        iteration,
        refreshed_snapshot,
        coverage,
        refreshed_gap_log,
    )

    # Budget guard: update this round's sq_counts; write any still-underfunded SQs to gap-log
    updated_sq_counts = dict(sq_counts)
    for q in queries:
        sq = q["subquestion"]
        updated_sq_counts[sq] = updated_sq_counts.get(sq, 0) + 1
    _log_budget_gaps(workspace, iteration + 1, sq_ids, updated_sq_counts, min_per_sq)

    # Iterative expansion: extract new tool names from this round's search results
    # into gap-log so the next round's Planner can generate category-B follow-up
    # queries (skipped in focus mode to avoid splitting attention).
    emerging_entities: list[str] = []
    if not focus_mode:
        emerging_entities = await _extract_emerging_entities(raw_sources, plan, iteration)
        if emerging_entities:
            entities_text = "\n".join(f"- {e}" for e in emerging_entities)
            append_workspace_file(
                workspace,
                "gap-log.md",
                f"\n\n## newly discovered entities (round {iteration + 1})\n{entities_text}\n",
            )

    sources = _build_sources(raw_sources)
    claims = _collect_claims(extractions, existing_claims)

    # URLs newly fetched this round (LIVE/THIN_CONTENT both count; UNREACHABLE
    # is excluded so the next round can retry rather than skip).
    new_fetched_urls = [
        s["url"] for s in raw_sources
        if s.get("status") not in ("UNREACHABLE",) and s.get("url")
    ]

    return {
        "search_count": used + searches_used,
        "sources": sources,
        "claims": claims,
        "fetched_urls": new_fetched_urls,
        "execution_log": [
            f"Phase 1a round {iteration + 1}: "
            f"searched {searches_used}, deep-read {len(raw_sources)}, "
            f"extracted {len(claims)} claims (cumulative {used + searches_used}/{budget})"
            + (f", brief entity seed {len(seed_queries)} (entities={len(brief_entities)})" if seed_queries else "")
            + (f", newly discovered entities: {len(emerging_entities)}" if emerging_entities else "")
        ],
    }


# ---------------------------------------------------------------------------
# Stage 1: Planner — produces the query list
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = FOCUSED_EXEC_PROMPT + """You are the research-search Planner. Given the research plan and current coverage, produce the list of search queries to run next round.

## Responsibilities
- Only plan queries; do not execute searches.
- Incremental: in round 1 each subquestion only produces the minimum set (advocate 1 family + critic 1 family).
- In later rounds, add queries to fill coverage gaps.

## Query rules
1. Each query should be 5-10 words.
2. Advocate and critic queries must be clearly different (not the same query reworded).
3. A query family usually has two language versions: en + zh-TW. zh-TW queries **must only** use `engines: ["serper_tw"]` and must not mix with brave or serper_en (avoids Traditional-Chinese results being diluted by English indexes).
4. Semantically dedupe against the "already searched queries" list; do not repeat.
5. Academic topics may append `site:arxiv.org` or `site:semanticscholar.org`.

## Search-engine configuration (specify engines list per query)
- `brave`: English independent index
- `serper_en`: Google English
- `serper_tw`: Google Traditional Chinese (zh-TW query only; do not mix with brave/serper_en)
- `serper_cn`: Google China
- `serper_scholar`: dedicated Google Scholar API — returns pdfUrl + citation counts + publicationInfo; use this (not a `site:arxiv.org` web hack) for every E-type academic query
- `arxiv`: direct arxiv.org API — canonical arxiv_id + published date, free and no API-key budget; pair with `serper_scholar` on E-type queries that look for papers (not repos). Catches recent preprints that Scholar has not yet indexed.

## Output (strict JSON, no other text)
```json
{
  "queries": [
    {"subquestion": "Q1", "role": "advocate", "query": "AI transcription accuracy Mandarin", "lang": "en", "engines": ["brave", "serper_en"]},
    {"subquestion": "Q1", "role": "advocate", "query": "<zh-TW query about speech-to-text accuracy>", "lang": "zh-TW", "engines": ["serper_tw"]},
    {"subquestion": "Q1", "role": "critic", "query": "AI transcription errors limitations", "lang": "en", "engines": ["brave", "serper_en"]}
  ]
}
```
Important: produce the actual Traditional Chinese query text for any zh-TW entry (the placeholder above is illustrative only).

## Discovery Query Family (round 1 must include these)
Beyond advocate / critic, each subquestion needs at least **2 discovery queries**:

| Category | Template | engines |
|----------|----------|---------|
| A latest tools | `best {topic} {YEAR}` / `top {topic} tools {YEAR}` | brave, serper_en |
| B competitors  | `alternative to {known tool} {YEAR}` / `{known tool} competitors {YEAR}` | brave, serper_en |
| C local (Taiwan) | `{topic} <Traditional Chinese recommendation phrase>` / `site:ithome.com.tw {topic}` | serper_tw (this engine only) |
| D community    | `site:reddit.com {topic} {YEAR} recommendation` | serper_en |
| E academic / source | `{entity} arxiv` / `{entity} paper pdf` → `arxiv, serper_scholar, brave`; `{entity} github` → `github, brave, serper_en` | (see inline) |

Rules:
- Every subquestion must cover at least category A + category C; categories B and D as budget allows.
- Category C queries must **only** use `engines: ["serper_tw"]` — never combine with brave or serper_en.
- If the Gap Log introduces a new tool name in a later round, add a category-B query for it.
- When the user_msg contains a "newly discovered entities" field, **produce at least 1 B-type follow-up query per entity** (e.g. `{entity name} review {YEAR}` or `alternative to {entity name}`).
- **Hard rule for academic / paper topics**: when the research involves keywords like paper, model, agent, SOTA, framework, or benchmark, **every entity in the `known tools` field must have at least 1 category-E query** (arxiv or github, pick one); these queries do not need a zh-TW counterpart.
- **Year-window policy** (relative time windows, NOT hardcoded single year):
  - **Academic / SOTA / paper / benchmark topics** (categories E, and any A/B/D query about specific models/frameworks/papers): **do NOT append any year**. Leading papers are often from prior years (e.g. AIDE 2024, MLE-bench 2024); hardcoding `{YEAR}` silently excludes them.
  - **Tool comparison / "best X" / alternatives / reviews** (categories A, B when they target living products): append the 2-year window `{LAST_2_YEARS}` (e.g. `best LLM coding agent 2025..2026`) — this captures current tools without missing last year's dominant option.
  - **Trend / forecast / prediction / "2026 outlook"** (typically category A forecast phrasing, D community predictions): append the single `{YEAR}` — these queries genuinely mean "this year".
  - **When in doubt, drop the year**. Over-specifying the year is the #1 cause of missed SOTA results; under-specifying costs at most 1 extra result page.
- **Off-topic gap rule**: when the Gap Log flags a subquestion with `[OFF_TOPIC_RATIO >= 0.5]`, do NOT reuse that SQ's previous angles; rewrite from a different angle (e.g. switch advocate phrasing, target a different sub-aspect, or use a different entity). Producing minor rewordings that repeat the same angle counts as a rule violation.
- **Per-SQ S_t rule** (Tongyi-style evolving state): the "Per-SQ evidence snapshot from last iteration (S_t)" section shows which SQs already have LIVE evidence (title + URL listed) and which still have `no LIVE evidence yet`. For SQs already covered by LIVE evidence, do NOT re-ask queries whose answers are visibly satisfied — probe the remaining gaps instead (missing sub-aspects, contradictions, newer evidence). For SQs marked `no LIVE evidence yet`, treat them as highest priority and rewrite the query angle aggressively.
- **Adversarial hard rule**: every subquestion must emit at least 1 query whose `role` is `critic` AND whose wording directly attacks the subquestion's premise. Acceptable phrasings: `"{subquestion topic} failures"`, `"why {approach} does not work"`, `"criticism of {approach}"`, `"{approach} antipattern"`, `"{approach} vs alternatives"`. This is separate from general "limitations" queries — it must name a concrete failure mode or alternative. A subquestion with only advocate + generic limitation queries violates this rule.
- Use the `{YEAR}` and `{LAST_2_YEARS}` values from the user_msg's "this round's time window" field.

## Taiwan source locking
When the research topic concerns Taiwan, prefer these site: prefixes to produce precise Traditional-Chinese queries:
- `site:ithome.com.tw`, `site:ithelp.ithome.com.tw` — IT media
- `site:techbang.com`, `site:kocpc.com.tw` — computer / 3C reviews
- `site:mobile01.com` — 3C forum
- `site:inside.com.tw`, `site:bnext.com.tw` — digital startup media
- `site:apps.apple.com` — App Store (ratings and reviews)

Such queries always pair with `engines: ["serper_tw"]`; do not add any other engine.

## Budget control
- Each query costs 1 search (multiple engines for the same query still count as one).
- Do not exceed the remaining budget.
"""


async def _plan_queries(
    *,
    plan: str,
    coverage: str,
    gap_log: str,
    sq_progress: str = "",
    iteration: int,
    remaining_budget: int,
    already_searched: list[str],
    depth: str,
    underfunded_sqs: list[str] | None = None,
    sq_counts: dict[str, int] | None = None,
    min_per_sq: int = 8,
    current_year: str = "",
    known_tools: list[str] | None = None,
    focus_sqs: list[str] | None = None,
    emerging_entities: list[str] | None = None,
) -> dict:
    """Stage 1: LLM plans the query list for the next round."""
    searched_text = (
        "\n".join(f"- {q}" for q in already_searched) if already_searched else "(none yet)"
    )

    # Budget guard: if any SQ is below min, inject a priority instruction
    if underfunded_sqs:
        sq_counts = sq_counts or {}
        underfunded_text = "\n".join(
            f"  - {sq}: searched {sq_counts.get(sq, 0)}/{min_per_sq} times"
            for sq in underfunded_sqs
        )
        sq_priority_section = f"""
## ⚠️ Subquestions with insufficient budget (must be topped up this round)
The following subquestions have not reached the minimum query count ({min_per_sq}). You **must** prioritise producing queries for them this round:
{underfunded_text}

Rule: until each of these subquestions reaches {min_per_sq} queries, do not produce extra queries for subquestions that already met the minimum.
"""
    else:
        sq_priority_section = ""

    # Focused refetch mode (triggered by trigger_fallback_node)
    if focus_sqs:
        focus_section = f"""
## 🎯 Focused refetch mode (Fallback)
**This round only refetches the following SQs; other subquestions are paused:**
{', '.join(focus_sqs)}

Refetch requirements:
- For each specified SQ at minimum: 1 advocate + 1 critic + 1 Discovery category-C (serper_tw) set
- Prioritise Taiwan-local resources (serper_tw) and academic/official resources (T1/T2 tier)
- Target the failure reasons in the Gap Log specifically
- Total queries must not exceed {remaining_budget}
"""
    else:
        focus_section = ""

    # Known tool seed (used by Discovery Query category B)
    if known_tools:
        tools_text = ", ".join(known_tools[:15])
        known_tools_section = f"\n## Known tools (for Discovery Query category B)\n{tools_text}\n"
    else:
        known_tools_section = ""

    # Time windows for Discovery Queries.
    #   {YEAR}         → current year (trend/forecast queries only)
    #   {LAST_2_YEARS} → e.g. "2025..2026" (tool-comparison queries)
    # Academic / SOTA / paper / benchmark queries append NEITHER; see the
    # Year-window policy in _PLANNER_SYSTEM.
    if current_year:
        try:
            cy = int(current_year)
            last_2_years = f"{cy - 1}..{cy}"
        except (TypeError, ValueError):
            cy = 2026
            last_2_years = "2025..2026"
        year_section = (
            f"\n## This round's time window\n"
            f"- `{{YEAR}}` = {current_year} (use for trend/forecast queries)\n"
            f"- `{{LAST_2_YEARS}}` = {last_2_years} (use for tool-comparison queries)\n"
            f"- Academic / SOTA / paper / benchmark queries: append neither.\n"
        )
    else:
        year_section = ""

    # newly discovered entities (extracted by _extract_emerging_entities from the prior round's results)
    if emerging_entities:
        entities_text = "\n".join(f"- {e}" for e in emerging_entities)
        emerging_section = (
            "\n## newly discovered entities (surfaced from last round's results; "
            "must have additional queries)\n"
            f"{entities_text}\n"
        )
    else:
        emerging_section = ""

    # Tongyi-style evolving state S_t: compact view of last round's LIVE evidence
    # per SQ. Planner uses this to avoid re-asking queries whose answers are
    # already in hand, and to prioritise SQs with zero LIVE evidence.
    if sq_progress:
        sq_progress_section = f"\n## Per-SQ evidence snapshot from last iteration (S_t)\n{sq_progress}\n"
    else:
        sq_progress_section = ""

    user_msg = f"""## Research plan
{plan}

## Coverage Checklist
{coverage}

## Gap Log
{gap_log}
{sq_progress_section}
## Already-searched queries (avoid duplicates)
{searched_text}
{focus_section}{sq_priority_section}{year_section}{known_tools_section}{emerging_section}
## This round's constraints
- This is round {iteration + 1} (round 1 uses the minimum set; later rounds add queries per coverage gap)
- Remaining search budget: {remaining_budget}
- Research depth: {depth}

Produce the query list to run this round."""

    # role="verifier" — query planning is structured JSON generation (logic task).
    # Doesn't need Opus's creative writing; using Gemini/GPT-fast saves 5-10 minutes.
    # max_tokens=8192: give JSON output enough room (thinking_budget=0 set in llm.py).
    response = await safe_ainvoke_chain(
        role="verifier",
        messages=[SystemMessage(content=_PLANNER_SYSTEM), HumanMessage(content=user_msg)],
        max_tokens=8192,
        temperature=0.2,
    )

    json_match = re.search(r"\{[\s\S]*\}", response.content)
    if not json_match:
        return {"queries": []}
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"queries": []}

    # Normalize: drop queries with missing fields or duplicates
    seen: set[str] = set()
    clean: list[dict] = []
    for q in data.get("queries", []):
        query = (q.get("query") or "").strip()
        if not query or query in seen:
            continue
        seen.add(query)
        # Normalize subquestion to Q{n} format — LLM might return a full title like
        # "Subquestion 1: mainstream tools survey" or "1" instead of "Q1".
        subq_raw = (q.get("subquestion") or "Q1").strip()
        sq_m = re.match(r"Q(\d+)", subq_raw, re.IGNORECASE)
        if sq_m:
            subq = f"Q{sq_m.group(1)}"
        else:
            num_m = re.search(r"(\d+)", subq_raw)
            subq = f"Q{num_m.group(1)}" if num_m else "Q1"
        clean.append({
            "subquestion": subq,
            "role": q.get("role", "advocate"),
            "query": query,
            "lang": q.get("lang", "en"),
            "engines": q.get("engines") or ["brave", "serper_en"],
        })
    return {"queries": clean}


# ---------------------------------------------------------------------------
# Duplicate query rollback (MiroThinker-style)
# ---------------------------------------------------------------------------
# Planner's prompt asks it to dedupe against already_searched, but LLMs still
# paraphrase. Cross-round detection here catches the paraphrase cases and drops
# them pre-execution so search budget is not wasted on near-identical queries.

def _detect_duplicate_queries(
    llm_queries: list[dict],
    already_searched: list[str],
    ratio: float = 0.9,
) -> tuple[list[dict], list[dict]]:
    """Separate planned queries into (kept, dropped) based on similarity to prior rounds.

    A query is dropped if:
      - normalized form matches any prior-round normalized form, OR
      - SequenceMatcher ratio against any prior query >= `ratio` (default 0.9)

    The 0.9 threshold is looser than the claim-dedup default (0.92) because query
    text is short (5-10 words) and even minor rewording changes ratio a lot.
    """
    prior_norm: set[str] = {normalize_for_dedup(q) for q in already_searched if q}
    prior_raw: list[str] = [q for q in already_searched if q]
    kept: list[dict] = []
    dropped: list[dict] = []
    for q in llm_queries:
        qtext = (q.get("query") or "").strip()
        if not qtext:
            continue
        qnorm = normalize_for_dedup(qtext)
        if qnorm in prior_norm:
            dropped.append(q)
            continue
        if any(is_near_duplicate(qtext, prev, ratio=ratio) for prev in prior_raw):
            dropped.append(q)
            continue
        kept.append(q)
        # Extend across within-batch paraphrases: subsequent queries must dedupe
        # against the ones we already accepted this round, not just prior rounds.
        prior_norm.add(qnorm)
        prior_raw.append(qtext)
    return kept, dropped


def _log_duplicate_rollback(workspace: str, iteration: int, dropped: list[dict]) -> None:
    """Write a `[DUPLICATE ROLLBACK]` entry to gap-log so next round's planner sees it.

    Rule enforced downstream: an SQ that accumulates 5 consecutive rounds of
    duplicate rollback is treated as stuck. The planner's prompt (off_topic_gap
    rule) already requires switching angle after such signals — we add one more
    surface here.
    """
    if not dropped:
        return
    from collections import Counter
    sq_counts = Counter(q.get("subquestion", "Q?") for q in dropped)
    lines = [
        f"\n\n## [DUPLICATE ROLLBACK] round {iteration + 1}",
        "Planner produced queries that near-duplicate prior rounds' queries "
        "(normalize or SequenceMatcher >= 0.9). Dropped before execution:",
    ]
    for sq, cnt in sq_counts.most_common():
        sample = next(
            (q.get("query", "") for q in dropped if q.get("subquestion") == sq),
            "",
        )
        lines.append(f"- **{sq}**: {cnt} duplicate queries dropped (e.g. `{sample[:80]}`)")
    lines.append(
        "Next round MUST switch angle for these SQs — not just reword synonyms. "
        "Acceptable switches: different sub-aspect, new entity, opposing stance, "
        "different time-window, or different source type (paper vs blog vs forum)."
    )
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _count_consecutive_stuck_rounds(gap_log: str, sq: str) -> int:
    """Count how many of the most-recent rounds tagged `[DUPLICATE ROLLBACK]`
    for this SQ consecutively (without an intervening round that did NOT
    trigger a rollback).

    Used by the planner caller to escalate to `[STUCK]` at >= 5.
    """
    if not gap_log or not sq:
        return 0
    rounds = re.findall(
        r"^## \[DUPLICATE ROLLBACK\] round (\d+)([\s\S]*?)(?=^## |\Z)",
        gap_log,
        re.MULTILINE,
    )
    if not rounds:
        return 0
    rounds.sort(key=lambda x: int(x[0]), reverse=True)
    consecutive = 0
    expected = int(rounds[0][0])
    for num_str, body in rounds:
        num = int(num_str)
        if num != expected:
            break
        if f"**{sq}**" in body:
            consecutive += 1
            expected -= 1
        else:
            break
    return consecutive


def _extract_searched_queries(exec_log: str) -> list[str]:
    """Fetch every already-searched query from execution-log.md.

    Matches the `- {query} [Q.../role/lang]` format that _append_execution_log writes.
    """
    if not exec_log:
        return []
    # Capture `- {query} [Q.../role/lang]` — the query is the text before [...]
    pattern = re.compile(r"^[-*]\s+(.+?)\s+\[[^\]]+\]\s*$", re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(exec_log)]


def _extract_sq_ids(plan: str) -> list[str]:
    """Extract ordered, unique subquestion IDs (Q1, Q2, ...) from phase0-plan.md."""
    matches = re.findall(r'\b(Q\d+)\b', plan)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


_REMOVED_MARKER_RE = re.compile(r"\[REMOVED:[^\]]*\]\([^)]+\)")


def _extract_seed_urls(plan: str) -> list[dict]:
    """Pull valid URLs + arxiv IDs out of the phase0 plan so phase1a can
    directly fetch them in round 1 without spending search budget.

    Why: in the 2026-04-14 failure workspace phase0 listed `arxiv.org/abs/2310.05193`
    (ResearchAgent) and several other legitimate papers in the plan body, and
    phase1a completely ignored them — then searched for "ResearchAgent paper"
    and didn't find this URL. Phase0 already validated these URLs against
    arxiv's API, so the URLs we see in the post-validation plan are the trusted
    set (hallucinated ones were replaced with `[REMOVED: ...]` markers).

    Each seed gets tagged with the SQ whose heading most recently preceded it,
    so coverage is correctly attributed (`## Q3: ...` paragraph containing an
    arxiv link → the seed is a Q3 source, not a Q1 source).

    Returns list of dicts shaped like search hits so they can be appended to
    ``selected`` before ``_fetch_pages`` without any pipeline changes.
    """
    if not plan:
        return []

    # First strip `[REMOVED: ...]` markers — we don't want to fetch known-bad URLs
    # that phase0 already marked as hallucinated.
    cleaned_plan = _REMOVED_MARKER_RE.sub("", plan)

    # Collect (position, url) so we can resolve the nearest preceding `## Q{n}:` heading.
    url_positions: list[tuple[int, str]] = []
    for m in re.finditer(r"https?://[^\s<>\"')\]]+", cleaned_plan):
        url = m.group(0).rstrip(".,;:!?)]}>，。；：、")
        if url:
            url_positions.append((m.start(), url))

    # Synthesize arxiv URLs from bare IDs (the validator already confirmed them).
    for m in re.finditer(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", cleaned_plan):
        url_positions.append((m.start(), f"https://arxiv.org/abs/{m.group(1)}"))

    if not url_positions:
        return []

    # Build a list of (position, Q-id) pairs for heading lookup.
    sq_positions: list[tuple[int, str]] = []
    for m in re.finditer(r"^##\s+(Q\d+)(?::|\s|$)", cleaned_plan, re.MULTILINE):
        sq_positions.append((m.start(), m.group(1)))

    def _nearest_sq(pos: int) -> str:
        nearest = "Q1"
        for hp, qid in sq_positions:
            if hp <= pos:
                nearest = qid
            else:
                break
        return nearest

    # Dedupe while preserving first-occurrence SQ attribution.
    seeds: list[dict] = []
    seen_urls: set[str] = set()
    for pos, url in sorted(url_positions):
        if url in seen_urls:
            continue
        seen_urls.add(url)
        seeds.append({
            "url": url,
            "title": "(seed from phase0 plan)",
            "description": "",
            "subquestion": _nearest_sq(pos),
            "role": "seed",
            "engines": ["seed"],
        })
    return seeds


def _parse_sq_texts(coverage_md: str) -> dict[str, str]:
    """Parse `## Q1: description` entries from coverage.chk into {sq_id: text}.

    Why: the LLM extractor previously saw only "Q1" as a label — not what Q1 was
    actually asking. So it extracted any reasonable-looking fact from the page,
    producing on-topic-shaped but off-topic claims. Tongyi DeepResearch's visit
    tool carries a `goal` on every page read and asks the extractor to first
    identify which sections are relevant; if none are, the page yields nothing.
    This function recovers the per-SQ goal text so we can apply the same gate.

    coverage.chk is authoritative: _generate_coverage_checklist already
    normalizes all phase0 plan formats (Q1:/Subquestion N:/DAG list) into the
    same `## Q1: text` shape, so we read that rather than re-parsing the plan.
    Placeholder lines `(to be filled by Phase 1a)` resolve to empty.
    """
    result: dict[str, str] = {}
    for m in re.finditer(r"^##\s+(Q\d+):\s*(.+)$", coverage_md, re.MULTILINE):
        sid, desc = m.group(1), m.group(2).strip()
        if desc and "(to be filled" not in desc.lower():
            result[sid] = desc
    return result


def _count_queries_per_sq(exec_log: str) -> dict[str, int]:
    """Count queries used per SQ from execution-log.md.

    Matches the `- {query} [Q{n}/role/lang]` format and reads the Q number inside brackets.
    """
    if not exec_log:
        return {}
    pattern = re.compile(r"^[-*]\s+.+?\s+\[(Q\d+)/[^\]]+\]\s*$", re.MULTILINE)
    counts: dict[str, int] = {}
    for m in pattern.finditer(exec_log):
        sq = m.group(1)
        counts[sq] = counts.get(sq, 0) + 1
    return counts


def _log_off_topic_ratio(
    workspace: str,
    iteration: int,
    raw_sources: list[dict],
    threshold: float = 0.5,
) -> None:
    """Per-SQ, how many sources got flagged OFF_TOPIC by the goal-aware extractor?

    Written into gap-log so the next round's Planner can see which SQs are getting
    systematically off-topic results and rewrite the angle. Complements
    `_log_budget_gaps` (quantity) with a quality signal.

    Only SQs whose OFF_TOPIC ratio >= threshold are logged — noise floor filter.
    """
    from collections import defaultdict
    per_sq_total: dict[str, int] = defaultdict(int)
    per_sq_off: dict[str, int] = defaultdict(int)
    for s in raw_sources:
        sq = s.get("subquestion") or ""
        if not sq or not s.get("content"):
            continue
        per_sq_total[sq] += 1
        if s.get("status") == "OFF_TOPIC":
            per_sq_off[sq] += 1

    flagged = []
    for sq, total in per_sq_total.items():
        if total < 2:
            continue
        off = per_sq_off.get(sq, 0)
        ratio = off / total
        if ratio >= threshold:
            flagged.append((sq, off, total, ratio))

    if not flagged:
        return

    lines = [f"\n\n## off-topic gap (after round {iteration})"]
    lines.append(
        "<!-- The goal-aware extractor rejected at least half of these SQs' sources "
        "as NOT_RELEVANT. Next round, rewrite the SQ's query angle rather than "
        "re-searching with minor rewording. -->"
    )
    for sq, off, total, ratio in flagged:
        lines.append(
            f"- {sq}: [OFF_TOPIC_RATIO >= {threshold:.1f}] "
            f"{off}/{total} sources were rejected as NOT_RELEVANT vs SQ goal — rewrite angle"
        )
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _build_sq_evidence_snapshot(
    workspace: str,
    iteration: int,
    raw_sources: list[dict],
    sq_texts: dict[str, str],
) -> None:
    """Per-SQ evidence snapshot written to sq-progress.md so the next iteration's
    Planner sees what was **actually found**, not only what failed.

    Tongyi DeepResearch Tech_Report §3.1 (Context Management): instead of passing
    the full interaction history, pass a compact "evolving report S_t" that
    summarises the current investigation state. This enables structured reasoning
    and prevents the planner from re-asking queries whose answers are already in
    hand, while still surfacing SQs with zero LIVE evidence as next-round priorities.

    Complements gap-log.md (negative signal: what failed) with a positive signal
    (what succeeded + which SQs still need help).
    """
    from collections import defaultdict
    if not raw_sources:
        return

    buckets: dict[str, dict] = defaultdict(
        lambda: {"live": [], "off_topic": 0, "unreachable": 0, "thin": 0}
    )
    for s in raw_sources:
        sq = s.get("subquestion") or ""
        if not sq:
            continue
        status = s.get("status")
        b = buckets[sq]
        if status == "LIVE":
            title = (s.get("title") or "").strip()[:100]
            url = s.get("url") or ""
            if title or url:
                b["live"].append((title, url))
        elif status == "OFF_TOPIC":
            b["off_topic"] += 1
        elif status == "UNREACHABLE":
            b["unreachable"] += 1
        elif status == "THIN_CONTENT":
            b["thin"] += 1

    if not buckets:
        return

    def _sort_key(sq: str) -> int:
        if sq.startswith("Q") and sq[1:].isdigit():
            return int(sq[1:])
        return 999

    lines = [f"\n\n## Iteration {iteration + 1} — per-SQ evidence snapshot"]
    lines.append(
        "<!-- Tongyi-style evolving state (S_t): compact view of what was found this "
        "round, so the next iteration's Planner can focus on genuine gaps rather than "
        "re-asking questions whose answers are already in hand. -->"
    )
    for sq in sorted(buckets.keys(), key=_sort_key):
        b = buckets[sq]
        sq_text = sq_texts.get(sq, "")
        header = f"- **{sq}**"
        if sq_text:
            header += f" — {sq_text[:80]}"
        lines.append(header)
        total = len(b["live"]) + b["off_topic"] + b["unreachable"] + b["thin"]
        status_parts = [f"{total} sources this round", f"LIVE {len(b['live'])}"]
        if b["off_topic"]:
            status_parts.append(f"OFF_TOPIC {b['off_topic']}")
        if b["unreachable"]:
            status_parts.append(f"UNREACHABLE {b['unreachable']}")
        if b["thin"]:
            status_parts.append(f"THIN {b['thin']}")
        lines.append(f"  - status: {', '.join(status_parts)}")
        if b["live"]:
            lines.append("  - LIVE evidence titles (planner: do not re-ask queries these already cover):")
            for title, url in b["live"][:5]:
                entry = f"    - {title or '(no title)'}"
                if url:
                    entry += f" — {url}"
                lines.append(entry)
        elif total > 0:
            lines.append(
                "  - **no LIVE evidence yet** — next iteration should rewrite the query angle for this SQ"
            )
    append_workspace_file(workspace, "sq-progress.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Planner reflection (open_deep_research `think_tool` pattern)
# ---------------------------------------------------------------------------
# open_deep_research forces the planner to invoke a `think_tool` after every
# batch of results, listing (a) what is still missing and (b) the next concrete
# step. Without this forced reflection the planner drifts and produces queries
# that don't target real gaps. We emulate it with a small structured LLM call
# appended to `sq-progress.md` so the next round's planner reads both the raw
# evidence snapshot AND the explicit reflection.

_REFLECTION_SYSTEM = """You are a research-loop reflector. Given the current
per-SQ evidence snapshot, coverage checklist, and gap log, produce a brief
reflection that the NEXT iteration's planner will read.

For each subquestion (Q1, Q2, ...) that still lacks coverage or strong evidence,
output exactly this block:

- **{SQ}**:
  - missing: <what concrete sub-aspect / entity / data point is not yet covered>
  - next step: <one specific query angle or source type to try next round — name entities or site:/API switches>

Rules:
- Be terse — one short line per field, <= 25 words each.
- Do NOT list SQs that already have 3+ LIVE sources AND are marked done in coverage.
- Prefer concrete names (tool name, paper title, domain) over vague phrases.
- If the same SQ has been `[DUPLICATE ROLLBACK]`-tagged recently, the next step
  MUST be a new angle (different sub-aspect, entity, or source type) — not a rewording.

Output raw markdown only (no code fences, no preamble)."""


async def _write_planner_reflection(
    workspace: str,
    iteration: int,
    sq_progress_snapshot: str,
    coverage: str,
    gap_log: str,
) -> None:
    """Append `## Iteration N — planner reflection` to sq-progress.md.

    Called right after `_build_sq_evidence_snapshot`. Errors are swallowed
    (reflection is a nice-to-have, not a hard dependency).
    """
    if not sq_progress_snapshot and not coverage:
        return
    user_msg = (
        f"## Per-SQ evidence snapshot (this round)\n{sq_progress_snapshot or '(empty)'}\n\n"
        f"## Coverage checklist\n{coverage or '(empty)'}\n\n"
        f"## Gap log (recent)\n{gap_log[-4000:] if gap_log else '(empty)'}\n\n"
        "Produce the reflection now."
    )
    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[
                SystemMessage(content=_REFLECTION_SYSTEM),
                HumanMessage(content=user_msg),
            ],
            max_tokens=2048,
            temperature=0.2,
        )
        body = (response.content or "").strip()
        if not body:
            return
        append_workspace_file(
            workspace,
            "sq-progress.md",
            f"\n\n## Iteration {iteration + 1} — planner reflection\n"
            "<!-- think_tool-style reflection: explicit 'missing + next step' per SQ, "
            "read by the next round's planner. -->\n"
            f"{body}\n",
        )
    except Exception as e:
        logger.warning("planner reflection skipped (%s)", e)


# ---------------------------------------------------------------------------
# Source-pool curator (gpt-researcher `skills/curator.py` pattern)
# ---------------------------------------------------------------------------
# gpt-researcher's SourceCurator asks an LLM to score each candidate source and
# prune marketing/off-topic pages before spending expensive generation on them.
# We run the curator AFTER extraction (post-hoc scoring) so it only annotates
# rather than blocks — phase1b can consume the scores later to tighten Bedrock
# grounding thresholds on low-scoring sources. Annotation-only keeps the blast
# radius small for the first PR.

_CURATOR_SYSTEM = FOCUSED_EXEC_PROMPT + """You are a source-pool curator. Given the research goal
(plan + coverage) and a list of fetched sources (id, url, title, tier, short
snippet, claim count), score each source on three dimensions:

- relevance   (0-5): how directly does this source address the plan's subquestions?
- credibility (0-5): source tier (T1/T2 > T3/T4 > T5/T6), author identifiability, publisher reputation.
- quant_value (0-5): does the source carry specific numbers/benchmarks/dates we can cite? (0 = pure opinion, 5 = data paper)

Return STRICT JSON, no other text:
```json
{"scores":[{"source_id":"S001","relevance":4,"credibility":3,"quant_value":2,"note":"12 words max — one-line reason"}, ...]}
```

Rules:
- Output one entry per input source_id — do not skip, do not invent new IDs.
- note <= 12 words; use hints such as "off-topic marketing", "seminal paper",
  "blog summary of a real paper", "forum anecdote".
- If a source is clearly off-topic for every subquestion, give relevance=0."""


async def _curate_sources(
    raw_sources: list[dict],
    extractions: list[dict],
    plan: str,
    coverage: str,
) -> list[dict]:
    """Score each fetched source on relevance/credibility/quant_value.

    Returns list shaped like:
      [{"source_id":"S001","relevance":4,"credibility":3,"quant_value":2,"note":"..."}, ...]
    The caller writes these scores to `source-curation.md` and a low-score
    warning to gap-log if average < 2.0.
    """
    # Only score sources that actually made it through fetching (skip UNREACHABLE).
    live_sources = [s for s in raw_sources if s.get("content") or s.get("status") == "THIN_CONTENT"]
    if not live_sources:
        return []
    claim_counts = {e.get("source_id", ""): len(e.get("claims", []) or []) for e in extractions}

    bullet_lines = []
    for s in live_sources:
        sid = s.get("source_id", "S???")
        url = s.get("url", "")
        title = (s.get("title") or "")[:100]
        snippet = (s.get("content") or "")[:300].replace("\n", " ")
        tier = classify_tier(url, title, s.get("content") or "") if s.get("status") != "THIN_CONTENT" else "T6"
        bullet_lines.append(
            f"- {sid} | tier {tier} | {s.get('subquestion', '?')}/{s.get('role', '?')} "
            f"| {url}\n  title: {title}\n  claims: {claim_counts.get(sid, 0)}\n  snippet: {snippet}"
        )
    user_msg = (
        f"## Research plan\n{plan[:3000]}\n\n"
        f"## Coverage checklist\n{coverage[:2000]}\n\n"
        f"## Fetched sources (score every one)\n" + "\n".join(bullet_lines)
    )
    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[
                SystemMessage(content=_CURATOR_SYSTEM),
                HumanMessage(content=user_msg),
            ],
            max_tokens=4096,
            temperature=0.1,
        )
        json_match = re.search(r"\{[\s\S]*\}", response.content or "")
        if not json_match:
            return []
        data = json.loads(json_match.group())
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("source curator skipped (%s)", e)
        return []

    # Normalise and clip scores
    scores = []
    seen_ids: set[str] = set()
    for item in data.get("scores", []):
        sid = (item.get("source_id") or "").strip()
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        try:
            rel = max(0, min(5, int(item.get("relevance", 0))))
            cred = max(0, min(5, int(item.get("credibility", 0))))
            quant = max(0, min(5, int(item.get("quant_value", 0))))
        except (TypeError, ValueError):
            continue
        note = (item.get("note") or "").strip()[:100]
        scores.append({
            "source_id": sid,
            "relevance": rel,
            "credibility": cred,
            "quant_value": quant,
            "note": note,
        })
    return scores


def _write_source_curation(workspace: str, iteration: int, scores: list[dict]) -> None:
    """Write source curator scores to `source-curation.md`.

    One cumulative file (append-only) so later phases can read the full scoring
    history. Low-scoring sources (avg < 2.0) also get a warning in gap-log so
    the planner/phase2 know to treat their claims with caution.
    """
    if not scores:
        return
    header_needed = not (
        read_workspace_file(workspace, "source-curation.md") or ""
    ).startswith("| source_id")
    lines: list[str] = []
    if header_needed:
        lines.append(
            "| source_id | round | relevance | credibility | quant_value | avg | note |\n"
            "|-----------|-------|-----------|-------------|-------------|-----|------|"
        )
    low: list[tuple[str, float]] = []
    for s in scores:
        avg = (s["relevance"] + s["credibility"] + s["quant_value"]) / 3
        lines.append(
            f"| {s['source_id']} | {iteration + 1} | {s['relevance']} "
            f"| {s['credibility']} | {s['quant_value']} | {avg:.2f} | {s['note']} |"
        )
        if avg < 2.0:
            low.append((s["source_id"], avg))
    append_workspace_file(workspace, "source-curation.md", "\n".join(lines) + "\n")

    if low:
        warn = [f"\n\n## [CURATOR WARNING] low-quality sources (round {iteration + 1})"]
        warn.append("<!-- avg < 2.0 on relevance/credibility/quant_value — phase1b should tighten grounding thresholds. -->")
        for sid, avg in low:
            warn.append(f"- {sid}: avg {avg:.2f}")
        append_workspace_file(workspace, "gap-log.md", "\n".join(warn) + "\n")


def _log_budget_gaps(
    workspace: str,
    iteration: int,
    sq_ids: list[str],
    sq_counts: dict[str, int],
    min_per_sq: int,
) -> None:
    """Append any SQs still below minimum budget to the "Budget gaps" section of gap-log.md.

    Called at the end of each round so the next round's Planner can see which SQs
    need priority top-up via gap_log.
    """
    gaps = [
        (sq, sq_counts.get(sq, 0))
        for sq in sq_ids
        if sq_counts.get(sq, 0) < min_per_sq
    ]
    if not gaps:
        return
    lines = [f"\n\n## budget gap (after round {iteration})"]
    for sq, have in gaps:
        need = min_per_sq - have
        lines.append(f"- {sq}: searched {have}, minimum required {min_per_sq}, still short by {need}")
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _extract_known_tools(plan: str) -> list[str]:
    """Extract known tool / service names from the research plan text for the
    planner's Discovery Query category B.

    Look inside parentheses (CJK or ASCII) for lists separated by enumeration or
    comma, and keep items that contain latin letters. Do not split on whitespace,
    to preserve multi-word tool names like "Clova Note" or "Good Tape".
    """
    bracket_content = re.findall(r'\(([^)\n]{3,120})\)', plan)
    candidates: list[str] = []
    for content in bracket_content:
        # Split on commas / semicolons (tool names may contain spaces)
        parts = re.split(r'[,;]+', content)
        for p in parts:
            p = p.strip()
            # Strip trailing "etc." / "et al." and leading list-leaders like "such as" / "e.g."
            p = re.sub(r'\s+(etc\.?|et al\.?)$', '', p, flags=re.IGNORECASE).strip()
            p = re.sub(r'^(such as|e\.g\.|eg\.?|including|like)\s+', '', p, flags=re.IGNORECASE).strip()
            # Keep 2-40-char entries containing latin letters (tool names are usually English or mixed)
            if 1 < len(p) <= 40 and re.search(r'[A-Za-z]', p):
                candidates.append(p)
    # Dedupe preserving order; return at most 20
    seen: set[str] = set()
    result: list[str] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= 20:
            break
    return result


_ENTITY_LEADER_RE = re.compile(
    r"^(such\s+as|including|like|e\.g\.|eg\.?|i\.e\.|ie\.?)\s+",
    re.IGNORECASE,
)
_ENTITY_SUFFIX_RE = re.compile(
    r"\s+(tools?|services?|products?|applications?|implementations?|"
    r"frameworks?|models?|systems?|platforms?|packages?|libraries)$",
    re.IGNORECASE,
)
_ENTITY_CAPS_RE = re.compile(r"^[A-Z][\w.\-]*(?:\s+[A-Z0-9][\w.\-]*){0,3}$")
# Allow camelCase / hyphens / dots (GPT-4.1 / LangGraph / LangChain / OpenAI) and
# compound words where lowercase butts up against uppercase (e.g. "OpenAI deep research"
# yields "OpenAI" via this pattern; the tail "deep research" is picked up by the
# bullet/label pattern below).


def _clean_entity_candidate(raw: str) -> str:
    """Normalize an entity candidate — strip leading descriptors, trailing
    meta suffixes, and punctuation. Returns empty string when invalid."""
    name = raw.strip()
    if not name:
        return ""
    # Strip leading leader words
    name = _ENTITY_LEADER_RE.sub("", name).strip()
    # Strip trailing meta suffixes like "tools" / "services"
    name = _ENTITY_SUFFIX_RE.sub("", name).strip()
    name = re.sub(r"\s+(etc\.?|et al\.?)$", "", name, flags=re.IGNORECASE).strip()
    # Strip trailing punctuation
    name = name.rstrip(".,:;)]}")
    # Strip leading bullet / numbering
    name = re.sub(r"^[-*+•·●◆■□▪▫]+\s*", "", name).strip()
    name = re.sub(r"^\d+[\.\)]\s+", "", name).strip()
    return name


def _is_valid_entity(name: str) -> bool:
    if not (2 <= len(name) <= 60):
        return False
    if not re.search(r"[A-Za-z]", name):
        return False
    # Too many spaces → probably a sentence
    if name.count(" ") > 4:
        return False
    # All-lowercase with multiple spaces → looks like prose
    if re.search(r"\s", name) and name == name.lower():
        return False
    # Ends with "?" / "!" → probably a question
    if re.search(r"[?!]$", name):
        return False
    return True


def _extract_brief_entities(text: str) -> list[str]:
    """Extract entity names from research-brief.md / phase0-plan.md.

    Three patterns:
      A Comma-separated inline list of 3+ capitalized words (rejecting full sentences)
      B Markdown bullet leading an entity (e.g. `- AIDE` / `- MLE-Agent: ...`)
      C Labelled list following tags like facets / known tools / evaluation target

    Plus a fallback: 2+ capitalized words inside parentheses, comma-separated.
    """
    if not text:
        return []

    seen_lower: set[str] = set()
    entities: list[str] = []

    def _add(raw: str) -> None:
        name = _clean_entity_candidate(raw)
        if not _is_valid_entity(name):
            return
        key = name.lower()
        if key in seen_lower:
            return
        seen_lower.add(key)
        entities.append(name)

    # Pattern A: comma-separated inline list (evaluate each line independently, need >=3 capitalized candidates)
    # Rather than matching the whole line with _ENTITY_CAPS_RE — the first item often has prefix
    # pollution, the last item often has suffix pollution. Use findall to extract the "longest
    # name-token" from each split fragment instead.
    entity_token_re = re.compile(r"[A-Z][\w.\-]*(?:\s+[A-Z0-9][\w.\-]*){0,3}")
    for line in text.splitlines():
        if not re.search(r",", line):
            continue
        raw_parts = [p for p in re.split(r",", line) if p.strip()]
        caps_candidates: list[str] = []
        for p in raw_parts:
            cand = p.strip()
            # Strip parenthetical addenda first
            cand = re.sub(r"\s*\([^)]*\)\s*", "", cand).strip()
            # First try full clean + whole-string match (handles clean cases like "AutoML-Agent")
            cleaned = _clean_entity_candidate(cand)
            if cleaned and _ENTITY_CAPS_RE.match(cleaned):
                caps_candidates.append(cleaned)
                continue
            # Fallback: extract the longest name-token from the fragment
            matches = entity_token_re.findall(cand)
            if matches:
                caps_candidates.append(max(matches, key=len))
        if len(caps_candidates) >= 3:
            for c in caps_candidates:
                _add(c)

    # Pattern B: markdown bullet leading line (at most one entity per bullet)
    bullet_re = re.compile(r"^\s*[-*+]\s+([^\n]+)", re.MULTILINE)
    for m in bullet_re.finditer(text):
        body = m.group(1).strip()
        head = re.split(r"[:(]|\s+[-—]\s+|\s*—\s*", body, maxsplit=1)[0].strip()
        head = _clean_entity_candidate(head)
        if head and _ENTITY_CAPS_RE.match(head):
            _add(head)

    # Pattern C: labelled list (facets, known tools, evaluation target, ...)
    label_re = re.compile(
        r"(?:facets?|entities|known\s*tools|known\s*entities|tool\s*list|"
        r"evaluation\s*target|important\s*tools|core\s*tools|core\s*targets|"
        r"tools?/entities|research\s*targets?)\s*:\s*([^\n]+)",
        re.IGNORECASE,
    )
    for m in label_re.finditer(text):
        content = m.group(1)
        # Strip parenthetical addenda outright
        content = re.sub(r"\([^)]*\)", " ", content)
        for part in re.split(r"[,/\|]", content):
            _add(part)

    # Fallback: comma list inside parentheses (>=2 capitalized candidates)
    for inner in re.findall(r"\(([^)\n]{3,160})\)", text):
        parts = [p.strip() for p in re.split(r",+", inner) if p.strip()]
        caps = [p for p in parts if _ENTITY_CAPS_RE.match(_clean_entity_candidate(p))]
        if len(caps) >= 2:
            for p in caps:
                _add(p)

    return entities[:20]


def _seed_paper_queries(
    entities: list[str],
    sq_ids: list[str] | None,
    budget_cap: int,
) -> list[dict]:
    """Generate 2 mandatory paper/repo queries per brief entity (arxiv + github).

    Parallel to the A/B/C/D query families produced by the LLM planner — these are the
    **must-search** E-class seed queries. Each entity gets up to 2 queries
    (`{entity} arxiv`, `{entity} github`), round-robin assigned across sq_ids to avoid
    quota imbalance. `role="seed"` maps to the relaxed `_QUOTA_PER_ROLE["seed"]=6` quota.
    """
    if not entities or budget_cap <= 0:
        return []

    sqs = sq_ids or ["Q1"]
    queries: list[dict] = []
    idx = 0
    for kind in ("arxiv", "github"):
        # ``github`` and ``arxiv`` are now first-class engines with direct
        # API calls (see _run_single_search). Web-search engines stay as
        # secondary fallbacks in case the official API is rate-limited or
        # returns an empty set.
        engines = (
            ["arxiv", "serper_scholar", "brave"] if kind == "arxiv"
            else ["github", "brave", "serper_en"]
        )
        for entity in entities:
            if len(queries) >= budget_cap:
                return queries
            queries.append({
                "subquestion": sqs[idx % len(sqs)],
                "role": "seed",
                "query": f"{entity} {kind}",
                "lang": "en",
                "engines": engines,
            })
            idx += 1
    return queries


async def _extract_emerging_entities(
    raw_sources: list[dict],
    plan: str,
    iteration: int,
) -> list[str]:
    """Extract new tool/product/service names from this round's search results that are not yet in the plan.

    Uses an LLM (role="verifier") to extract tool names from the title + content snippet
    of each source, excludes those already known in the plan, and returns up to 15
    deduplicated names.
    """
    if not raw_sources:
        return []

    # Collect title + first 500 chars of content (enough for LLM to identify tool names, avoids context bloat)
    snippets: list[str] = []
    for s in raw_sources:
        title = s.get("title", "")
        content = s.get("content", "") or ""
        if title or content:
            snippets.append(f"[{s.get('source_id', '?')}] {title}\n{content[:500]}")
    if not snippets:
        return []

    combined = "\n\n---\n\n".join(snippets[:30])  # up to 30 sources

    # Tools already known in the plan (exclusion list)
    known_in_plan = _extract_known_tools(plan)
    known_text = ", ".join(known_in_plan) if known_in_plan else "(none)"

    prompt = f"""The following are titles and summaries from the latest round of search results.

Task: Identify tool / software / app / service / hardware product names mentioned in the text, **excluding** the following tools already known in the research plan:
{known_text}

Output format: one name per line, up to 15 names. Only output names, no explanations.

Search results:
{combined}"""

    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[HumanMessage(content=prompt)],
            max_tokens=1024,
            temperature=0.1,
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"_extract_emerging_entities LLM call failed: {e}")
        return []

    # Parse output: one per line, filter invalid lines
    entities: list[str] = []
    seen_set: set[str] = set()
    known_lower = {k.lower() for k in known_in_plan}

    for line in raw_text.splitlines():
        name = line.strip().lstrip("-•·* ").strip()
        if not name or len(name) < 2 or len(name) > 60:
            continue
        if name.lower() in known_lower:
            continue
        if name not in seen_set:
            seen_set.add(name)
            entities.append(name)
        if len(entities) >= 15:
            break

    logger.info(f"Round {iteration + 1} newly discovered entities: {entities}")
    return entities


def _extract_emerging_from_gap_log(gap_log: str, iteration: int) -> list[str]:
    """Parse the most recently written "newly discovered entities" list from gap-log.md.

    Only picks the last `## newly discovered entities (round N)` section, to avoid
    passing old rounds' entities to the Planner again.
    """
    # Find all `## newly discovered entities (round N)` section starts
    pattern = re.compile(r"## newly discovered entities \(round \d+\)\n(.*?)(?=\n## |\Z)", re.DOTALL)
    matches = list(pattern.finditer(gap_log))
    if not matches:
        return []

    # Take the last section
    last_match = matches[-1]
    block = last_match.group(1)

    entities: list[str] = []
    for line in block.splitlines():
        name = line.strip().lstrip("-•·* ").strip()
        if name and 2 <= len(name) <= 60:
            entities.append(name)
        if len(entities) >= 15:
            break
    return entities


# ---------------------------------------------------------------------------
# Stage 2: Executor — search + urlhealth + fetch raw
# ---------------------------------------------------------------------------

async def _execute_searches(
    queries: list[dict],
    remaining_budget: int,
) -> tuple[list[dict], int]:
    """Run all search tasks in parallel. Returns (hits, searches_used).

    hits: [{"subquestion", "role", "query", "lang", "engine", "url", "title", "description"}, ...]
    Each (query, engine) combination consumes 1 search, but the number of engines is
    bounded by remaining_budget.
    """
    tasks: list[asyncio.Task] = []
    task_meta: list[dict] = []
    searches_used = 0

    for q in queries:
        for engine in q["engines"]:
            if searches_used >= remaining_budget:
                break
            searches_used += 1
            tasks.append(asyncio.create_task(_run_single_search(q["query"], engine)))
            task_meta.append({
                "subquestion": q["subquestion"],
                "role": q["role"],
                "query": q["query"],
                "lang": q["lang"],
                "engine": engine,
            })
        if searches_used >= remaining_budget:
            break

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    hits: list[dict] = []
    for meta, result in zip(task_meta, raw_results):
        if isinstance(result, Exception) or not result:
            continue
        for item in result:
            hits.append({
                **meta,
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
            })
    return hits, searches_used


async def _run_single_search(query: str, engine: str) -> list[dict]:
    """Dispatch to the appropriate underlying search function for each engine."""
    # Scholar-intent queries on generic web engines get an auto site: filter
    # (Whisper P1-6) — see _maybe_upgrade_to_scholar_query for rationale.
    query = _maybe_upgrade_to_scholar_query(query, engine)
    try:
        if engine == "brave":
            return await brave_search(query, count=10)
        if engine == "serper_en":
            return await serper_search(query, gl="us", hl="en", num=10)
        if engine == "serper_tw":
            return await serper_search(query, gl="tw", hl="zh-TW", num=10)
        if engine == "serper_cn":
            return await serper_search(query, gl="cn", hl="zh-CN", num=10)
        if engine == "serper_scholar":
            # Hit the real Google Scholar endpoint (not site:arxiv.org on web search):
            # gives us publication info, citation counts, and pdf links that
            # generic web search strips out.
            return await serper_scholar(query, num=10)
        if engine == "arxiv":
            # Direct arxiv API — no key needed, returns canonical arxiv_id +
            # published date so validate_arxiv_id can later confirm it's real.
            # Complements serper_scholar: catches papers Google Scholar hasn't
            # indexed yet (new arxiv submissions lag in Scholar by days/weeks).
            return await arxiv_search(query, max_results=10)
        if engine == "github":
            # Direct GitHub search — bypasses generic-web ranking that
            # frequently misses canonical repos (failed-workspace had AIDE,
            # MLE-Agent, ResearchAgent buried under marketing blogs).
            # Sort by stars to surface the authoritative repo card.
            return await github_repo_search(query, max_results=10, sort="stars")
    except Exception:
        return []
    return []


async def _verify_urls(hits: list[dict]) -> dict[str, str]:
    """Batch-invoke the urlhealth CLI for all hit URLs, returns {url: status}."""
    urls = list({h["url"] for h in hits if h.get("url")})
    if not urls:
        return {}

    payload = json.dumps({"urls": urls})
    try:
        proc = await asyncio.create_subprocess_exec(
            _URLHEALTH_PY, _URLHEALTH_CLI, "--cli",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate(payload.encode("utf-8"))
        data = json.loads(stdout.decode("utf-8"))
    except Exception:
        # Fall back to UNKNOWN when verification fails; subsequent fetch will still try,
        # and failures will be logged as UNREACHABLE
        return {u: "UNKNOWN" for u in urls}

    status_map: dict[str, str] = {}
    for r in data.get("results", []):
        status_map[r.get("url", "")] = r.get("status", "UNKNOWN")
    return status_map


def _select_urls_by_quota(
    hits: list[dict],
    url_health: dict[str, str],
    queries: list[dict],
) -> list[dict]:
    """Pick URLs for deep reading based on (subquestion, role) quota.

    Rules:
      - Skip LIKELY_HALLUCINATED
      - The same URL hit by multiple engines gets a cross-engine score bonus
      - Each (subq, role) picks up to _QUOTA_PER_ROLE[role] URLs
    """
    # Step 1: aggregate — merge same-URL hits across engines
    by_url: dict[str, dict] = {}
    for h in hits:
        url = h.get("url", "")
        if not url:
            continue
        status = url_health.get(url, "UNKNOWN")
        if status == "LIKELY_HALLUCINATED":
            continue
        bucket = by_url.setdefault(url, {
            "url": url,
            "title": h["title"],
            "description": h["description"],
            "status": status,
            "subq_roles": set(),
            "engines": set(),
        })
        bucket["subq_roles"].add((h["subquestion"], h["role"]))
        bucket["engines"].add(h["engine"])

    # Step 2: Sort each (subq, role) bucket independently + take the quota
    buckets: dict[tuple[str, str], list[dict]] = {}
    for info in by_url.values():
        score = len(info["engines"])  # cross-engine hit
        for subq, role in info["subq_roles"]:
            item = {**info, "score": score, "subquestion": subq, "role": role}
            buckets.setdefault((subq, role), []).append(item)

    selected: list[dict] = []
    selected_urls: set[str] = set()
    for (subq, role), items in buckets.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        quota = _QUOTA_PER_ROLE.get(role, 1)
        for item in items:
            if len(selected) >= 1000:  # safety cap
                break
            if item["url"] in selected_urls:
                continue
            if sum(1 for s in selected if s["subquestion"] == subq and s["role"] == role) >= quota:
                break
            selected.append(item)
            selected_urls.add(item["url"])
    return selected


async def _fetch_pages(
    selected: list[dict],
    workspace: str,
    id_start: int = 1,
) -> list[dict]:
    """Three-tier fetch: WebFetch -> Serper scrape -> mark UNREACHABLE.

    Each source is written to workspace/search-results/{subq}/{source_id}_raw.md
    Returns the meta dict for each source (includes source_id, url, raw_path, content, status...)
    `id_start` is used for cross-round cumulative numbering so that round 2 does not
    restart from S001 and collide with round 1.
    """
    tasks = [_fetch_one(item, i + 1) for i, item in enumerate(selected)]
    fetched = await asyncio.gather(*tasks)

    raw_sources: list[dict] = []
    for offset, (item, (content, method)) in enumerate(zip(selected, fetched)):
        sid = f"S{id_start + offset:03d}"
        subq = item["subquestion"]
        if content:
            raw_path = f"search-results/{subq}/{sid}_raw.md"
            header = (
                f"# {item['title']}\n\n"
                f"- URL: {item['url']}\n"
                f"- Fetch Method: {method}\n"
                f"- Fetched Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"---\n\n"
            )
            write_workspace_file(workspace, raw_path, header + content[:_RAW_CHAR_LIMIT])
        else:
            raw_path = ""
        # THIN_CONTENT overrides the url_health status so downstream can detect and filter it
        effective_status = "THIN_CONTENT" if method == "thin_content" else item["status"]
        raw_sources.append({
            "source_id": sid,
            "url": item["url"],
            "title": item["title"],
            "description": item["description"],
            "subquestion": subq,
            "role": item["role"],
            "engines": list(item["engines"]),
            "status": effective_status,
            "fetch_method": method,
            "content": content,
            "raw_path": raw_path,
        })
    return raw_sources


async def _fetch_one(item: dict, idx: int) -> tuple[str, str]:
    """Three-tier fetch for a single URL. Returns (content, method).

    method values:
      "web_fetch"      - WebFetch succeeded and >= 500 chars
      "serper_scrape"  - Serper scrape succeeded and >= 500 chars
      "thin_content"   - at least returned something but < 500 chars (truncated page or no substantive content)
      "unreachable"    - completely unreachable (content is empty string)

    A title ending in ellipsis/... hints that WebFetch only got a truncated version;
    always retry with Serper scrape in that case.
    """
    url = item["url"]
    title = item.get("title", "")
    thin_candidate = ""  # Record the longest < 500 chars content as a fallback

    # 1. WebFetch
    try:
        text = await web_fetch(url)
        if text:
            if len(text.strip()) >= 500:
                return text, "web_fetch"
            elif text.strip():
                thin_candidate = text   # Returned something but too short
    except Exception:
        pass

    # 2. Serper scrape
    # Conditions: haven't fetched enough content yet, or title is truncated (hints WebFetch only got a stub)
    title_truncated = title.rstrip().endswith(("...", "…"))
    if SERPER_API_KEY and (not thin_candidate or title_truncated):
        try:
            text = await serper_scrape(url)
            if text:
                if len(text.strip()) >= 500:
                    return text, "serper_scrape"
                elif len(text.strip()) > len(thin_candidate.strip()):
                    thin_candidate = text
        except Exception:
            pass

    if thin_candidate:
        return thin_candidate, "thin_content"
    return "", "unreachable"


# ---------------------------------------------------------------------------
# Stage 3: Extractor — independent LLM per source, fresh context
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM = FOCUSED_EXEC_PROMPT + """You are a precise transcriber. Extract evidence relevant to the subquestion from a single source document.

## Task
For this one source document, in this exact order:
1. **Relevance check** (rational): identify which sections of the source address the research goal. If NO section addresses the goal — even if the page contains well-written, factual-looking content — set rational to "NOT_RELEVANT" and return empty quotes / numbers / claims. Off-topic pages produce off-topic claims that look grounded, so the gate goes here at the entrance, not at scoring time.
2. Locate passages relevant to the subquestion (at most 3)
3. Copy the key sentences verbatim as QUOTE (no rewriting, summarizing, or merging)
4. Record sentences containing numbers separately as NUMBER (must include the full original sentence)
5. Based on these QUOTE/NUMBER items, form 1-3 pending claims

## Hard rules
- QUOTE must be verbatim from the source; rewriting is forbidden
- **The text/sentence fields of QUOTE and NUMBER must preserve the source's original language. If the source is English, quote text must be English; if the source is Traditional Chinese (zh-TW) or Simplified Chinese (zh-CN), quote text must be in that language. Translation is strictly prohibited.**
- claim_text should be written in English, but the text/sentence fields of QUOTE/NUMBER must remain in the source language and must not be translated
- NUMBER must include the complete original sentence containing the number (in source language)
- Every claim must link to at least 1 quote_id or number_id
- Inferences without explicit evidence must not be turned into claims

## Forbidden extraction content (common noise, always skip)
The following content is unrelated to the research subquestion; even if it appears in the source, it must NOT be extracted as a claim:
- Physical address, postal code, or street number of a company/organization
- Contact phone numbers, support email, customer service links
- Company employee headcount, office city/country, and other company-profile background info
- Company founding year, founder names (background info unrelated to product features or performance)
- Cookie notices, privacy policies, terms of use, advertisement copy
- Website header / footer / navigation boilerplate
- SEO marketing phrases (e.g., "Our team of experts", "We are dedicated to...", "Contact us today")

## start_char / end_char fields (important)
For each QUOTE / NUMBER, additionally output start_char and end_char,
representing the character-level start/end index of that text within "the source above"
(0-based, Python string slicing semantics).
Verification: raw_content[start_char:end_char] must equal text (or sentence).

Inaccurate estimates are not penalized — the program will fall back to locating the text
by string search.
But the text field itself must still be a verbatim excerpt from the source, otherwise
the entire item will be rejected.

If text appears multiple times in the source, the index of any one occurrence is fine
(the program will take the first matching position it can find).

## Output (strict JSON)
```json
{
  "rational": "1-2 sentence statement of which passages of the source address the research goal. If none, write exactly NOT_RELEVANT followed by a one-sentence reason.",
  "quotes": [
    {"quote_id": "Q1", "text": "verbatim sentence from source", "start_char": 123, "end_char": 234}
  ],
  "numbers": [
    {"number_id": "N1", "value": "92.5", "unit": "%",
     "sentence": "full original sentence containing the number", "start_char": 456, "end_char": 567}
  ],
  "claims": [
    {
      "claim_text": "one-sentence factual statement",
      "claim_type": "numeric|comparative|causal|forecast|qualitative",
      "evidence_quote_ids": ["Q1"],
      "evidence_number_ids": ["N1"]
    }
  ]
}
```

If rational begins with NOT_RELEVANT, quotes / numbers / claims MUST be empty arrays. Only output JSON, no other text."""


# Shared index-based quote utilities: resolve_quote_index / verify_indexed_items now live in
# deep_research.harness.validators; we keep local module-level aliases for readability.
_resolve_quote_index = resolve_quote_index


def _verify_indexed_items(raw, items, text_field, chunk_offset=0):
    return verify_indexed_items(
        raw, items, text_field, chunk_offset=chunk_offset, log_prefix="Tier1/index"
    )


async def _extract_all_sources(
    raw_sources: list[dict],
    workspace: str,
    sq_texts: dict[str, str] | None = None,
) -> list[dict]:
    """Independent LLM call per source for extraction. rate_limiter is configured in llm.py.

    THIN_CONTENT sources are skipped (do not enter claim extraction).

    sq_texts maps each Q id to its goal text (from coverage.chk). Passed through
    so the extractor can gate off-topic pages; falls back to empty string when
    the SQ text is unknown, in which case extraction works as before.
    """
    sq_texts = sq_texts or {}
    eligible = [s for s in raw_sources if s["content"] and s.get("status") != "THIN_CONTENT"]
    tasks = [_extract_one(src, workspace, sq_texts.get(src.get("subquestion", ""), "")) for src in eligible]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    cleaned: list[dict] = []
    for src, res in zip(eligible, results):
        if isinstance(res, Exception) or not res:
            cleaned.append({"source_id": src["source_id"], "quotes": [], "numbers": [], "claims": []})
            continue
        if res.get("off_topic"):
            src["status"] = "OFF_TOPIC"
        cleaned.append({"source_id": src["source_id"], **res})
    return cleaned


def _strip_html_for_extraction(content: str) -> str:
    """Strip HTML tags and decode entities from raw fetched content before LLM extraction.

    HTML pages contain CSS/JS noise that confuses the extraction LLM. After stripping:
    1. The LLM receives clean, readable text — verbatim quotes become findable strings.
    2. Tier1 verification uses the same stripped text, so extracted quotes always match.

    Without this, LLMs often generate Chinese translations of English quotes (which fail
    Tier1 find) or quote from JS code instead of visible content.
    """
    import html as _html_module
    # Remove script and style blocks entirely (JS/CSS noise)
    content = re.sub(
        r'<(script|style)[^>]*>.*?</(script|style)>',
        ' ', content, flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove all other HTML tags
    content = re.sub(r'<[^>]+>', ' ', content)
    # Decode HTML entities (&amp; → &, &#8217; → ', etc.)
    content = _html_module.unescape(content)
    # Collapse whitespace
    content = re.sub(r'[ \t]+', ' ', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


async def _extract_one(src: dict, workspace: str, sq_text: str = "") -> dict | None:
    """Run the Extractor on a single source and write the final S{id}.md.

    Short docs (<= _CHUNK_SIZE) use a single LLM call; long docs use sliding-window
    chunked extraction: each chunk is extracted by an independent LLM call concurrently,
    then merged, deduplicated, and renumbered.

    sq_text is the research-goal body text for the SQ this source was searched
    under (empty if the coverage.chk has no text for this Q). When non-empty, the
    extractor is asked to first gate "is any part of this source relevant to the
    goal?" — a NOT_RELEVANT verdict returns an empty result with off_topic=True
    so the caller can mark the source and skip claim emission.

    Why we chunk: feeding a 25K-char source as a single block to the LLM causes
    mid-document quotes/numbers to be missed due to Lost in the Middle, violating
    hard rule 4 "complete citation chain". Chunked + overlap ensures every section
    has near-range context.
    """
    raw_original = (src.get("content") or "")[:_RAW_CHAR_LIMIT]
    # Strip HTML if the content is an HTML page (CSS/JS noise causes extraction failures).
    # Use the SAME stripped text for both LLM extraction AND Tier1 verification so that
    # verbatim quotes extracted by the LLM are always findable in the reference string.
    if raw_original and "<html" in raw_original[:2000].lower():
        raw = _strip_html_for_extraction(raw_original)
        # Preserve metadata header (lines before ---) in the raw for readability
        if len(raw) < 300:
            raw = raw_original  # Stripping left too little — keep original
    else:
        raw = raw_original

    if not raw:
        return None

    if len(raw) <= _CHUNK_SIZE:
        data_raw = await _extract_one_pass(src, raw, sq_text=sq_text)
        if not data_raw:
            return None
        # Goal-aware gate: NOT_RELEVANT at the entrance short-circuits extraction
        # so off-topic pages never get a chance to emit claims that look grounded.
        rational = (data_raw.get("rational") or "").strip()
        if rational.upper().startswith("NOT_RELEVANT"):
            logger.info("[%s] extractor marked OFF_TOPIC vs SQ goal; skipping claim emission", src["source_id"])
            return {"quotes": [], "numbers": [], "claims": [], "off_topic": True, "rational": rational}
        # Short doc: LLM sees the full raw, indices are global. One-pass verification suffices.
        data = {
            "quotes": _verify_indexed_items(raw, data_raw.get("quotes", []), "text"),
            "numbers": _verify_indexed_items(raw, data_raw.get("numbers", []), "sentence"),
            "claims": data_raw.get("claims", []),
        }
    else:
        data = await _extract_chunked(src, raw, sq_text=sq_text)
        if data.get("off_topic"):
            logger.info("[%s] all chunks NOT_RELEVANT; skipping claim emission", src["source_id"])
            return {"quotes": [], "numbers": [], "claims": [], "off_topic": True}

    if not data:
        return None

    # Renumber ids: ensures Q1, Q2, ... within a single source are consecutive and unique
    # (especially important after merging in chunked mode)
    sid = src["source_id"]
    qid_map: dict[str, str] = {}
    quotes: list[dict] = []
    for i, q in enumerate(data.get("quotes", []), 1):
        text = (q.get("text") or "").strip()
        if not text:
            continue
        new_qid = f"{sid}-Q{i}"
        orig = q.get("quote_id", "")
        if orig:
            qid_map[orig] = new_qid
        quotes.append({
            "quote_id": new_qid,
            "text": text,
            "start": q.get("start"),
            "end": q.get("end"),
        })

    nid_map: dict[str, str] = {}
    numbers: list[dict] = []
    for i, n in enumerate(data.get("numbers", []), 1):
        sentence = (n.get("sentence") or "").strip()
        if not sentence:
            continue
        new_nid = f"{sid}-N{i}"
        orig = n.get("number_id", "")
        if orig:
            nid_map[orig] = new_nid
        numbers.append({
            "number_id": new_nid,
            "value": n.get("value", ""),
            "unit": n.get("unit", ""),
            "sentence": sentence,
            "start": n.get("start"),
            "end": n.get("end"),
        })

    claims = []
    for c in data.get("claims", []):
        text = (c.get("claim_text") or "").strip()
        if not text:
            continue
        q_ids = [qid_map.get(x, f"{sid}-{x}") for x in c.get("evidence_quote_ids", [])]
        n_ids = [nid_map.get(x, f"{sid}-{x}") for x in c.get("evidence_number_ids", [])]
        claims.append({
            "claim_text": text,
            "claim_type": c.get("claim_type", "qualitative"),
            "quote_ids": q_ids + n_ids,
            "source_id": sid,
            "subquestion": src["subquestion"],
        })

    # --- Tier 1 hard-rule validation ------------------------------------
    # Hard rules 2 + 4: quote must truly exist in source + quote_id referenced by claim must truly exist
    # On violation, drop (don't pass downstream to be misused); log via logger.warning for debugging.
    quotes, numbers, claims = _apply_tier1_validation(
        sid=sid,
        raw_content=raw,
        quotes=quotes,
        numbers=numbers,
        claims=claims,
    )

    # Write the final S{id}.md (phase1b grounding reads this)
    _write_source_file(workspace, src, quotes, numbers)

    return {"quotes": quotes, "numbers": numbers, "claims": claims}


def _apply_tier1_validation(
    sid: str,
    raw_content: str,
    quotes: list[dict],
    numbers: list[dict],
    claims: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Tier 1 hard rules: drop quotes not found in source and quote_ids absent from the ledger.

    Flow:
      1. validate_quotes_indexed: drop quotes whose start/end is invalid or
         source[s:e] != text (strict version of hard rule 2). _verify_indexed_items has
         already dropped unlocatable ones earlier; this is a double safety net — ensuring
         every quote seen downstream in phase1a can be reconstructed purely by index.
      2. Same validation for numbers (using sentence/start/end).
         Legacy quote format (no start/end) will be flagged as a violation in steps 1/2;
         if backward compatibility is needed (e.g. external quotes not going through phase1a),
         validate_quotes_exist can be invoked additionally.
      3. validate_quote_ids_in_ledger: a claim referencing a non-existent quote_id is
         treated as violating hard rule 4 — remove the invalid quote_id from the claim;
         if the claim loses all quote_ids -> drop the claim entirely.
    """
    if not raw_content:
        # No source content (e.g. fetch-failed source) — Tier 1 passes through; later phases will handle it
        return quotes, numbers, claims

    # 1. quotes — pure index validation (strictest)
    quote_violations = validate_quotes_indexed(quotes, raw_content)
    bad_quote_ids: set[str] = set()
    if quote_violations:
        for v in quote_violations:
            logger.warning("[%s][Tier1/indexed] %s", sid, v)
            bad_id = v.split(":", 1)[0].strip()
            bad_quote_ids.add(bad_id)
        quotes = [q for q in quotes if q.get("quote_id") not in bad_quote_ids]

    # 2. numbers — same index validation (sentence as text_field)
    num_violations = validate_quotes_indexed(numbers, raw_content)
    bad_number_ids: set[str] = set()
    if num_violations:
        for v in num_violations:
            logger.warning("[%s][Tier1/indexed] number %s", sid, v)
            bad_id = v.split(":", 1)[0].strip()
            bad_number_ids.add(bad_id)
        numbers = [n for n in numbers if n.get("number_id") not in bad_number_ids]

    # 3. claims: strip invalid quote_id references; drop the entire claim if its quote_ids become empty
    bad_ids = bad_quote_ids | bad_number_ids
    valid_ids = (
        {q.get("quote_id") for q in quotes if q.get("quote_id")}
        | {n.get("number_id") for n in numbers if n.get("number_id")}
    )

    cleaned_claims: list[dict] = []
    for c in claims:
        cleaned_qids = [qid for qid in c.get("quote_ids", []) if qid in valid_ids and qid not in bad_ids]
        if not cleaned_qids:
            logger.warning(
                "[%s][Tier1] claim '%s' has no valid quote_ids after Tier 1 — dropped",
                sid, (c.get("claim_text") or "")[:60],
            )
            continue
        cleaned_claims.append({**c, "quote_ids": cleaned_qids})

    return quotes, numbers, cleaned_claims


async def _extract_one_pass(
    src: dict,
    content: str,
    chunk_idx: int | None = None,
    chunk_total: int | None = None,
    sq_text: str = "",
) -> dict | None:
    """Single LLM call: extract quote/number/claim from a content segment (could be full doc or one chunk).

    sq_text carries the full subquestion description (parsed from coverage.chk) so the
    LLM can perform the relevance check at the front — mirrors Tongyi DeepResearch's
    goal-aware `readpage_jina(url, goal)` pattern. Without this, the extractor only sees
    "Q1" as a label and produces on-topic-shaped claims from off-topic pages.
    """
    chunk_note = ""
    if chunk_idx is not None and chunk_total is not None:
        chunk_note = (
            f"\n\n(This is chunk {chunk_idx + 1}/{chunk_total} of a long document; "
            f"extract only from this chunk and do not assume surrounding context.)"
        )

    goal_section = ""
    if sq_text:
        goal_section = f"""## Research Goal (what this source should answer)
{src['subquestion']}: {sq_text}

"""

    user_msg = f"""## Subquestion
{src['subquestion']} (role: {src['role']})

{goal_section}## Source metadata
- source_id: {src['source_id']}
- url: {src['url']}
- title: {src['title']}{chunk_note}

## Source text
{content}

Based on the source text above, extract QUOTE / NUMBER / claim."""

    try:
        # role="verifier" — extraction/verification task; Gemini-led (lowest hallucination rate for grounded summarization)
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[SystemMessage(content=_EXTRACTOR_SYSTEM), HumanMessage(content=user_msg)],
            max_tokens=4096,
            temperature=0.1,
        )
    except Exception:
        return None

    json_match = re.search(r"\{[\s\S]*\}", response.content)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


async def _extract_chunked(src: dict, content: str, sq_text: str = "") -> dict:
    """Sliding-window extraction for long docs: split into chunks -> concurrent LLM calls -> merge + dedupe.

    sq_text is piped through to each chunk so every chunk's LLM pass can see the goal and
    perform its own relevance check. A page is flagged off_topic=True only when EVERY
    chunk returns NOT_RELEVANT — a page with one relevant section still emits claims.

    Within each chunk, quote_id / number_id are prefixed with c{chunk_idx}- to avoid
    collisions; the outer _extract_one will then renumber IDs finally.

    Index handling:
      - The LLM sees only a single chunk, so its output start_char/end_char are chunk-local.
      - _verify_indexed_items(chunk_content, items, ..., chunk_offset=chunk_global_pos)
        converts the chunk-local verified index into the global coordinate of the full raw.
      - Deduplication keys on (start, end) tuple — more precise than deduplicating by
        text content (the same passage extracted twice in the chunk-overlap region will
        share the same global span).

    Claim deduplication still uses claim_text (claims don't always have spans).
    """
    step = max(_CHUNK_SIZE - _CHUNK_OVERLAP, 1)
    chunks: list[str] = []
    offsets: list[int] = []
    pos = 0
    while pos < len(content):
        chunk = content[pos : pos + _CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
            offsets.append(pos)
        pos += step
        if pos >= len(content):
            break
    if not chunks:
        return {"quotes": [], "numbers": [], "claims": []}

    total = len(chunks)
    tasks = [_extract_one_pass(src, chunk, idx, total, sq_text=sq_text) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_quotes: list[dict] = []
    merged_numbers: list[dict] = []
    merged_claims: list[dict] = []
    successful_chunks = 0
    not_relevant_chunks = 0
    seen_quote_spans: set[tuple[int, int]] = set()
    seen_number_spans: set[tuple[int, int]] = set()
    # Upgraded claim dedup: use normalize_for_dedup key instead of raw string
    # key = normalized text -> catches near-duplicates differing only in punctuation/whitespace
    seen_claim_norm: set[str] = set()
    # Also keep raw text for is_near_duplicate comparison (handles near-duplicates that
    # remain distinct after normalization)
    seen_claim_raw: list[str] = []

    for chunk_idx, res in enumerate(results):
        if isinstance(res, Exception) or not res:
            continue
        successful_chunks += 1
        rational = (res.get("rational") or "").strip()
        if rational.upper().startswith("NOT_RELEVANT"):
            not_relevant_chunks += 1
            continue
        chunk_content = chunks[chunk_idx]
        chunk_offset = offsets[chunk_idx]

        verified_q = _verify_indexed_items(
            chunk_content,
            res.get("quotes", []),
            "text",
            chunk_offset=chunk_offset,
        )
        for q in verified_q:
            span = (q["start"], q["end"])
            if span in seen_quote_spans:
                continue
            seen_quote_spans.add(span)
            merged_quotes.append({
                "quote_id": f"c{chunk_idx}-{q.get('quote_id', 'Q?')}",
                "text": q["text"],
                "start": q["start"],
                "end": q["end"],
            })

        verified_n = _verify_indexed_items(
            chunk_content,
            res.get("numbers", []),
            "sentence",
            chunk_offset=chunk_offset,
        )
        for n in verified_n:
            span = (n["start"], n["end"])
            if span in seen_number_spans:
                continue
            seen_number_spans.add(span)
            merged_numbers.append({
                "number_id": f"c{chunk_idx}-{n.get('number_id', 'N?')}",
                "value": n.get("value", ""),
                "unit": n.get("unit", ""),
                "sentence": n["sentence"],
                "start": n["start"],
                "end": n["end"],
            })

        for c in res.get("claims", []):
            text = (c.get("claim_text") or "").strip()
            if not text:
                continue
            norm = normalize_for_dedup(text)
            # 1) Fast exact match after normalization
            if norm in seen_claim_norm:
                continue
            # 2) SequenceMatcher near-duplicate (common in chunk-overlap region with minor rephrasing)
            if any(is_near_duplicate(text, prev) for prev in seen_claim_raw):
                continue
            seen_claim_norm.add(norm)
            seen_claim_raw.append(text)
            merged_claims.append({
                "claim_text": text,
                "claim_type": c.get("claim_type", "qualitative"),
                "evidence_quote_ids": [
                    f"c{chunk_idx}-{x}" for x in c.get("evidence_quote_ids", [])
                ],
                "evidence_number_ids": [
                    f"c{chunk_idx}-{x}" for x in c.get("evidence_number_ids", [])
                ],
            })

    all_chunks_off_topic = (
        successful_chunks > 0
        and not_relevant_chunks == successful_chunks
        and not merged_claims
    )
    return {
        "quotes": merged_quotes,
        "numbers": merged_numbers,
        "claims": merged_claims,
        "off_topic": all_chunks_off_topic,
    }


def _write_source_file(
    workspace: str,
    src: dict,
    quotes: list[dict],
    numbers: list[dict],
) -> None:
    subq = src["subquestion"]
    sid = src["source_id"]
    path = f"search-results/{subq}/{sid}.md"
    lines = [
        f"# Source {sid}: {src['title']}",
        "",
        f"- URL: {src['url']}",
        f"- Fetched Title: {src['title']}",
        f"- URL Status: {src['status']}",
        f"- Fetch Method: {src.get('fetch_method', 'unknown')}",
        f"- Fetch Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"- Engines: {', '.join(src['engines'])}",
        f"- Role: {src['role']}",
        f"- Subquestion: {subq}",
        "",
        "## Verbatim Quotes",
        "",
        "<!-- Each @[start:end] is a TextSpan (relative to the truncated raw_content, "
        "0-based, end-exclusive). Downstream can verify/render via raw[start:end] slicing. -->",
    ]
    for q in quotes:
        span = _format_span(q.get("start"), q.get("end"))
        lines.append(f'QUOTE[{q["quote_id"]}]{span}: "{q["text"]}"')
    for n in numbers:
        span = _format_span(n.get("start"), n.get("end"))
        lines.append(
            f'NUMBER[{n["number_id"]}]{span}: {n["value"]} {n["unit"]} — '
            f'Original: "{n["sentence"]}"'
        )
    write_workspace_file(workspace, path, "\n".join(lines) + "\n")


def _format_span(start, end) -> str:
    """Format start/end index as @[s:e]; empty string if either is not an int (backward-compatible)."""
    if isinstance(start, int) and isinstance(end, int):
        return f" @[{start}:{end}]"
    return ""


# ---------------------------------------------------------------------------
# Stage 4: Registry
# ---------------------------------------------------------------------------

def _update_source_registry(workspace: str, raw_sources: list[dict]) -> None:
    """Append new sources to source-registry.md."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    for s in raw_sources:
        if not s["content"] and s.get("status") != "THIN_CONTENT":
            continue  # UNREACHABLE is not registered in the main registry; THIN_CONTENT stays for transparency
        engines = ",".join(s["engines"])
        # THIN_CONTENT forces T6; others classified by domain
        if s.get("status") == "THIN_CONTENT":
            tier = "T6"
        else:
            tier = classify_tier(s["url"], s.get("title", ""), s.get("content", ""))
        lines.append(
            f"| {s['source_id']} | {s['url']} | {s['title']} | {s['title']} "
            f"| {tier} | {s['status']} | {today} | {engines} | {s['role']} | {s['subquestion']} |"
        )
    if lines:
        append_workspace_file(workspace, "source-registry.md", "\n".join(lines) + "\n")


def _append_execution_log(
    workspace: str,
    iteration: int,
    queries: list[dict],
    searches_used: int,
) -> None:
    """Append this round to execution-log.md, updating the searched-query list and search count."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = [
        "",
        f"### Round {iteration + 1} [{ts}] (used {searches_used} searches)",
    ]
    # Append to the Searched Queries list
    query_lines = [f"- {q['query']} [{q['subquestion']}/{q['role']}/{q['lang']}]" for q in queries]
    block.append("")
    append_workspace_file(workspace, "execution-log.md", "\n".join(block) + "\n" + "\n".join(query_lines) + "\n")


def _log_unreachable(
    workspace: str,
    url_health: dict[str, str],
    raw_sources: list[dict],
) -> None:
    """Record UNREACHABLE / THIN_CONTENT / LIKELY_HALLUCINATED URLs to gap-log.md."""
    bad = [s for s in raw_sources if not s["content"]]
    thin = [s for s in raw_sources if s.get("status") == "THIN_CONTENT"]
    hallucinated = [u for u, st in url_health.items() if st == "LIKELY_HALLUCINATED"]
    if not bad and not thin and not hallucinated:
        return
    lines = [""]
    if bad:
        lines.append("### UNREACHABLE URLs (all three fetch tiers failed)")
        for s in bad:
            lines.append(f"- {s['url']} [{s['subquestion']}/{s['role']}]")
    if thin:
        lines.append("### THIN_CONTENT URLs (< 500 chars, skipped claim extraction)")
        for s in thin:
            chars = len((s.get("content") or "").strip())
            lines.append(f"- {s['url']} [{s['subquestion']}/{s['role']}] ({chars} chars)")
    if hallucinated:
        lines.append("### LIKELY_HALLUCINATED URLs (urlhealth verdict)")
        for u in hallucinated:
            lines.append(f"- {u}")
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _extract_domain(url: str) -> str:
    """Extract hostname from URL; strip the www. prefix."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.") if host else ""
    except Exception:
        return ""


def _log_domain_bias(
    workspace: str,
    iteration: int,
    existing_sources: list,
    new_raw_sources: list[dict],
    threshold: float = 0.30,
) -> None:
    """Scan the domain distribution across all accumulated sources.

    If a domain's share exceeds threshold (default 30%), append a [BIAS WARNING]
    to gap-log.md so that Phase 2 can lower the confidence of claims from that
    domain during integration.
    """
    domain_counts: dict[str, int] = {}

    # Existing Source objects (from prior rounds)
    for s in existing_sources:
        url = s.url if hasattr(s, "url") else (s.get("url", "") if isinstance(s, dict) else "")
        d = _extract_domain(url)
        if d:
            domain_counts[d] = domain_counts.get(d, 0) + 1

    # Newly fetched this round (raw dicts)
    for s in new_raw_sources:
        url = s.get("url", "")
        if s.get("status") == "UNREACHABLE":
            continue  # Unfetched sources don't count
        d = _extract_domain(url)
        if d:
            domain_counts[d] = domain_counts.get(d, 0) + 1

    total = sum(domain_counts.values())
    if total == 0:
        return

    biased: list[tuple[str, int, float]] = []
    for domain, count in domain_counts.items():
        pct = count / total
        if pct > threshold:
            biased.append((domain, count, pct))

    if not biased:
        return

    biased.sort(key=lambda x: -x[2])
    lines = [
        f"\n\n## [BIAS WARNING] Source domain concentration too high (cumulative at round {iteration + 1})",
        f"The following domains account for > {threshold:.0%} of all fetched sources; possible self-promotion or viewpoint bias:",
    ]
    for domain, count, pct in biased:
        lines.append(f"- **{domain}**: {count}/{total} sources ({pct:.0%})")
    lines.append(
        "During Phase 2 integration, claims from these domains should be marked CONFLICTING "
        "unless corroborated by an independent T1-T3 source."
    )
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Build returned sources / claims
# ---------------------------------------------------------------------------

def _build_sources(raw_sources: list[dict]) -> list[Source]:
    out: list[Source] = []
    for s in raw_sources:
        if not s["content"]:
            continue
        role = s["role"] if s["role"] in ("advocate", "critic", "perspective") else "advocate"
        out.append(Source(
            source_id=s["source_id"],
            url=s["url"],
            title=s["title"],
            fetched_title=s["title"],
            tier="T6" if s.get("status") == "THIN_CONTENT" else classify_tier(s["url"], s.get("title", ""), s.get("content", "")),
            url_status=s["status"] if s["status"] in ("LIVE", "STALE", "UNREACHABLE", "UNKNOWN", "THIN_CONTENT") else "UNKNOWN",
            fetch_date=datetime.now().strftime("%Y-%m-%d"),
            engines=s["engines"],
            role=role,
            subquestion=s["subquestion"],
        ))
    return out


def _next_source_id_index(existing: list) -> int:
    """Compute the next S{n} numbering starting point from existing sources."""
    max_n = 0
    for s in existing:
        sid = s.source_id if hasattr(s, "source_id") else s.get("source_id", "")
        m = re.match(r"S(\d+)", sid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _collect_claims(extractions: list[dict], existing_claims: list | None = None) -> list[Claim]:
    """Expand all pending claims produced by Extractor into Claim objects.

    `existing_claims` is used for cross-round cumulative numbering, so round 2 does
    not restart from Q1-C1 and collide. Also filters out claims near-duplicate with
    existing_claims via is_near_duplicate (cross-round dedup).
    """
    out: list[Claim] = []
    counter: dict[str, int] = {}

    # Build a fast-lookup structure for cross-round dedup: each subquestion -> list of existing claim texts
    # Also keep a normalized-key set for O(1) quick filtering
    existing_raw: dict[str, list[str]] = {}   # subq -> [raw_texts]
    existing_norm: dict[str, set[str]] = {}   # subq -> {norm_texts}

    if existing_claims:
        for c in existing_claims:
            cid = c.claim_id if hasattr(c, "claim_id") else c.get("claim_id", "")
            sq = c.subquestion if hasattr(c, "subquestion") else c.get("subquestion", "")
            # Match any claim_id format ending in -C{n}: "Q1-C3", "SQ1-C3", etc.
            m = re.match(r"(.+)-C(\d+)$", cid)
            if m:
                subq_c = sq or m.group(1)  # prefer subquestion field over ID prefix
                n = int(m.group(2))
                counter[subq_c] = max(counter.get(subq_c, 0), n)
            txt = c.claim_text if hasattr(c, "claim_text") else c.get("claim_text", "")
            if sq and txt:
                existing_raw.setdefault(sq, []).append(txt)
                existing_norm.setdefault(sq, set()).add(normalize_for_dedup(txt))

    for ext in extractions:
        sid = ext.get("source_id", "S???")
        # Build per-source lookup so each claim's quote_ids can be resolved to
        # verbatim text. Both quotes (text field) and numbers (sentence field)
        # are referenced by claims via a shared quote_ids list — we merge them.
        qid_to_text: dict[str, str] = {}
        for q in ext.get("quotes", []) or []:
            qid = (q.get("quote_id") or "").strip()
            txt = (q.get("text") or "").strip()
            if qid and txt:
                qid_to_text[qid] = txt
        for n in ext.get("numbers", []) or []:
            nid = (n.get("number_id") or "").strip()
            sent = (n.get("sentence") or "").strip()
            if nid and sent:
                qid_to_text[nid] = sent

        for c in ext.get("claims", []):
            subq = c.get("subquestion", "Q1")
            claim_text = (c.get("claim_text") or "").strip()
            if not claim_text:
                continue

            # Cross-round dedup: first O(1) check via normalized set, then SequenceMatcher
            norm = normalize_for_dedup(claim_text)
            if norm in existing_norm.get(subq, set()):
                continue
            if any(is_near_duplicate(claim_text, prev) for prev in existing_raw.get(subq, [])):
                continue

            counter[subq] = counter.get(subq, 0) + 1
            claim_id = f"{subq}-C{counter[subq]}"
            ctype = c.get("claim_type", "qualitative")
            if ctype not in ("numeric", "comparative", "causal", "forecast", "qualitative"):
                ctype = "qualitative"

            # Resolve quote_ids -> verbatim snippets so downstream MARCH blind
            # check sees original text (not opaque `S001-Q3` IDs). Keep the ID
            # list for backwards compatibility; evidence_quotes is additive.
            qids = c.get("quote_ids", [])
            evidence = [qid_to_text[q] for q in qids if q in qid_to_text]

            out.append(Claim(
                claim_id=claim_id,
                subquestion=subq,
                claim_text=claim_text,
                claim_type=ctype,
                source_ids=[sid],
                quote_ids=qids,
                evidence_quotes=evidence,
                status="pending",
            ))

            # Add the new claim to the dedup pool to prevent duplicates from different sources in the same batch
            existing_raw.setdefault(subq, []).append(claim_text)
            existing_norm.setdefault(subq, set()).add(norm)

    return out
