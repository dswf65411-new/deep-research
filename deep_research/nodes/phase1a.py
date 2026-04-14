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
from deep_research.state import Claim, ResearchState, Source
from deep_research.tools.search import (
    BRAVE_API_KEY,
    SERPER_API_KEY,
    brave_search,
    serper_search,
    serper_scrape,
    web_fetch,
)
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

    # gap_log accumulates round over round (by round 5 it holds every UNREACHABLE /
    # MISSING / CONFLICT from rounds 1-4). For the Planner, only recent gaps carry
    # useful signal; older content is just a distractor. Keep the last ~2000 chars
    # (~650 tokens) to avoid context amplifier effects.
    if len(gap_log) > 2000:
        gap_log = "...[earlier content omitted, keeping only recent gaps]...\n\n" + gap_log[-2000:]
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
    extractions = await _extract_all_sources(raw_sources, workspace)

    # ── Stage 4: Registry ────────────────────────────────────────────
    _update_source_registry(workspace, raw_sources)
    _append_execution_log(workspace, iteration, queries, searches_used)
    _log_unreachable(workspace, url_health, raw_sources)
    _log_domain_bias(workspace, iteration, existing_sources, raw_sources)

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

_PLANNER_SYSTEM = """You are the research-search Planner. Given the research plan and current coverage, produce the list of search queries to run next round.

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
- `serper_scholar`: Google + site:arxiv.org/semanticscholar.org

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
| E academic / source | `{entity} arxiv` / `{entity} github` / `{entity} paper pdf` | serper_scholar, brave |

Rules:
- Every subquestion must cover at least category A + category C; categories B and D as budget allows.
- Category C queries must **only** use `engines: ["serper_tw"]` — never combine with brave or serper_en.
- If the Gap Log introduces a new tool name in a later round, add a category-B query for it.
- When the user_msg contains a "newly discovered entities" field, **produce at least 1 B-type follow-up query per entity** (e.g. `{entity name} review {YEAR}` or `alternative to {entity name}`).
- **Hard rule for academic / paper topics**: when the research involves keywords like paper, model, agent, SOTA, framework, or benchmark, **every entity in the `known tools` field must have at least 1 category-E query** (arxiv or github, pick one); these queries do not need a zh-TW counterpart.
- Use the {YEAR} value from the user_msg's "this round's year" field.

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

    # Current year (used for the {YEAR} placeholder in Discovery Queries)
    year_section = (
        f"\n## This round's year\n- Current year: {current_year or '2026'} "
        "(use this number for year values inside queries)\n"
        if current_year else ""
    )

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

    user_msg = f"""## Research plan
{plan}

## Coverage Checklist
{coverage}

## Gap Log
{gap_log}

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
        engines = (
            ["serper_scholar", "brave"] if kind == "arxiv"
            else ["brave", "serper_en"]
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
            # Prefix the query with site:arxiv.org OR site:semanticscholar.org
            scholar_q = f"({query}) (site:arxiv.org OR site:semanticscholar.org)"
            return await serper_search(scholar_q, gl="us", hl="en", num=10)
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

_EXTRACTOR_SYSTEM = """You are a precise transcriber. Extract evidence relevant to the subquestion from a single source document.

## Task
For this one source document:
1. Locate passages relevant to the subquestion (at most 3)
2. Copy the key sentences verbatim as QUOTE (no rewriting, summarizing, or merging)
3. Record sentences containing numbers separately as NUMBER (must include the full original sentence)
4. Based on these QUOTE/NUMBER items, form 1-3 pending claims

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

Only output JSON, no other text."""


# Shared index-based quote utilities: resolve_quote_index / verify_indexed_items now live in
# deep_research.harness.validators; we keep local module-level aliases for readability.
_resolve_quote_index = resolve_quote_index


def _verify_indexed_items(raw, items, text_field, chunk_offset=0):
    return verify_indexed_items(
        raw, items, text_field, chunk_offset=chunk_offset, log_prefix="Tier1/index"
    )


async def _extract_all_sources(raw_sources: list[dict], workspace: str) -> list[dict]:
    """Independent LLM call per source for extraction. rate_limiter is configured in llm.py.

    THIN_CONTENT sources are skipped (do not enter claim extraction).
    """
    eligible = [s for s in raw_sources if s["content"] and s.get("status") != "THIN_CONTENT"]
    tasks = [_extract_one(src, workspace) for src in eligible]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    cleaned: list[dict] = []
    for src, res in zip(eligible, results):
        if isinstance(res, Exception) or not res:
            cleaned.append({"source_id": src["source_id"], "quotes": [], "numbers": [], "claims": []})
            continue
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


async def _extract_one(src: dict, workspace: str) -> dict | None:
    """Run the Extractor on a single source and write the final S{id}.md.

    Short docs (<= _CHUNK_SIZE) use a single LLM call; long docs use sliding-window
    chunked extraction: each chunk is extracted by an independent LLM call concurrently,
    then merged, deduplicated, and renumbered.

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
        data_raw = await _extract_one_pass(src, raw)
        if not data_raw:
            return None
        # Short doc: LLM sees the full raw, indices are global. One-pass verification suffices.
        data = {
            "quotes": _verify_indexed_items(raw, data_raw.get("quotes", []), "text"),
            "numbers": _verify_indexed_items(raw, data_raw.get("numbers", []), "sentence"),
            "claims": data_raw.get("claims", []),
        }
    else:
        data = await _extract_chunked(src, raw)

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
) -> dict | None:
    """Single LLM call: extract quote/number/claim from a content segment (could be full doc or one chunk)."""
    chunk_note = ""
    if chunk_idx is not None and chunk_total is not None:
        chunk_note = (
            f"\n\n(This is chunk {chunk_idx + 1}/{chunk_total} of a long document; "
            f"extract only from this chunk and do not assume surrounding context.)"
        )

    user_msg = f"""## Subquestion
{src['subquestion']} (role: {src['role']})

## Source metadata
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


async def _extract_chunked(src: dict, content: str) -> dict:
    """Sliding-window extraction for long docs: split into chunks -> concurrent LLM calls -> merge + dedupe.

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
    tasks = [_extract_one_pass(src, chunk, idx, total) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_quotes: list[dict] = []
    merged_numbers: list[dict] = []
    merged_claims: list[dict] = []
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

    return {
        "quotes": merged_quotes,
        "numbers": merged_numbers,
        "claims": merged_claims,
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
            out.append(Claim(
                claim_id=claim_id,
                subquestion=subq,
                claim_text=claim_text,
                claim_type=ctype,
                source_ids=[sid],
                quote_ids=c.get("quote_ids", []),
                status="pending",
            ))

            # Add the new claim to the dedup pool to prevent duplicates from different sources in the same batch
            existing_raw.setdefault(subq, []).append(claim_text)
            existing_norm.setdefault(subq, set()).add(norm)

    return out
