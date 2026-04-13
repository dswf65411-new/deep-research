"""Phase 1a: Planner-Executor-Extractor。

取代舊的 ReAct agent 設計。原本 create_react_agent 把每輪 tool_result（網頁全文）
都累積到下一次 LLM 請求，導致單次 request 輕易超過 Anthropic 30K ITPM。

新架構分 4 階段，每次 LLM call 的 context 都固定且小：
  1. Planner (LLM × 1)    讀 plan/coverage/gap，產出 query 清單
  2. Executor (程式)      平行搜尋、urlhealth 驗證、並發 WebFetch/Serper 寫入 raw
  3. Extractor (LLM × N)  每篇獨立 fresh context，抽 QUOTE / NUMBER / pending claim
  4. Registry (程式)      更新 source-registry、execution-log、gap-log

Extractor 受 deep_research.llm 裡的 rate_limiter 節流，避免 429。
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

# 深讀配額：每子問題、每輪
_QUOTA_PER_ROLE = {"advocate": 2, "critic": 2, "perspective": 1}

# 抓取全文截斷上限（避免單篇破萬 token；Extractor 看單篇 ≈ 6-8K token）
# 從 45000 下修：原本一篇 45K chars ≈ 15K tokens，整篇塞 LLM 容易踩 Lost in the Middle，
# 中段 quote/number 被忽略 → 鐵律 4「溯源鏈完整」斷裂。改為超過 _CHUNK_SIZE 走 chunked。
_RAW_CHAR_LIMIT = 25000

# Taiwan 權威域名白名單：source_tier.py 使用，命中者自動升 T3
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

# Chunked extraction：sliding window 切片，每 chunk 獨立 LLM call 並發抽取後合併去重
_CHUNK_SIZE = 8000
_CHUNK_OVERLAP = 1000


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

async def phase1a_search(state: ResearchState) -> dict:
    """Phase 1a 入口：規劃 → 搜尋 → 深讀 → 抽取。"""
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

    # gap_log 會隨輪數累積（第 5 輪時已累積前 4 輪的所有 UNREACHABLE / MISSING / CONFLICT）。
    # 對 Planner 而言只有「最近發生的 gap」是有效訊號，前段老資訊是 distractor。
    # 保留最後 ~2000 chars（~650 tokens），避免 context 放大器效應。
    if len(gap_log) > 2000:
        gap_log = "...[前段省略，只保留最近 gap]...\n\n" + gap_log[-2000:]
    # plan 若超過 8000 chars（罕見但偶發），保留頭尾避免 Planner 看不完。
    if len(plan) > 8000:
        plan = plan[:4000] + "\n\n...[中段省略]...\n\n" + plan[-4000:]

    # 累積編號用：跨輪的現有 sources / claims
    existing_sources = state.get("sources", [])
    existing_claims = state.get("claims", [])

    # 確保 source-registry 存在
    if not read_workspace_file(workspace, "source-registry.md"):
        init_source_registry(workspace)

    remaining = budget - used
    if remaining <= 0:
        return {
            "execution_log": [f"Phase 1a 第 {iteration + 1} 輪：預算用盡，跳過"],
        }

    # Focus mode：trigger_fallback_node 指定需補搜的 SQ
    needs_refetch: list[str] = state.get("needs_refetch", [])
    focus_mode = bool(needs_refetch)
    if focus_mode:
        # 預算守衛回跳（large needs_refetch）vs 品質失敗回跳（small needs_refetch）
        # - 品質失敗（≤ 5 SQ）→ 聚焦補搜，預算上限 25
        # - 預算守衛（> 5 SQ）→ 全面繼續搜尋，預算上限為 remaining 的一半（至少 40）
        if len(needs_refetch) > 5:
            budget_cap = max(40, remaining // 2)
            remaining = min(remaining, budget_cap)
            logger.info(f"Phase 1a 預算守衛補搜：SQ={len(needs_refetch)} 個, budget={remaining}")
        else:
            remaining = min(remaining, 25)
            logger.info(f"Phase 1a 聚焦補搜：SQ={needs_refetch}, budget={remaining}")

    # ── Stage 1: Planner ─────────────────────────────────────────────
    already_searched = _extract_searched_queries(exec_log)

    # Budget 守衛：計算每 SQ 已用 query 數，找出未達 min 的 SQ
    min_per_sq = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["deep"])["min_budget_per_sq"]
    sq_ids = _extract_sq_ids(plan)
    sq_counts = _count_queries_per_sq(exec_log)
    underfunded_sqs = [sq for sq in sq_ids if sq_counts.get(sq, 0) < min_per_sq]

    # 發現面 query：當前年份 + 已知工具 seed
    current_year = str(datetime.now().year)
    known_tools = _extract_known_tools(plan)

    # 從上輪 gap-log 解析新發現實體（僅在 iteration > 0 且非 focus mode 時生效）
    prior_emerging = _extract_emerging_from_gap_log(gap_log, iteration) if iteration > 0 and not focus_mode else []

    query_plan = await _plan_queries(
        plan=plan,
        coverage=coverage,
        gap_log=gap_log,
        iteration=iteration,
        remaining_budget=remaining,
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

    queries = query_plan.get("queries", [])
    if not queries:
        return {
            "execution_log": [f"Phase 1a 第 {iteration + 1} 輪：Planner 未產出 query，跳過"],
        }

    # ── Stage 2: Executor ────────────────────────────────────────────
    search_hits, searches_used = await _execute_searches(queries, remaining)
    url_health = await _verify_urls(search_hits)
    selected = _select_urls_by_quota(search_hits, url_health, queries)

    # 跨輪 URL 去重：跳過已在前幾輪抓過的 URL，避免浪費預算 + 產生重複 claim
    prior_fetched: set[str] = set(state.get("fetched_urls", []))
    if prior_fetched:
        before = len(selected)
        selected = [s for s in selected if s["url"] not in prior_fetched]
        skipped = before - len(selected)
        if skipped:
            logger.info("phase1a: 跨輪去重跳過 %d 個已抓 URL", skipped)

    source_id_start = _next_source_id_index(existing_sources)
    raw_sources = await _fetch_pages(selected, workspace, source_id_start)

    # ── Stage 3: Extractor（每篇獨立 LLM，fresh context）──────────────
    extractions = await _extract_all_sources(raw_sources, workspace)

    # ── Stage 4: Registry ────────────────────────────────────────────
    _update_source_registry(workspace, raw_sources)
    _append_execution_log(workspace, iteration, queries, searches_used)
    _log_unreachable(workspace, url_health, raw_sources)
    _log_domain_bias(workspace, iteration, existing_sources, raw_sources)

    # Budget 守衛：更新本輪 sq_counts，把仍不足的 SQ 寫入 gap-log
    updated_sq_counts = dict(sq_counts)
    for q in queries:
        sq = q["subquestion"]
        updated_sq_counts[sq] = updated_sq_counts.get(sq, 0) + 1
    _log_budget_gaps(workspace, iteration + 1, sq_ids, updated_sq_counts, min_per_sq)

    # Iterative expansion：從本輪 search results 抽出新工具名，寫入 gap-log
    # 供下輪 Planner 生成 B 類 follow-up query（不在 focus mode 時才執行，避免分散注意）
    emerging_entities: list[str] = []
    if not focus_mode:
        emerging_entities = await _extract_emerging_entities(raw_sources, plan, iteration)
        if emerging_entities:
            entities_text = "\n".join(f"- {e}" for e in emerging_entities)
            append_workspace_file(
                workspace,
                "gap-log.md",
                f"\n\n## 新發現實體（第 {iteration + 1} 輪）\n{entities_text}\n",
            )

    sources = _build_sources(raw_sources)
    claims = _collect_claims(extractions, existing_claims)

    # 本輪新抓到的 URL（LIVE/THIN_CONTENT 都算，UNREACHABLE 不加入防止下輪仍被跳過）
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
            f"Phase 1a 第 {iteration + 1} 輪："
            f"搜尋 {searches_used} 次、深讀 {len(raw_sources)} 篇、"
            f"抽出 {len(claims)} claims（累計 {used + searches_used}/{budget}）"
            + (f"、新發現實體 {len(emerging_entities)} 個" if emerging_entities else "")
        ],
    }


# ---------------------------------------------------------------------------
# Stage 1: Planner — 產出 query 清單
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """你是研究搜尋 Planner。根據研究計畫和當前 coverage，產出下一輪要執行的搜尋 query 清單。

## 職責
- 只規劃 query，不執行搜尋。
- 漸進式：第一輪每個子問題只生成最小集（advocate 1 family + critic 1 family）。
- 後續輪次依 coverage gap 增補。

## Query 規則
1. 每個 query 5-10 詞。
2. advocate 和 critic 的 query 必須有明顯差異（不是同一 query 換措辭）。
3. 同一 query family 通常搭配兩個語言版本：en + zh-TW。zh-TW query **必須只**使用 `engines: ["serper_tw"]`，不可加 brave 或 serper_en（避免繁中結果被英文索引稀釋）。
4. 對照「已搜過的 query 清單」做語義去重，不要重複。
5. 學術主題可加上 `site:arxiv.org` 或 `site:semanticscholar.org`。

## 搜尋引擎配置（每個 query 指定 engines 列表）
- `brave`：英文獨立索引
- `serper_en`：Google 英文
- `serper_tw`：Google 繁體中文（zh-TW query 專用，不可與 brave/serper_en 混用）
- `serper_cn`：Google 中國
- `serper_scholar`：Google + site:arxiv.org/semanticscholar.org

## 輸出（嚴格 JSON，不要其他文字）
```json
{
  "queries": [
    {"subquestion": "Q1", "role": "advocate", "query": "AI transcription accuracy Mandarin", "lang": "en", "engines": ["brave", "serper_en"]},
    {"subquestion": "Q1", "role": "advocate", "query": "中文語音轉文字 準確率", "lang": "zh-TW", "engines": ["serper_tw"]},
    {"subquestion": "Q1", "role": "critic", "query": "AI transcription errors limitations", "lang": "en", "engines": ["brave", "serper_en"]}
  ]
}
```

## Discovery Query Family（第 1 輪必須包含）
除 advocate / critic 外，每個子問題至少需要 **2 個發現面 query**：

| 類型 | 模板 | engines |
|------|------|---------|
| A 最新工具 | `best {主題} {YEAR}` / `top {主題} tools {YEAR}` | brave, serper_en |
| B 競品 | `alternative to {已知工具名} {YEAR}` / `{已知工具名} competitors {YEAR}` | brave, serper_en |
| C 在地 | `{主題} 台灣 推薦` / `site:ithome.com.tw {主題}` | serper_tw（只用此 engine） |
| D 社群 | `site:reddit.com {主題} {YEAR} recommendation` | serper_en |

規則：
- 每個子問題至少涵蓋 A 類 + C 類；B 類和 D 類視 budget 加入。
- C 類 query 必須**只**使用 `engines: ["serper_tw"]`，不可加 brave 或 serper_en。
- 後續輪次如果 Gap Log 出現新工具名，也必須為其補一個 B 類 query。
- user_msg 有「新發現實體」欄位時，**每個實體至少產出 1 個 B 類 follow-up query**（例如 `{實體名} review {YEAR}` 或 `alternative to {實體名}`）。
- {YEAR} 請使用 user_msg 中「本輪年份」提供的數字。

## 台灣來源鎖定
研究涉及 Taiwan/台灣相關主題時，優先使用以下 site: 前綴產出精準繁中 query：
- `site:ithome.com.tw`、`site:ithelp.ithome.com.tw` — IT 媒體
- `site:techbang.com`、`site:kocpc.com.tw` — 電腦/3C 評測
- `site:mobile01.com` — 3C 討論區
- `site:inside.com.tw`、`site:bnext.com.tw` — 數位創業媒體
- `site:apps.apple.com` — App Store（含評分與評論）

這類 query 一律搭配 `engines: ["serper_tw"]`，不要加其他 engine。

## 預算控制
- 每個 query 消耗 1 次搜尋（多 engines 對同一 query 視為同一次）。
- 不得超過剩餘預算。
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
    """Stage 1: LLM 規劃下一輪要執行的 query 清單。"""
    searched_text = (
        "\n".join(f"- {q}" for q in already_searched) if already_searched else "（尚無）"
    )

    # Budget 守衛：若有 SQ 未達 min，注入優先補指示
    if underfunded_sqs:
        sq_counts = sq_counts or {}
        underfunded_text = "\n".join(
            f"  - {sq}：已搜 {sq_counts.get(sq, 0)}/{min_per_sq} 次"
            for sq in underfunded_sqs
        )
        sq_priority_section = f"""
## ⚠️ 子問題預算不足（本輪必須優先補）
以下子問題的搜尋 query 數尚未達到最低要求（{min_per_sq} 次），本輪**必須**優先為這些子問題產出 query：
{underfunded_text}

規則：在這些子問題各自達到 {min_per_sq} 次之前，不得為已達標的子問題產出額外 query。
"""
    else:
        sq_priority_section = ""

    # 聚焦補搜模式（由 trigger_fallback_node 觸發）
    if focus_sqs:
        focus_section = f"""
## 🎯 聚焦補搜模式（Fallback）
**本輪只為以下 SQ 補搜，其他子問題暫停：**
{', '.join(focus_sqs)}

補搜要求：
- 每個指定 SQ 至少：advocate 1 組 + critic 1 組 + Discovery C 類（serper_tw）1 組
- 優先台灣本土資源（serper_tw）和學術/官方資源（T1/T2 tier）
- 對照 Gap Log 中的失敗原因，針對性補充
- 總 query 數不超過 {remaining_budget} 次
"""
    else:
        focus_section = ""

    # 已知工具 seed（用於 Discovery Query B 類）
    if known_tools:
        tools_text = "、".join(known_tools[:15])
        known_tools_section = f"\n## 已知工具名（用於生成 Discovery Query B 類）\n{tools_text}\n"
    else:
        known_tools_section = ""

    # 當前年份（用於 Discovery Query 的 {YEAR} 佔位）
    year_section = f"\n## 本輪年份\n- 當前年份：{current_year or '2026'}（query 中的 year 請使用此數字）\n" if current_year else ""

    # 新發現實體（由 _extract_emerging_entities 從上輪搜尋結果中抽出）
    if emerging_entities:
        entities_text = "\n".join(f"- {e}" for e in emerging_entities)
        emerging_section = f"\n## 新發現實體（本輪從搜尋結果中發現，須補充 query）\n{entities_text}\n"
    else:
        emerging_section = ""

    user_msg = f"""## 研究計畫
{plan}

## Coverage Checklist
{coverage}

## Gap Log
{gap_log}

## 已搜過的 Query（避免重複）
{searched_text}
{focus_section}{sq_priority_section}{year_section}{known_tools_section}{emerging_section}
## 本輪限制
- 這是第 {iteration + 1} 輪（第 1 輪請用最小集；後續輪次依 coverage gap 增補）
- 剩餘搜尋預算：{remaining_budget}
- 研究深度：{depth}

請產出本輪要執行的 query 清單。"""

    # role="verifier" — query 規劃是 structured JSON generation（邏輯任務）
    # 不需要 Opus 的創意寫作能力，改用 Gemini/GPT-fast 節省 5-10 分鐘
    # max_tokens=8192：給 JSON output 足夠空間（thinking_budget=0 已在 llm.py 設定）
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

    # 規範化：清掉缺欄位或重複的 query
    seen: set[str] = set()
    clean: list[dict] = []
    for q in data.get("queries", []):
        query = (q.get("query") or "").strip()
        if not query or query in seen:
            continue
        seen.add(query)
        # Normalize subquestion to Q{n} format — LLM might return full title like
        # "子問題 1: 主流工具盤點" or "1" instead of "Q1".
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
    """從 execution-log.md 撈出所有已搜尋過的 query 文字。

    匹配 `- {query} [Q.../role/lang]` 格式（_append_execution_log 寫入的格式）。
    """
    if not exec_log:
        return []
    # 抓出 `- {query} [Q.../role/lang]` — query 是 [...] 之前的部分
    pattern = re.compile(r"^[-*]\s+(.+?)\s+\[[^\]]+\]\s*$", re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(exec_log)]


def _extract_sq_ids(plan: str) -> list[str]:
    """從 phase0-plan.md 抽出有序不重複的子問題 ID（Q1, Q2, ...）。"""
    matches = re.findall(r'\b(Q\d+)\b', plan)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _count_queries_per_sq(exec_log: str) -> dict[str, int]:
    """從 execution-log.md 統計每個 SQ 已用 query 次數。

    匹配 `- {query} [Q{n}/role/lang]` 格式，取方括號內 Q 號。
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
    """把預算仍不足的 SQ 追加到 gap-log.md 的「預算缺口」段落。

    在每輪結束後呼叫，讓下一輪 Planner 能透過 gap_log 得知需要優先補的 SQ。
    """
    gaps = [
        (sq, sq_counts.get(sq, 0))
        for sq in sq_ids
        if sq_counts.get(sq, 0) < min_per_sq
    ]
    if not gaps:
        return
    lines = [f"\n\n## 預算缺口（第 {iteration} 輪後）"]
    for sq, have in gaps:
        need = min_per_sq - have
        lines.append(f"- {sq}：已搜 {have} 次，最低要求 {min_per_sq} 次，仍缺 {need} 次")
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _extract_known_tools(plan: str) -> list[str]:
    """從 research plan 文字抽出已知工具/服務名稱，供 planner 生成 Discovery Query 用。

    從括號（中英文）裡找以頓號或逗號分隔的工具清單，過濾出含英文字母的詞。
    不用空格分割，以保留「Clova Note」「Good Tape」這類多詞工具名。
    """
    bracket_content = re.findall(r'[（(]([^)）\n]{3,120})[)）]', plan)
    candidates: list[str] = []
    for content in bracket_content:
        # 只用頓號/逗號分割，保留工具名中的空格
        parts = re.split(r'[、，,]+', content)
        for p in parts:
            p = p.strip()
            # 去掉結尾的「等」和開頭的助詞（「如」「例如」「含」等）
            if p.endswith('等'):
                p = p[:-1].strip()
            p = re.sub(r'^(如|例如|含|包含|及|或)\s+', '', p).strip()
            # 保留 2-40 字、含英文字母的詞（工具名通常是英文或中英混合）
            if 1 < len(p) <= 40 and re.search(r'[A-Za-z]', p):
                candidates.append(p)
    # 去重保持出現順序，最多回傳 20 個
    seen: set[str] = set()
    result: list[str] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= 20:
            break
    return result


async def _extract_emerging_entities(
    raw_sources: list[dict],
    plan: str,
    iteration: int,
) -> list[str]:
    """從本輪 search results 抽取 plan 中尚未出現的新工具/產品/服務名稱。

    以 LLM（role="verifier"）從各來源的 title + content 片段提取工具名，
    排除 plan 已知的，回傳去重後最多 15 個。
    """
    if not raw_sources:
        return []

    # 收集 title + 前 500 字 content（夠 LLM 識別工具名，避免 context 膨脹）
    snippets: list[str] = []
    for s in raw_sources:
        title = s.get("title", "")
        content = s.get("content", "") or ""
        if title or content:
            snippets.append(f"[{s.get('source_id', '?')}] {title}\n{content[:500]}")
    if not snippets:
        return []

    combined = "\n\n---\n\n".join(snippets[:30])  # 最多 30 篇

    # plan 裡已知的工具（排除清單）
    known_in_plan = _extract_known_tools(plan)
    known_text = "、".join(known_in_plan) if known_in_plan else "（無）"

    prompt = f"""以下是最新一輪搜尋結果的標題和摘要。

任務：找出文中提到的工具/軟體/App/服務/硬體產品名稱，**排除**以下研究計畫中已知的工具：
{known_text}

輸出格式：每行一個名稱，最多 15 個。只輸出名稱，不要解釋。

搜尋結果：
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

    # 解析輸出：每行一個，過濾無效行
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

    logger.info(f"第 {iteration + 1} 輪新發現實體：{entities}")
    return entities


def _extract_emerging_from_gap_log(gap_log: str, iteration: int) -> list[str]:
    """從 gap-log.md 中解析最近一輪寫入的「新發現實體」清單。

    只取最後一個 ## 新發現實體（第 N 輪）段落，避免把舊輪實體重複送給 Planner。
    """
    # 找所有「## 新發現實體（第 N 輪）」段落起始位置
    pattern = re.compile(r"## 新發現實體（第 \d+ 輪）\n(.*?)(?=\n## |\Z)", re.DOTALL)
    matches = list(pattern.finditer(gap_log))
    if not matches:
        return []

    # 取最後一個段落
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
# Stage 2: Executor — 搜尋 + urlhealth + fetch raw
# ---------------------------------------------------------------------------

async def _execute_searches(
    queries: list[dict],
    remaining_budget: int,
) -> tuple[list[dict], int]:
    """平行執行所有搜尋 task。回傳 (hits, searches_used)。

    hits: [{"subquestion", "role", "query", "lang", "engine", "url", "title", "description"}, ...]
    每個 (query, engine) 組合消耗 1 次搜尋，但 engines 數量受 remaining_budget 限制。
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
    """依引擎分派到對應的底層搜尋函數。"""
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
            # 把 query 前綴 site:arxiv.org OR site:semanticscholar.org
            scholar_q = f"({query}) (site:arxiv.org OR site:semanticscholar.org)"
            return await serper_search(scholar_q, gl="us", hl="en", num=10)
    except Exception:
        return []
    return []


async def _verify_urls(hits: list[dict]) -> dict[str, str]:
    """對所有命中 URL 批次呼叫 urlhealth CLI，回傳 {url: status}。"""
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
        # 無法驗證時降級為 UNKNOWN；後續 fetch 仍會嘗試，失敗會記 UNREACHABLE
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
    """按 (subquestion, role) 配額挑選要深讀的 URL。

    規則：
      - 跳過 LIKELY_HALLUCINATED
      - 同一 URL 如果命中多個 engine，跨引擎加分
      - 每個 (subq, role) 至多挑 _QUOTA_PER_ROLE[role] 篇
    """
    # Step 1: 聚合——同 URL 合併各 engine 命中
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

    # Step 2: 每個 (subq, role) 桶獨立排序 + 取配額
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
    """三階梯抓取：WebFetch → Serper scrape → 標記 UNREACHABLE。

    每篇寫入 workspace/search-results/{subq}/{source_id}_raw.md
    回傳每篇的 meta dict（含 source_id, url, raw_path, content, status...）
    `id_start` 用於跨輪累積編號，避免第二輪重新從 S001 撞號。
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
        # THIN_CONTENT 覆寫 url_health status，讓下游可以識別並過濾
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
    """三階梯抓取單一 URL。回傳 (content, method)。

    method 取值：
      "web_fetch"      — WebFetch 成功且 >= 500 chars
      "serper_scrape"  — Serper scrape 成功且 >= 500 chars
      "thin_content"   — 至少有回傳但 < 500 chars（頁面截斷或無實質內容）
      "unreachable"    — 完全無法取得（content 為空字串）

    標題結尾為 …/... 暗示 WebFetch 只抓到截斷版，一律嘗試 Serper scrape。
    """
    url = item["url"]
    title = item.get("title", "")
    thin_candidate = ""  # 記下最長的 < 500 chars 內容作 fallback

    # 1. WebFetch
    try:
        text = await web_fetch(url)
        if text:
            if len(text.strip()) >= 500:
                return text, "web_fetch"
            elif text.strip():
                thin_candidate = text   # 有回傳但太短
    except Exception:
        pass

    # 2. Serper scrape
    # 條件：尚未成功取得足量內容，或標題截斷（暗示 WebFetch 只拿到 stub）
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
# Stage 3: Extractor — 每篇獨立 LLM，fresh context
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM = """你是精準抄錄員。從一篇來源原文中抽出與子問題相關的證據。

## 任務
對這一篇原文：
1. 找出與子問題相關的段落（最多 3 處）
2. 逐字複製關鍵句為 QUOTE（不得改寫、摘要、合併）
3. 含數字的句子單獨記為 NUMBER（必須連同完整原句）
4. 基於這些 QUOTE/NUMBER 形成 1-3 個 pending claim

## 鐵律
- QUOTE 必須是原文逐字，禁止改寫
- **QUOTE 和 NUMBER 的 text/sentence 欄位必須保持來源的原始語言。若來源是英文，quote text 必須是英文；若來源是中文，quote text 必須是中文。嚴禁翻譯。**
- claim_text 可以用中文撰寫，但 QUOTE/NUMBER 的 text/sentence 欄位必須保持原文語言，不可翻譯
- NUMBER 必須附含數字的完整原句（保持原文語言）
- 每個 claim 必須連結至少 1 個 quote_id 或 number_id
- 無明確證據的推論禁止建成 claim

## 禁止抽取的內容（常見雜訊，一律跳過）
以下內容與研究子問題無關，即使出現在原文中也絕對不得抽為 claim：
- 公司/機構的實體地址、郵遞區號、門牌號碼
- 聯絡電話、客服 email、客服連結
- 公司員工人數、辦公室城市/國家等公司簡介資訊
- 公司創辦年份、創辦人姓名（與產品功能、性能無關的背景資訊）
- Cookie 聲明、隱私政策、使用者條款、廣告文字
- 網站 header / footer / navigation 樣板文字
- SEO 行銷語句（如 "Our team of experts"、"We are dedicated to..."、"Contact us today"）

## start_char / end_char 欄位（重要）
對每個 QUOTE / NUMBER，額外輸出 start_char 與 end_char，
代表該段文字在「上面的原文」中的字元起訖 index（0-based，Python 字串切片語意）。
驗證方式：raw_content[start_char:end_char] 會等於 text（或 sentence）。

估不準不會被懲罰 — 程式會用 text 去回找真位置做 fallback。
但 text 欄位必須仍是原文的逐字摘錄，否則整筆會被 reject。

若 text 在原文出現多次，任選一處的 index 即可（程式會認第一個能 find 到的位置）。

## 輸出（嚴格 JSON）
```json
{
  "quotes": [
    {"quote_id": "Q1", "text": "原文逐字複製的句子", "start_char": 123, "end_char": 234}
  ],
  "numbers": [
    {"number_id": "N1", "value": "92.5", "unit": "%",
     "sentence": "含該數字的完整原句", "start_char": 456, "end_char": 567}
  ],
  "claims": [
    {
      "claim_text": "一句話陳述事實",
      "claim_type": "numeric|comparative|causal|forecast|qualitative",
      "evidence_quote_ids": ["Q1"],
      "evidence_number_ids": ["N1"]
    }
  ]
}
```

只輸出 JSON，不要其他文字。"""


# Index-based 引用共用工具：resolve_quote_index / verify_indexed_items 現在住在
# deep_research.harness.validators，這裡保留本模組內部別名方便閱讀。
_resolve_quote_index = resolve_quote_index


def _verify_indexed_items(raw, items, text_field, chunk_offset=0):
    return verify_indexed_items(
        raw, items, text_field, chunk_offset=chunk_offset, log_prefix="Tier1/index"
    )


async def _extract_all_sources(raw_sources: list[dict], workspace: str) -> list[dict]:
    """對每篇獨立 LLM call 抽取。rate_limiter 已在 llm.py 配置。

    THIN_CONTENT source 跳過（不進 claim extraction）。
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
    """對單篇來源執行 Extractor，並寫入最終的 S{id}.md。

    短文（≤ _CHUNK_SIZE）走單次 LLM；長文做 sliding window chunked extraction：
    每 chunk 獨立 LLM call 並發抽取，合併去重後重新編號。

    為什麼要切：原本一篇 25K chars 整塊餵 LLM，中段 quote/number 容易因 Lost in the
    Middle 被忽略，破壞鐵律 4「溯源鏈完整」。chunked + overlap 讓每段都有近端 context。
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
        # 短文：LLM 看到的就是完整 raw，index 為 global。單次驗證即可。
        data = {
            "quotes": _verify_indexed_items(raw, data_raw.get("quotes", []), "text"),
            "numbers": _verify_indexed_items(raw, data_raw.get("numbers", []), "sentence"),
            "claims": data_raw.get("claims", []),
        }
    else:
        data = await _extract_chunked(src, raw)

    if not data:
        return None

    # 重編 id：保證單篇 source 內 Q1, Q2, ... 連續且唯一（chunked 模式合併後尤其重要）
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

    # ─── Tier 1 硬規則驗證 ────────────────────────────────────────
    # 鐵律 2 + 4：quote 必須真實存在於原文 + claim 引用 quote_id 必須真實存在
    # 違規即 drop（不傳給下游被誤用），記入 logger.warning 方便 debug。
    quotes, numbers, claims = _apply_tier1_validation(
        sid=sid,
        raw_content=raw,
        quotes=quotes,
        numbers=numbers,
        claims=claims,
    )

    # 寫入最終 S{id}.md（phase1b grounding 要讀這個）
    _write_source_file(workspace, src, quotes, numbers)

    return {"quotes": quotes, "numbers": numbers, "claims": claims}


def _apply_tier1_validation(
    sid: str,
    raw_content: str,
    quotes: list[dict],
    numbers: list[dict],
    claims: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Tier 1 硬規則：drop 在原文找不到的 quote、ledger 不存在的 quote_id。

    流程：
      1. validate_quotes_indexed：drop quote 的 start/end 無效或 source[s:e]!=text 的
         （鐵律 2 硬版）。_verify_indexed_items 早前已 drop 過無法定位者，這裡是
         雙重保險 — 確保 phase1a 下游看到的每筆 quote 都能純 index 還原。
      2. number 同樣驗證（使用 sentence/start/end）
         舊 quote 格式（無 start/end）會在步驟 1/2 被標成 violation；
         若需向後相容（e.g. 不走 phase1a 的外部 quote），可再走 validate_quotes_exist。
      3. validate_quote_ids_in_ledger：claim 引用不存在的 quote_id 視為破壞鐵律 4
         — 把該無效 quote_id 從 claim 中移除；若 claim 失去所有 quote_ids → drop claim
    """
    if not raw_content:
        # 無原文（e.g. 抓取失敗的 source）— Tier 1 直接放行，後續 phase 自會處理
        return quotes, numbers, claims

    # 1. quotes — 純 index 驗證（最硬）
    quote_violations = validate_quotes_indexed(quotes, raw_content)
    bad_quote_ids: set[str] = set()
    if quote_violations:
        for v in quote_violations:
            logger.warning("[%s][Tier1/indexed] %s", sid, v)
            bad_id = v.split(":", 1)[0].strip()
            bad_quote_ids.add(bad_id)
        quotes = [q for q in quotes if q.get("quote_id") not in bad_quote_ids]

    # 2. numbers — 同樣走 index 驗證（sentence 當 text_field）
    num_violations = validate_quotes_indexed(numbers, raw_content)
    bad_number_ids: set[str] = set()
    if num_violations:
        for v in num_violations:
            logger.warning("[%s][Tier1/indexed] number %s", sid, v)
            bad_id = v.split(":", 1)[0].strip()
            bad_number_ids.add(bad_id)
        numbers = [n for n in numbers if n.get("number_id") not in bad_number_ids]

    # 3. claims：清掉引用無效 quote_id 的部分；空 quote_ids 的 claim 整條 drop
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
    """單次 LLM call：抽取一段 content（可能是整篇或一個 chunk）的 quote/number/claim。"""
    chunk_note = ""
    if chunk_idx is not None and chunk_total is not None:
        chunk_note = (
            f"\n\n（這是長文的第 {chunk_idx + 1}/{chunk_total} 段，"
            f"請只就這段內容抽取，不要假設前後文。）"
        )

    user_msg = f"""## 子問題
{src['subquestion']}（角色：{src['role']}）

## 來源 metadata
- source_id: {src['source_id']}
- url: {src['url']}
- title: {src['title']}{chunk_note}

## 原文
{content}

請根據以上原文抽取 QUOTE / NUMBER / claim。"""

    try:
        # role="verifier" — 抽取/核對類，Gemini 主導（grounded summarization 幻覺率最低）
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
    """長文 sliding window 抽取：切 chunks → 並發 LLM call → 合併去重。

    每 chunk 內的 quote_id / number_id 加上 c{chunk_idx}- 前綴避免衝突，
    外層 _extract_one 會再做最終 ID 重編。

    Index 處理：
      - LLM 看到的是單一 chunk，輸出的 start_char/end_char 為 chunk-local
      - _verify_indexed_items(chunk_content, items, ..., chunk_offset=chunk_global_pos)
        會把 chunk-local 驗證後的 index 轉為整篇 raw 的 global 座標
      - 去重以 (start, end) tuple 為 key — 比 text 內容去重更精確
        （同一段在 chunk overlap 區被兩次抽到會有相同 global span）

    Claim 去重仍用 claim_text（claim 不一定有 span）。
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
    # 升級 claim 去重：用 normalize_for_dedup key 代替純字串
    # key = normalized text → 抓得到 punctuation/whitespace 差異的近重複
    seen_claim_norm: set[str] = set()
    # 同時保留原始文字供 is_near_duplicate 比對（處理 normalized 後仍不同的近重複）
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
            # 1) 快速 normalized 完全相符
            if norm in seen_claim_norm:
                continue
            # 2) SequenceMatcher 近似重複（chunk overlap 區常見輕微改寫）
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
        "<!-- 每筆 @[start:end] 為 TextSpan（對應截斷後的 raw_content，"
        "0-based、end-exclusive）。下游可用 raw[start:end] 切片驗證 / render。 -->",
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
    """將 start/end index 格式化為 @[s:e]；任一不是 int → 空字串（向後相容）。"""
    if isinstance(start, int) and isinstance(end, int):
        return f" @[{start}:{end}]"
    return ""


# ---------------------------------------------------------------------------
# Stage 4: Registry
# ---------------------------------------------------------------------------

def _update_source_registry(workspace: str, raw_sources: list[dict]) -> None:
    """追加新來源到 source-registry.md。"""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    for s in raw_sources:
        if not s["content"] and s.get("status") != "THIN_CONTENT":
            continue  # UNREACHABLE 不登記到主 registry；THIN_CONTENT 留著透明記錄
        engines = ",".join(s["engines"])
        # THIN_CONTENT 強制 T6；其他依 domain 分級
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
    """追加本輪到 execution-log.md，更新已搜 query 清單和搜尋計數。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = [
        "",
        f"### 第 {iteration + 1} 輪 [{ts}]（消耗 {searches_used} 次）",
    ]
    # 追加到已搜 Query 清單
    query_lines = [f"- {q['query']} [{q['subquestion']}/{q['role']}/{q['lang']}]" for q in queries]
    block.append("")
    append_workspace_file(workspace, "execution-log.md", "\n".join(block) + "\n" + "\n".join(query_lines) + "\n")


def _log_unreachable(
    workspace: str,
    url_health: dict[str, str],
    raw_sources: list[dict],
) -> None:
    """把 UNREACHABLE / THIN_CONTENT / LIKELY_HALLUCINATED URL 記入 gap-log.md。"""
    bad = [s for s in raw_sources if not s["content"]]
    thin = [s for s in raw_sources if s.get("status") == "THIN_CONTENT"]
    hallucinated = [u for u, st in url_health.items() if st == "LIKELY_HALLUCINATED"]
    if not bad and not thin and not hallucinated:
        return
    lines = [""]
    if bad:
        lines.append("### UNREACHABLE URLs（三階梯抓取全失敗）")
        for s in bad:
            lines.append(f"- {s['url']} [{s['subquestion']}/{s['role']}]")
    if thin:
        lines.append("### THIN_CONTENT URLs（內容不足 500 chars，不進 claim extraction）")
        for s in thin:
            chars = len((s.get("content") or "").strip())
            lines.append(f"- {s['url']} [{s['subquestion']}/{s['role']}] ({chars} chars)")
    if hallucinated:
        lines.append("### LIKELY_HALLUCINATED URLs（urlhealth 判定）")
        for u in hallucinated:
            lines.append(f"- {u}")
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


def _extract_domain(url: str) -> str:
    """從 URL 提取 hostname，去掉 www. 前綴。"""
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
    """掃描所有累積 sources 的 domain 分佈。

    若某 domain 佔比超過 threshold（預設 30%），在 gap-log.md 追加
    [BIAS WARNING]，供 Phase 2 整合時降低該 domain claims 的信心等級。
    """
    domain_counts: dict[str, int] = {}

    # 已存在的 Source 物件（前幾輪）
    for s in existing_sources:
        url = s.url if hasattr(s, "url") else (s.get("url", "") if isinstance(s, dict) else "")
        d = _extract_domain(url)
        if d:
            domain_counts[d] = domain_counts.get(d, 0) + 1

    # 本輪新抓到的（raw dicts）
    for s in new_raw_sources:
        url = s.get("url", "")
        if s.get("status") == "UNREACHABLE":
            continue  # 未抓到的不算
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
        f"\n\n## [BIAS WARNING] 來源 domain 濃度過高（第 {iteration + 1} 輪累計）",
        f"以下 domain 佔所有已抓來源 > {threshold:.0%}，可能存在自家宣傳或觀點偏頗：",
    ]
    for domain, count, pct in biased:
        lines.append(f"- **{domain}**：{count}/{total} 篇（{pct:.0%}）")
    lines.append(
        "Phase 2 整合時，來自這些 domain 的 claims 應標記 🟠CONFLICTING，"
        "除非有 T1-T3 獨立來源佐證。"
    )
    append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 建立回傳的 sources / claims
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
    """從現有 sources 算下一個 S{n} 編號起點。"""
    max_n = 0
    for s in existing:
        sid = s.source_id if hasattr(s, "source_id") else s.get("source_id", "")
        m = re.match(r"S(\d+)", sid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _collect_claims(extractions: list[dict], existing_claims: list | None = None) -> list[Claim]:
    """展開所有 Extractor 產出的 pending claim 成 Claim object。

    `existing_claims` 用於跨輪累積編號，避免第二輪重新從 Q1-C1 撞號。
    同時以 is_near_duplicate 過濾與 existing_claims 近似重複的 claim（跨輪去重）。
    """
    out: list[Claim] = []
    counter: dict[str, int] = {}

    # 建立跨輪去重的快速查詢結構：每個 subquestion → 已有 claim 文字 list
    # 同時記 normalized key set 做 O(1) 快速篩
    existing_raw: dict[str, list[str]] = {}   # subq → [raw_texts]
    existing_norm: dict[str, set[str]] = {}   # subq → {norm_texts}

    if existing_claims:
        for c in existing_claims:
            cid = c.claim_id if hasattr(c, "claim_id") else c.get("claim_id", "")
            sq = c.subquestion if hasattr(c, "subquestion") else c.get("subquestion", "")
            # Match any claim_id format ending in -C{n}: "Q1-C3", "子問題1-C3", etc.
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

            # 跨輪去重：先用 normalized set O(1) 比，再用 SequenceMatcher
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

            # 把新 claim 也加入去重池，防止同批次不同 source 的重複
            existing_raw.setdefault(subq, []).append(claim_text)
            existing_norm.setdefault(subq, set()).add(norm)

    return out
