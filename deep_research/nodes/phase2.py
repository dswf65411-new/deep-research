"""Phase 2: Integration + Conflict Resolution.

Workflow node — reads approved claims, resolves contradictions,
writes report sections with confidence levels.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.claim_dedup import is_near_duplicate
from deep_research.harness.validators import validate_claims_for_phase2, validate_numeric_claims
from deep_research.harness.source_tier import tier_rank
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
from deep_research.state import Claim, ResearchState, Source
from deep_research.tools.workspace import (
    read_workspace_file,
    write_workspace_file,
)

from deep_research.config import get_prompt
from deep_research.context import iterative_refine

logger = logging.getLogger(__name__)

# 掃 section 文字找 claim_id 引用的 pattern（e.g. Q1-C1, Q12-C23）
_CLAIM_ID_RE = re.compile(r"\bQ\d+-C\d+\b")

_DOMAIN_BIAS_THRESHOLD = 0.30  # 同 phase1a._log_domain_bias


def _extract_domain(url: str) -> str:
    """從 URL 提取 hostname，去掉 www. 前綴。"""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.") if host else ""
    except Exception:
        return ""


def _detect_biased_domains(sources: list[Source]) -> set[str]:
    """計算所有 sources 的 domain 分佈，回傳佔比超過 30% 的 domain 集合。"""
    from collections import Counter
    domains = [
        _extract_domain(s.url)
        for s in sources
        if s.url and s.url_status != "UNREACHABLE"
    ]
    domains = [d for d in domains if d]
    if not domains:
        return set()
    total = len(domains)
    counts = Counter(domains)
    return {d for d, n in counts.items() if n / total > _DOMAIN_BIAS_THRESHOLD}


def _dedup_approved_claims(claims: list[Claim]) -> list[Claim]:
    """Remove near-duplicate approved claims within each subquestion.

    Keeps the first occurrence (by claim_id order) when two claims share
    similarity >= 0.92. Cross-subquestion dedup is intentionally skipped to
    avoid removing legitimately similar facts that belong to different SQs.
    """
    by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        by_subq.setdefault(c.subquestion, []).append(c)

    result: list[Claim] = []
    for subq_claims in by_subq.values():
        accepted: list[Claim] = []
        for claim in subq_claims:
            if any(is_near_duplicate(claim.claim_text, prev.claim_text) for prev in accepted):
                logger.debug(
                    "phase2 dedup: skipped near-duplicate claim %s", claim.claim_id
                )
                continue
            accepted.append(claim)
        result.extend(accepted)

    return result


def _scan_phantom_claim_ids(section: str, approved_ids: set[str]) -> list[str]:
    """Return claim_ids 出現在 section 文字但不在 approved_ids 中的清單。

    phase2 LLM 被要求只引用 approved claims。偶發 LLM 會幻覺 ID（拼錯 / 編造
    不存在的 ID）。phase3 audit 會抓到鏈斷裂，但越早抓住越省事。

    Note:
        phase2 本身不做 span-based 驗證：它是「寫作者」而非「抽取者」，
        原始文字引用已經在 phase1a 被 index 驗證過；section 文字會在 phase3
        被 verify_indexed_items 切成 statement 並再度 index 驗證。這裡只做
        輕量的 claim_id referential integrity 檢查，不碰文字內容本身。
    """
    found = set(_CLAIM_ID_RE.findall(section or ""))
    return sorted(found - approved_ids)


async def phase2_integrate(state: ResearchState) -> dict:
    """Integrate approved claims into report sections."""
    workspace = state["workspace_path"]
    claims = state.get("claims", [])

    # Clear stale report-sections from any previous pipeline run on this workspace.
    # Phase 3 reads ALL *.md files from report-sections/, so leftovers from a prior
    # run would contaminate the new report.
    import glob as _glob
    _old_sections = _glob.glob(str(Path(workspace) / "report-sections" / "*.md"))
    for _f in _old_sections:
        try:
            Path(_f).unlink()
            logger.info("phase2: 清除舊 section 檔案 %s", Path(_f).name)
        except OSError:
            pass

    # Convert to Claim objects
    claim_objects = []
    for c in claims:
        if isinstance(c, Claim):
            claim_objects.append(c)
        elif isinstance(c, dict):
            claim_objects.append(Claim(**c))

    # Iron rule: only approved claims with quote_ids
    approved = validate_claims_for_phase2(claim_objects)

    # Near-duplicate dedup：同一 subquestion 內相似度 >= 0.92 的 claim 只保留第一個
    # 防止跨輪補搜把同語意 claim 重複送進 LLM（distractor + token 浪費）
    approved = _dedup_approved_claims(approved)

    # Iron rule: numeric claims must have number_tag
    violations = validate_numeric_claims(approved)
    blockers = []
    if violations:
        blockers.extend(violations)

    # Read phase instructions
    instructions = get_prompt("phase2-integrate.md")

    # Read gap log（source-registry 不再塞入 LLM context — 是 distractor，
    # LLM 整合本子問題段落時不需要看其他子問題的 source 列表）
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""

    # 取 full_research_topic
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # Group claims by subquestion
    by_subq: dict[str, list[Claim]] = {}
    for c in approved:
        by_subq.setdefault(c.subquestion, []).append(c)

    # 收集每個 subquestion 的 source 文字（供 iterative_refine 分批處理）
    # 傳入 sources 讓 _gather_source_texts 按 tier 排序（T1-T3 優先）
    all_sources: list[Source] = [
        s if isinstance(s, Source) else Source(**s)
        for s in state.get("sources", [])
        if s
    ]
    subq_sources = _gather_source_texts(workspace, approved, sources=all_sources)

    # Domain 濃度偵測：找出佔比過高的 domain（同 phase1a 的 _log_domain_bias）
    biased_domains: set[str] = _detect_biased_domains(all_sources)
    # 建立 source_id → domain 映射，供 claim 標記使用
    sid_to_domain: dict[str, str] = {
        s.source_id: _extract_domain(s.url)
        for s in all_sources
        if s.url
    }

    # BLOCKER 清單（由 trigger_fallback_node 寫入）
    all_blockers: list[str] = state.get("blockers", [])

    async def _write_one_section(
        subq: str,
        subq_claims: list[Claim],
    ) -> tuple[str, str, list[str]]:
        """Write report section for one subquestion.

        Returns (subq, section_text, local_blockers).
        Runs concurrently with other sections via asyncio.gather.
        """
        local_blockers: list[str] = []

        def _claim_line(c: Claim) -> str:
            base = f"- {c.claim_id} [{c.claim_type}] (bedrock={c.bedrock_score:.2f}): {c.claim_text}"
            if biased_domains:
                claim_domains = {
                    sid_to_domain[sid]
                    for sid in c.source_ids
                    if sid in sid_to_domain and sid_to_domain[sid] in biased_domains
                }
                if claim_domains:
                    tag = ", ".join(sorted(claim_domains))
                    base += f"  [⚠️BIASED_SOURCE: {tag}]"
            return base

        claims_text = "\n".join(_claim_line(c) for c in subq_claims)

        integrate_system = f"""{instructions}

你是研究報告整合器。根據已驗證的 approved claims 和來源原文，生成報告段落。

## 鐵律
1. 只使用以下 approved claims，禁止引用其他資訊
2. **每句事實必須在句末附 claim_id 標記，格式：[Q1-C2]**
   - 範例：「Whisper Large v3 的中文 WER 為 8.3%。[Q1-C5]」
   - 範例：「macOS 14 以上版本才支援此功能。[Q2-C3]」
   - 如果一句引用多個 claim：「...。[Q1-C5] [Q1-C7]」
3. 跨 claim 推導必須標記 [INFERENCE] 並附相關 claim_id：
   - 範例：「[INFERENCE] 因此本地化工具在隱私上占優勢。[Q1-C5] [Q2-C1]」
4. 數字必須標記 ORIGINAL/NORMALIZED/DERIVED
5. 信心等級必須分配（🟢HIGH / 🟡MEDIUM / 🟠CONFLICTING / 🔴LOW）
6. **標題、引導句、轉折句、摘要結論句不需要 claim_id** — 這些屬於報告結構，不是事實斷言
7. **[⚠️BIASED_SOURCE] 標記**：若 claim 帶有此標記，代表來源 domain 佔比 > 30%（可能自家宣傳）。
   - 除非有 T1-T3 獨立來源的 claim 佐證同一事實，否則必須標記 🟠CONFLICTING
   - 在報告中加括號說明：「（此資訊來自 {domain} 自家來源，僅供參考）」

## Iterative 模式說明

你可能會收到多輪來源文件。每輪你要：
1. 審閱本輪來源原文，將有價值的資訊整合進草稿
2. 保持草稿結構完整，新資訊插入到對應段落末尾
3. 輸出完整的最新報告段落（不是 diff）

語言：繁體中文（技術術語保留原文）。
請按照 Phase 2 Step 6 的格式生成 {subq} 的報告段落。"""

        extra = f"""## 子問題：{subq}

## Approved Claims
{claims_text}"""

        source_texts = subq_sources.get(subq, [])

        if source_texts:
            section = await iterative_refine(
                sources=source_texts,
                full_research_topic=full_research_topic,
                system_prompt=integrate_system,
                extra_context=extra,
                role="writer",
            )
        else:
            response = await safe_ainvoke_chain(
                role="writer",
                messages=[
                    SystemMessage(content=integrate_system),
                    HumanMessage(content=f"""{extra}\n\n（本子問題無搜尋結果原文，請僅根據 approved claims 生成報告段落）"""),
                ],
                max_tokens=16384,
                temperature=0.2,
            )
            section = response.content

        # Referential integrity check
        approved_ids = {c.claim_id for c in approved}
        phantom_ids = _scan_phantom_claim_ids(section, approved_ids)
        if phantom_ids:
            msg = f"phase2/{subq}: 引用未核准 claim_id: {', '.join(phantom_ids)}"
            logger.warning(msg)
            local_blockers.append(msg)

        # BLOCKER 免責聲明
        if any(f"[BLOCKER: {subq}" in b for b in all_blockers):
            disclaimer = (
                f"> ⚠️ **資料不足警告（{subq}）**\n"
                f"> 本子問題的 grounding 分數未達標準，且已進行最大次數補搜（2 次）。\n"
                f"> 以下分析僅依據現有有限資料，請謹慎參考，可能存在缺漏。\n\n"
            )
            section = disclaimer + section

        write_workspace_file(workspace, f"report-sections/{subq.lower()}_section.md", section)
        return subq, section, local_blockers

    # 所有子問題段落並行寫入（原序列 for 改為 asyncio.gather）
    ordered_subqs = sorted(by_subq.items())
    results = await asyncio.gather(
        *[_write_one_section(sq, sq_claims) for sq, sq_claims in ordered_subqs],
        return_exceptions=True,
    )

    report_sections = []
    for r in results:
        if isinstance(r, BaseException):
            logger.error("phase2 section error: %s", r)
            continue
        _, section, local_blockers = r
        report_sections.append(section)
        blockers.extend(local_blockers)

    return {
        "report_sections": report_sections,
        "blockers": blockers,
        "execution_log": [
            f"Phase 2 完成：{len(report_sections)} 段落，{len(approved)} approved claims 整合"
        ],
    }


def _gather_source_texts(
    workspace: str,
    claims: list[Claim],
    sources: list[Source] | None = None,
) -> dict[str, list[str]]:
    """Read search result files relevant to approved claims, grouped by subquestion.

    回傳 dict[subquestion, list[source_text]]，每篇 source 為一個獨立字串。
    這個格式讓 iterative_refine 可以分批處理（BM25 排序 + 貪心塞入）。

    T1-T3 sources are placed before T4-T5 so iterative_refine feeds higher-quality
    content to the integrator first (BM25 fallback still applies within each batch).
    """
    # Build tier lookup: source_id → sort key (T1=0 … T5=4)
    tier_map: dict[str, int] = {}
    if sources:
        for s in sources:
            tier_map[s.source_id] = tier_rank(s.tier)

    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}  # subq → seen source_ids

    for claim in claims:
        subq = claim.subquestion
        if subq not in result:
            result[subq] = []
            seen[subq] = set()

        # Sort source_ids by tier quality (T1 first, T5 last)
        sorted_sids = sorted(claim.source_ids, key=lambda sid: tier_map.get(sid, 3))

        for sid in sorted_sids:
            if sid in seen[subq]:
                continue

            # Try multiple path patterns
            for pattern in [
                f"search-results/{subq}/{sid}.md",
                f"search-results/{subq.upper()}/{sid}.md",
            ]:
                content = read_workspace_file(workspace, pattern)
                if content:
                    result[subq].append(f"--- {sid} ---\n{content}")
                    seen[subq].add(sid)
                    break

    return result
