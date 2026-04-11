"""Phase 2: Integration + Conflict Resolution.

Workflow node — reads approved claims, resolves contradictions,
writes report sections with confidence levels.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.validators import validate_claims_for_phase2, validate_numeric_claims
from deep_research.llm import get_llm
from deep_research.state import Claim, ResearchState
from deep_research.tools.workspace import (
    read_workspace_file,
    write_workspace_file,
)

from deep_research.config import get_prompt
from deep_research.context import iterative_refine


async def phase2_integrate(state: ResearchState) -> dict:
    """Integrate approved claims into report sections."""
    workspace = state["workspace_path"]
    claims = state.get("claims", [])

    # Convert to Claim objects
    claim_objects = []
    for c in claims:
        if isinstance(c, Claim):
            claim_objects.append(c)
        elif isinstance(c, dict):
            claim_objects.append(Claim(**c))

    # Iron rule: only approved claims with quote_ids
    approved = validate_claims_for_phase2(claim_objects)

    # Iron rule: numeric claims must have number_tag
    violations = validate_numeric_claims(approved)
    blockers = []
    if violations:
        blockers.extend(violations)

    # Read phase instructions
    instructions = get_prompt("phase2-integrate.md")

    # Read source files for context
    source_registry = read_workspace_file(workspace, "source-registry.md") or ""
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""

    # 取 full_research_topic
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # Group claims by subquestion
    by_subq: dict[str, list[Claim]] = {}
    for c in approved:
        by_subq.setdefault(c.subquestion, []).append(c)

    # 收集每個 subquestion 的 source 文字（供 iterative_refine 分批處理）
    subq_sources = _gather_source_texts(workspace, approved)

    report_sections = []

    for subq, subq_claims in sorted(by_subq.items()):
        claims_text = "\n".join(
            f"- {c.claim_id} [{c.claim_type}] (bedrock={c.bedrock_score:.2f}): {c.claim_text}"
            for c in subq_claims
        )

        # 整合 system prompt
        integrate_system = f"""{instructions}

你是研究報告整合器。根據已驗證的 approved claims 和來源原文，生成報告段落。

## 鐵律
1. 只使用以下 approved claims，禁止引用其他資訊
2. 每句事實必須附 claim_id
3. 跨 claim 推導必須標記 [INFERENCE]
4. 數字必須標記 ORIGINAL/NORMALIZED/DERIVED
5. 信心等級必須分配（🟢HIGH / 🟡MEDIUM / 🟠CONFLICTING / 🔴LOW）

## Iterative 模式說明

你可能會收到多輪來源文件。每輪你要：
1. 審閱本輪來源原文，將有價值的資訊整合進草稿
2. 保持草稿結構完整，新資訊插入到對應段落末尾
3. 輸出完整的最新報告段落（不是 diff）

語言：繁體中文（技術術語保留原文）。
請按照 Phase 2 Step 6 的格式生成 {subq} 的報告段落。"""

        # extra_context: claims + source_registry（每輪固定不變）
        extra = f"""## 子問題：{subq}

## Approved Claims
{claims_text}

## 來源登記
{source_registry}"""

        source_texts = subq_sources.get(subq, [])

        if source_texts:
            # 使用 iterative_refine：context 管理 + 分批整合
            section = await iterative_refine(
                sources=source_texts,
                full_research_topic=full_research_topic,
                system_prompt=integrate_system,
                extra_context=extra,
                tier="strong",
            )
        else:
            # 無 source — 直接用 claims 生成
            llm = get_llm(tier="strong", max_tokens=16384, temperature=0.2)
            response = await llm.ainvoke([
                SystemMessage(content=integrate_system),
                HumanMessage(content=f"""{extra}\n\n（本子問題無搜尋結果原文，請僅根據 approved claims 生成報告段落）"""),
            ])
            section = response.content

        # Write to workspace immediately
        section_file = f"report-sections/{subq.lower()}_section.md"
        write_workspace_file(workspace, section_file, section)
        report_sections.append(section)

    return {
        "report_sections": report_sections,
        "blockers": blockers,
        "execution_log": [
            f"Phase 2 完成：{len(report_sections)} 段落，{len(approved)} approved claims 整合"
        ],
    }


def _gather_source_texts(workspace: str, claims: list[Claim]) -> dict[str, list[str]]:
    """Read search result files relevant to approved claims, grouped by subquestion.

    回傳 dict[subquestion, list[source_text]]，每篇 source 為一個獨立字串。
    這個格式讓 iterative_refine 可以分批處理（BM25 排序 + 貪心塞入）。
    """
    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}  # subq → seen source_ids

    for claim in claims:
        subq = claim.subquestion
        if subq not in result:
            result[subq] = []
            seen[subq] = set()

        for sid in claim.source_ids:
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
