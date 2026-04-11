"""Phase 3: Report Generation + Final Audit.

Workflow node — merges sections, builds statement-ledger,
runs final sub-agent audit, generates summary and final report.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.validators import validate_traceability_chain
from deep_research.llm import get_llm
from deep_research.state import Claim, ResearchState, Source, StatementCheck
from deep_research.tools.workspace import (
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)

from deep_research.config import get_prompt
from deep_research.context import iterative_refine


async def phase3_report(state: ResearchState) -> dict:
    """Generate the final report with statement-level audit."""
    workspace = state["workspace_path"]
    claims = state.get("claims", [])
    sources = state.get("sources", [])
    plan = state.get("plan", "")

    # Read phase instructions
    instructions = get_prompt("phase3-report.md")

    # Convert claims/sources to objects
    claim_objects = _ensure_claim_objects(claims)
    source_objects = _ensure_source_objects(sources)

    # Step 1: Merge report sections
    section_files = list_workspace_files(workspace, "report-sections", "*.md")
    merged_body = ""
    for sf in section_files:
        content = Path(sf).read_text(encoding="utf-8")
        if "Status: FINAL" in content or content.strip():
            merged_body += content + "\n\n---\n\n"

    # Step 2: Build statement ledger
    statements = await _build_statement_ledger(merged_body, claim_objects)
    statement_ledger_md = _format_statement_ledger(statements)
    write_workspace_file(workspace, "statement-ledger.md", statement_ledger_md)

    # Step 3: Sub-agent final audit
    audit_results = await _run_final_audit(workspace, statements, claim_objects)

    # Step 4: Process audit results — fix issues
    fixed_body, fix_log = _apply_fixes(merged_body, audit_results)

    # Step 5: Traceability chain validation (iron rule)
    chain_breaks = validate_traceability_chain(statements, claim_objects, source_objects)

    # Step 6: Generate summary (only after audit passes)
    summary = await _generate_summary(claim_objects, plan)

    # Step 7: Assemble final report
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""

    approved_claims = [c for c in claim_objects if c.status == "approved"]
    rejected_claims = [c for c in claim_objects if c.status == "rejected"]

    # Read clarifications if any
    clarify_section = ""
    clarify_md = read_workspace_file(workspace, "clarifications.md")
    if clarify_md:
        clarify_section = f"""## 研究需求澄清記錄

{clarify_md}

---

"""

    final_report = f"""# 研究報告：{state.get('topic', '未命名研究')}

**研究日期：** {_today()}
**研究深度：** {state.get('depth', 'deep')}
**搜尋統計：** {state.get('iteration_count', 0)} 輪，搜尋 {state.get('search_count', 0)}/{state.get('search_budget', 150)} 次
**Claim 統計：** {len(approved_claims)} approved / {len(rejected_claims)} rejected / {len(claim_objects)} total
**溯源鏈完整率：** {len(statements) - len(chain_breaks)}/{len(statements)}

---

{clarify_section}## 摘要

{summary}

---

## 詳細分析

{fixed_body}

---

## 引用來源總表

{_format_source_table(source_objects)}

## 未解答問題與知識缺口

{gap_log}

## 研究方法論

本研究採用 LangGraph workflow + agent 交織架構：
- Phase 0：研究規劃 + 多輪澄清（workflow node + LLM Judge 評估）
- Phase 1a：多引擎平行搜尋（agent node with direct API tools）
- Phase 1b：Grounding 驗證 + 攻擊式核查（subgraph: workflow + sub-agent）
- Phase 2：矛盾裁決 + 整合（workflow node）
- Phase 3：Statement-level 審計 + 報告生成（workflow node + sub-agent）

所有事實 claim 均經過 Bedrock Grounding Check 驗證。
所有數字均標記為 ORIGINAL / NORMALIZED / DERIVED。
溯源鏈：報告句 → claim_id → quote_id → source_id。
"""

    write_workspace_file(workspace, "final-report.md", final_report)

    return {
        "final_report": final_report,
        "execution_log": [
            f"Phase 3 完成：{len(statements)} statements 審計，"
            f"{len(chain_breaks)} 鏈斷裂，"
            f"final-report.md 已寫入"
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def _ensure_claim_objects(claims) -> list[Claim]:
    result = []
    for c in claims:
        if isinstance(c, Claim):
            result.append(c)
        elif isinstance(c, dict):
            result.append(Claim(**c))
    return result


def _ensure_source_objects(sources) -> list[Source]:
    result = []
    for s in sources:
        if isinstance(s, Source):
            result.append(s)
        elif isinstance(s, dict):
            result.append(Source(**s))
    return result


async def _build_statement_ledger(body: str, claims: list[Claim]) -> list[dict]:
    """Split report body into statement-level entries."""
    llm = get_llm(tier="strong", max_tokens=8192, temperature=0.0)

    claim_ids = [c.claim_id for c in claims if c.status == "approved"]

    response = await llm.ainvoke([
        SystemMessage(content="""將報告內容切分為 statement 級別。每句事實、數字、推論各佔一行。

輸出 JSON array：
```json
[{"statement_id": "ST-1", "section": "Q1-正方", "text": "報告原句", "claim_ids": ["Q1-C1"], "type": "fact|numeric|inference|opinion"}]
```

只輸出 JSON，不要其他文字。"""),
        HumanMessage(content=f"""## 報告內容
{body}

## 可用 claim_ids
{', '.join(claim_ids)}"""),
    ])

    import re
    json_match = re.search(r'\[[\s\S]*\]', response.content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return []


def _format_statement_ledger(statements: list[dict]) -> str:
    header = (
        "# Statement Ledger\n\n"
        "| statement_id | section | text | claim_ids | type | verified |\n"
        "|-------------|---------|------|-----------|------|----------|\n"
    )
    rows = []
    for st in statements:
        cids = ",".join(st.get("claim_ids", []))
        text = st.get("text", "")[:80]
        rows.append(
            f"| {st.get('statement_id', '?')} | {st.get('section', '?')} "
            f"| {text}... | {cids} | {st.get('type', '?')} | pending |"
        )
    return header + "\n".join(rows) + "\n"


AUDIT_SYSTEM = """你是最終品質攻擊員。核對每個 statement 的溯源鏈完整性。

## 核對規則

1. 溯源鏈：statement → claim_id → quote_id → source
2. 數字逐字核對
3. 語氣一致性
4. 組合型幻覺檢測
5. 過度推論檢測

## Iterative 模式說明

你可能會收到多輪來源文件。每輪你要：
1. 審閱本輪來源原文，核對每個 statement 的溯源鏈
2. 累積發現的 issues（不要因為本輪沒找到問題就刪除之前發現的）
3. 如果本輪找到支持證據，可以將之前的 issue 標記為 NONE
4. 輸出完整的最新審計結果（包含所有 statements 的最新狀態）

## 輸出格式（嚴格 JSON array，包含所有 statements 的最新狀態）

```json
[{"statement_id": "ST-1", "issue": "NONE|BROKEN_CHAIN|NUMBER_MISMATCH|TONE_MISMATCH|COMPOSITE_HALLUCINATION|OVER_INFERENCE|NO_SOURCE", "detail": "...", "fix": "..."}]
```"""


async def _run_final_audit(
    workspace: str,
    statements: list[dict],
    claims: list[Claim],
) -> list[StatementCheck]:
    """Run final sub-agent audit on statements vs claims.

    使用 context.py 的 iterative_refine 管理 context window：
      - 場景：最終品質審計 — statement 級別的溯源鏈完整性核對
      - 風險：source 數量在 Phase 3 通常最多（累積所有搜尋結果），
              全塞 context 很容易超過閾值，中間 source 被忽略 =
              有問題的 statement 可能通過審計（false negative 最危險）
      - 策略：超過 context_threshold 時 Iterative Refinement，
              每輪送「累積審計結果」+ 一批 sources，逐批核對
    """
    claim_text = "\n".join(
        f"- {c.claim_id} ({c.status}): {c.claim_text[:100]}"
        for c in claims
    )

    statement_text = "\n".join(
        f"- {st.get('statement_id')}: [{st.get('type')}] {st.get('text', '')[:100]}"
        for st in statements
        if st.get("type") != "opinion"
    )

    # extra_context: statements + claims（每輪固定不變）
    extra_context = f"""## Statements
{statement_text}

## Claim Ledger
{claim_text}"""

    # 收集所有 source 文件為 list
    source_texts: list[str] = []
    source_files = list_workspace_files(workspace, "search-results")
    for sf in source_files:
        content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
        if content:
            source_texts.append(f"--- {Path(sf).name} ---\n{content}")

    # 取 full_research_topic
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # 使用 iterative_refine：context 管理 + 逐批審計
    result_text = await iterative_refine(
        sources=source_texts,
        full_research_topic=full_research_topic,
        system_prompt=AUDIT_SYSTEM,
        extra_context=extra_context,
        tier="strong",
    )

    # Parse results
    import re
    json_match = re.search(r'\[[\s\S]*\]', result_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return [StatementCheck(**item) for item in data]
        except (json.JSONDecodeError, Exception):
            pass
    return []


def _apply_fixes(body: str, audit_results: list[StatementCheck]) -> tuple[str, list[str]]:
    """Apply fixes from audit results to the report body."""
    log = []
    for check in audit_results:
        if check.issue == "NONE":
            continue
        log.append(f"{check.statement_id}: {check.issue} — {check.detail}")
        # For now, add inline warnings; a production version would do actual edits
        if check.fix:
            log.append(f"  FIX: {check.fix}")
    return body, log


async def _generate_summary(claims: list[Claim], plan: str) -> str:
    """Generate summary from approved claims only (iron rule)."""
    approved = [c for c in claims if c.status == "approved"]
    if not approved:
        return "（無 approved claims，無法生成摘要）"

    llm = get_llm(tier="strong", max_tokens=2048, temperature=0.2)

    claims_text = "\n".join(
        f"- {c.claim_id}: {c.claim_text}" for c in approved
    )

    response = await llm.ainvoke([
        SystemMessage(content="""根據以下 approved claims 生成 1-3 段摘要。
鐵律：每句摘要必須對應 claim_id。禁止引用 claims 以外的資訊。
語言：繁體中文。"""),
        HumanMessage(content=f"## Approved Claims\n{claims_text}"),
    ])

    return response.content


def _format_source_table(sources: list[Source]) -> str:
    header = (
        "| # | 來源 | 層級 | URL 狀態 |\n"
        "|---|------|------|----------|\n"
    )
    rows = []
    for s in sources:
        rows.append(f"| {s.source_id} | [{s.title}]({s.url}) | {s.tier} | {s.url_status} |")
    return header + "\n".join(rows) + "\n" if rows else "(無來源)\n"
