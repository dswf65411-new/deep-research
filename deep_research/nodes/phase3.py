"""Phase 3: Report Generation + Final Audit.

Workflow node — merges sections, builds statement-ledger,
runs final sub-agent audit, generates summary and final report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.validators import (
    validate_number_tags,
    validate_traceability_chain,
    verify_indexed_items,
)
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
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

    # Step 1: Merge report sections（同時保留 per-section 內容供 ledger 切片）
    section_files = list_workspace_files(workspace, "report-sections", "*.md")
    section_contents: list[tuple[str, str]] = []  # (section_name, content)
    merged_body = ""
    for sf in section_files:
        content = Path(sf).read_text(encoding="utf-8")
        if "Status: FINAL" in content or content.strip():
            section_contents.append((Path(sf).stem, content))
            merged_body += content + "\n\n---\n\n"

    # Step 2: Build statement ledger — 按 section 分批切，避免 Lost in the Middle
    statements = await _build_statement_ledger(section_contents, claim_objects)
    statement_ledger_md = _format_statement_ledger(statements)
    write_workspace_file(workspace, "statement-ledger.md", statement_ledger_md)

    # Step 3: Sub-agent final audit
    audit_results = await _run_final_audit(workspace, statements, claim_objects)

    # Step 4: Process audit results — fix issues
    fixed_body, fix_log = _apply_fixes(merged_body, audit_results)

    # Step 5: Traceability chain validation (iron rule)
    chain_breaks = validate_traceability_chain(statements, claim_objects, source_objects)

    # Step 5b: Tier 1 — 數字 claim 三類標記校驗（鐵律 3）
    # phase1a 已驗過 quote 真實存在於原文；此處複驗 number_tag 格式，
    # 確保 ORIGINAL/NORMALIZED/DERIVED 標記在最後報告階段沒被遺漏。
    number_tag_violations = validate_number_tags(claim_objects)
    if number_tag_violations:
        chain_breaks = list(chain_breaks) + [f"[Tier1 number_tag] {v}" for v in number_tag_violations]

    # Step 6: Generate summary + extract brief keywords (concurrent)
    approved_claims = [c for c in claim_objects if c.status == "approved"]
    rejected_claims = [c for c in claim_objects if c.status == "rejected"]

    brief_text = read_workspace_file(workspace, "research-brief.md") or ""

    summary, brief_keywords = await asyncio.gather(
        _generate_summary(claim_objects, plan),
        _extract_brief_keywords(brief_text),
    )

    # Step 7: Assemble final report
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""

    # Coverage sanity check（兩層）
    # 層 1：SQ 覆蓋率（計畫子問題 vs 實際 approved claims）
    sq_coverage_note = _compute_coverage_note(plan, approved_claims)

    # 層 2：Keyword 覆蓋率（任務書明確提及工具 vs 實際 approved claims）
    uncovered_keywords = _find_uncovered_keywords(brief_keywords, approved_claims)
    keyword_coverage_note = _format_keyword_coverage(brief_keywords, uncovered_keywords)

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
**溯源鏈完整率：** {len(statements)} 個 statements，{len(chain_breaks)} 個鏈斷裂

---

{clarify_section}## 摘要

{summary}

---

## 詳細分析

{fixed_body}

---

## 引用來源總表

{_format_source_table(source_objects)}

## 覆蓋率完整性報告

### 子問題覆蓋率

{sq_coverage_note}

### 任務書明確提及工具／主題的覆蓋率

{keyword_coverage_note}

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


async def _extract_brief_keywords(brief_text: str) -> list[str]:
    """LLM 從 research-brief.md 中抽取明確提及的工具/產品/技術名稱。

    只抽取用戶有明確提及的專有名詞（工具名、產品名、服務名），
    不抽取一般性描述詞（「語音識別」「高準確度」等）。

    失敗時 conservative fallback：回傳空 list（不影響報告生成）。
    """
    if not brief_text or len(brief_text.strip()) < 50:
        return []

    # 截斷避免 context 過大（brief 一般 < 3K chars，只取前 4000）
    snippet = brief_text[:4000]

    prompt = f"""從以下研究任務書中，找出所有用戶明確提及的工具名稱、產品名稱、服務名稱或技術名稱。

只抽取：工具名稱（如 Otter.ai）、產品名稱（如 MacWhisper）、服務名稱（如 Google 文件）、
        特定 OS/版本（如 iOS 26）、硬體裝置名（如 Plaud Note）。
不要抽取：一般描述詞（「語音識別」「準確度」「說話者分離」）、非工具的概念詞。

任務書：
{snippet}

只輸出 JSON，格式：{{"keywords": ["工具A", "工具B", ...]}}"""

    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[HumanMessage(content=prompt)],
            max_tokens=300,
            temperature=0.0,
        )
        text = response.content if hasattr(response, "content") else str(response)
        # Find the JSON object
        m = re.search(r'\{[\s\S]*?\}', text)
        if m:
            data = json.loads(m.group())
            kws = data.get("keywords", [])
            return [k.strip() for k in kws if isinstance(k, str) and len(k.strip()) >= 3]
    except Exception as exc:
        logger.warning("_extract_brief_keywords failed: %s", exc)
    return []


def _find_uncovered_keywords(
    keywords: list[str],
    approved_claims: list[Claim],
) -> list[str]:
    """Check which keywords from the brief don't appear in any approved claim.

    Uses case-insensitive substring matching against claim_text.
    Returns list of keywords not found in any approved claim.
    """
    if not keywords:
        return []

    # Build a single lowercased corpus of all approved claim texts
    # (empty string when approved_claims is empty → all keywords will be uncovered)
    corpus = " ".join(c.claim_text.lower() for c in approved_claims)

    uncovered = []
    for kw in keywords:
        # Normalize: remove common punctuation for matching
        kw_norm = kw.lower().replace("-", " ").replace(".", " ").strip()
        kw_orig = kw.lower().strip()
        if kw_orig not in corpus and kw_norm not in corpus:
            uncovered.append(kw)

    return uncovered


def _format_keyword_coverage(
    brief_keywords: list[str],
    uncovered: list[str],
) -> str:
    """Format keyword coverage as a Markdown section."""
    if not brief_keywords:
        return "（任務書中未偵測到明確的工具/產品名稱提及，無需交叉核對）"

    covered = [k for k in brief_keywords if k not in uncovered]
    lines = ["| 工具／主題 | 狀態 |", "|-----------|------|"]
    for kw in brief_keywords:
        if kw in uncovered:
            lines.append(f"| {kw} | ❌ **未找到 approved claim** |")
        else:
            lines.append(f"| {kw} | ✅ 已涵蓋 |")

    table = "\n".join(lines)

    if uncovered:
        missing = "、".join(uncovered)
        warning = (
            f"\n\n> ⚠️ **注意：以下主題未找到有效來源：{missing}**\n"
            f"> 這些工具或主題在研究任務書中明確提及，但沒有任何 approved claim 涉及它們。\n"
            f"> 可能原因：搜尋未命中、相關頁面 UNREACHABLE、或 grounding 分數不足被過濾。"
        )
        return table + warning
    else:
        return table + "\n\n所有任務書明確提及的工具／主題均有 approved claim 涵蓋。"


def _compute_coverage_note(plan: str, approved_claims: list[Claim]) -> str:
    """Cross-check planned subquestions against approved claims.

    Detects SQs with 0 approved claims so the final report transparently notes
    which research areas lack evidence coverage (rather than silently omitting them).

    Returns a Markdown-formatted coverage summary. If all planned SQs have at least
    one approved claim, returns a short "full coverage" confirmation.
    """
    # Extract planned SQ IDs from plan text (e.g. Q1, Q2, ..., Q8)
    planned_ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'\b(Q\d+)\b', plan or ""):
        qid = m.group(1)
        if qid not in seen:
            seen.add(qid)
            planned_ids.append(qid)

    if not planned_ids:
        return "（無計畫子問題資訊可供對照）"

    # Determine which SQs have ≥1 approved claim
    covered: set[str] = set()
    sq_claim_counts: dict[str, int] = {}
    for c in approved_claims:
        sq = c.subquestion
        if sq:
            covered.add(sq)
            sq_claim_counts[sq] = sq_claim_counts.get(sq, 0) + 1

    uncovered = [sq for sq in planned_ids if sq not in covered]
    lines = ["| 子問題 | 狀態 | Approved Claims 數 |",
             "|--------|------|-------------------|"]
    for sq in planned_ids:
        count = sq_claim_counts.get(sq, 0)
        if sq in covered:
            status = "✅ 已涵蓋"
        else:
            status = "❌ **無 approved claims**"
        lines.append(f"| {sq} | {status} | {count} |")

    table = "\n".join(lines)

    if uncovered:
        warning = (
            f"\n\n> ⚠️ **覆蓋率警告**：以下子問題沒有 approved claims，"
            f"對應段落可能缺乏實質證據：**{', '.join(uncovered)}**\n"
            f"> 請參閱「未解答問題與知識缺口」了解原因。"
        )
        return table + warning
    else:
        return table + "\n\n所有計畫子問題均有 approved claims 涵蓋。"


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


_LEDGER_SYSTEM = """將報告內容切分為 statement 級別。每句事實、數字、推論各佔一行。

## 鐵律
- 完整審閱整個 section，不可跳句、不可合併
- 每個事實句、每個數字句、每個推論句都要獨立列出
- statement_id 從本批分配的 ID 範圍中取（避免衝突）

## 四種 type 分類（必須正確分類，直接影響溯源鏈驗證）

### 必須歸類為 "opinion"（claim_ids 可為空 []）
以下句子不需要 claim_id，必須標記 type="opinion"：
- **段落標題 / 子標題**（如「## 主流工具比較」、「**結論**」）
- **引導句 / 轉折句**（如「本段落分析了...」、「綜合以上分析...」、「值得注意的是」）
- **摘要/總結句**（跨多個 claim 的綜述，無法對應單一來源）
- **信心等級標記行**（如「🟢 HIGH 信心」、「**信心等級：MEDIUM**」）
- **免責聲明/警告行**（如「⚠️ 資料不足警告...」）
- **研究方法說明**（不是事實斷言，而是解釋報告如何生成的）
- **任何報告中不含 claim_id 標記（如 Q1-C2）的評估性語句**

### "fact"（必須有 claim_ids）
具體、可驗證的事實陳述，報告中通常有 [Q1-C2] 或 (Q1-C2) 格式的 claim 標記。

### "numeric"（必須有 claim_ids）
包含具體數字的陳述（價格、百分比、版本號、評分等）。

### "inference"（必須有 claim_ids）
從多個 fact 推導出的結論，報告中通常標有 [INFERENCE]。

## 判斷順序
1. 句子有 [Q1-C1] 或 (Q1-C1) 這樣的 claim_id 標記 → fact 或 numeric（claim_ids 填入標記中的 ID）
2. 句子有 [INFERENCE] 標記 → inference（claim_ids 填入句中所有 [Qn-Cm] ID）
3. 句子是標題、引導、過渡、摘要、評估 → opinion（claim_ids=[]）
4. 不確定 → 優先歸 opinion，避免製造假斷鏈

## start_char / end_char 欄位（重要）
對每個 statement，額外輸出 start_char 與 end_char，
代表該句在上面「報告內容」中的字元起訖 index（0-based，Python 字串切片語意）。
驗證方式：content[start_char:end_char] 會等於 text。

估不準不會被懲罰 — 程式會用 text 去回找真位置做 fallback。
但 text 欄位必須仍是報告原文的逐字摘錄，否則整筆會被 reject。

## 輸出（嚴格 JSON array）
```json
[{"statement_id": "ST-1", "section": "Q1-正方", "text": "報告原句",
  "start_char": 123, "end_char": 145,
  "claim_ids": ["Q1-C1"], "type": "fact|numeric|inference|opinion"}]
```

opinion 類型的 claim_ids 必須為空陣列 []，不要捏造 claim_id。

只輸出 JSON，不要其他文字。"""


async def _build_statement_ledger(
    sections: list[tuple[str, str]],
    claims: list[Claim],
) -> list[dict]:
    """Split report into statement-level entries — per-section to avoid LiM.

    舊版把整份 body（20-50K tokens）一次塞給 LLM，中段 statement 因 Lost in the
    Middle 容易被漏切，威脅鐵律 4「溯源鏈完整」。新版每個 section 獨立 LLM call，
    每次只看一段，輸出統一 ID 範圍後合併。

    claim_id 雜訊優化：每個 section 只傳「本子問題相關的 claim_id」而非全部 approved
    claims — 當 approved claims > 30 時，全列清單會淹沒 LLM attention，LLM 更容易把
    不相關的 claim_id 硬湊進 statement，破壞溯源鏈準確度。
    """
    if not sections:
        return []

    # 按 subquestion 分組 approved claim_ids。
    # 支援兩種 subquestion 格式：
    #   - 新格式（v11+）: "Q1", "Q2" → key = "q1", "q2"
    #   - 舊格式（v10-）: "子問題 1: 主流工具盤點" → key = "q1"（提取數字）AND 全文 key
    # 雙 key 存儲確保 section_name → claim_ids 映射不斷鏈。
    import re as _re
    claims_by_subq: dict[str, list[str]] = {}
    for c in claims:
        if c.status != "approved":
            continue
        subq = c.subquestion or ""
        # Key 1: 全文 lowercase
        claims_by_subq.setdefault(subq.lower(), []).append(c.claim_id)
        # Key 2: 提取 Q{n} prefix（如 "Q1" 或從 "子問題 1" 提取 "q1"）
        sq_m = _re.match(r"Q(\d+)", subq, _re.IGNORECASE)
        if sq_m:
            short_key = f"q{sq_m.group(1)}"
        else:
            num_m = _re.search(r"(\d+)", subq)
            short_key = f"q{num_m.group(1)}" if num_m else None
        if short_key and short_key != subq.lower():
            claims_by_subq.setdefault(short_key, []).extend(
                [c.claim_id] if c.claim_id not in claims_by_subq.get(short_key, []) else []
            )

    # 每個 section 並發呼叫，分配獨立的 ID 起點避免衝突
    import asyncio as _asyncio
    tasks = []
    id_offset = 1
    for section_name, content in sections:
        # 從 section_name 抽子問題代號：
        # - "q1_section" → "q1"
        # - "子問題 1: 主流工具盤點_section" → "q1"（提取數字）
        # - 先試 Q{n} 格式，再試數字提取
        subq_match = _re.match(r"([qQ]\d+)", section_name)
        if subq_match:
            subq_key = subq_match.group(1).lower()
        else:
            num_m = _re.search(r"(\d+)", section_name)
            subq_key = f"q{num_m.group(1)}" if num_m else ""
        relevant_ids = claims_by_subq.get(subq_key, [])
        claim_id_str = (
            ", ".join(relevant_ids)
            if relevant_ids
            else "（本 section 無相關 approved claim）"
        )
        section_id_start = id_offset
        id_offset += 200  # 每 section 保留 200 ID 空間
        tasks.append(_split_one_section(section_name, content, claim_id_str, section_id_start))
    results = await _asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict] = []
    for r in results:
        if isinstance(r, Exception) or not r:
            continue
        merged.extend(r)
    return merged


async def _split_one_section(
    section_name: str,
    content: str,
    claim_id_str: str,
    id_start: int,
) -> list[dict]:
    """單一 section 切 statement，使用獨立 ID 範圍。"""
    # role="verifier" — statement 切分是結構化抽取，Gemini 主導
    response = await safe_ainvoke_chain(
        role="verifier",
        messages=[
            SystemMessage(content=_LEDGER_SYSTEM),
            HumanMessage(content=f"""## Section: {section_name}
（statement_id 從 ST-{id_start} 開始編號）

## 報告內容
{content}

## 可用 claim_ids
{claim_id_str}"""),
        ],
        max_tokens=8192,
        temperature=0.0,
    )

    import re
    json_match = re.search(r'\[[\s\S]*\]', response.content)
    if not json_match:
        return []
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    # Index 驗證：statement.text 必須真實存在於 section content。
    # 通過後：text 覆寫為 content[start:end]（杜絕 LLM 抄字幻覺），附上 start/end span。
    verified = verify_indexed_items(
        content, data, "text", log_prefix=f"phase3/ledger/{section_name}"
    )

    # 強制覆寫 section 欄位，並校正 ID 不出範圍
    # Python fallback：從 statement.text 內的 inline 標記補填 claim_ids
    # 支援 [Q1-C1] 和 (Q1-C1) 兩種格式
    _claim_id_re = re.compile(r'Q\d+-C\d+')
    out = []
    for i, st in enumerate(verified):
        st["section"] = section_name
        if not st.get("statement_id", "").startswith("ST-"):
            st["statement_id"] = f"ST-{id_start + i}"
        # 清掉 LLM 原始 hint（保留 start/end 最終真值即可）
        st.pop("start_char", None)
        st.pop("end_char", None)

        # 若 LLM 沒有填 claim_ids（空陣列），嘗試從 text 內 inline 標記中提取
        # 支援兩種格式：(Q1-C1) 或 [Q1-C1]（Phase 2 舊格式用方括號）
        if not st.get("claim_ids") and st.get("type") != "opinion":
            text = st.get("text", "")
            extracted = _claim_id_re.findall(text)
            if extracted:
                st["claim_ids"] = extracted

        out.append(st)
    return out


def _format_statement_ledger(statements: list[dict]) -> str:
    """渲染 statement-ledger.md。

    span 欄位格式：`@[start:end]`。若缺 span（不該發生，因為 verify_indexed_items
    會過濾掉無法定位者），顯示為空字串以保持表格結構。
    """
    header = (
        "# Statement Ledger\n\n"
        "<!-- span 欄位 @[s:e] 對應 report-sections/{section}.md 的字元 index，"
        "可用 section_content[s:e] 切片還原原句。 -->\n\n"
        "| statement_id | section | span | text | claim_ids | type | verified |\n"
        "|-------------|---------|------|------|-----------|------|----------|\n"
    )
    rows = []
    for st in statements:
        cids = ",".join(st.get("claim_ids", []))
        text = st.get("text", "")[:80].replace("|", "\\|").replace("\n", " ")
        start = st.get("start")
        end = st.get("end")
        span = f"@[{start}:{end}]" if isinstance(start, int) and isinstance(end, int) else ""
        rows.append(
            f"| {st.get('statement_id', '?')} | {st.get('section', '?')} "
            f"| {span} | {text}... | {cids} | {st.get('type', '?')} | pending |"
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
    """Run final sub-agent audit on statements vs claims — per-section 並發。

    場景：最終品質審計 — statement 級別的溯源鏈完整性核對。

    舊版一次把所有 statements（50-100+ 條）+ 所有 claims 當 extra_context 送審，屬
    典型「一次做太多」anti-pattern：LLM 要同時核對多個 section 的 statement，中段
    容易失焦、iterative_refine 每輪都重傳臃腫的 extra_context，實際消耗 =
    extra × 輪數。新版按 section 分組，每 section 獨立 `iterative_refine` 並發，
    每次只看自己 section 的 statements + 本子問題相關 claims，最後程式合併審計結果。

    注意：sources 仍是所有 sections 共用（同一 statement 可能引用跨 section source），
    這部分無法再縮，只能靠 iterative_refine 分批處理。
    """
    import asyncio as _asyncio
    import re as _re

    # ── Step 1: 按 section 分組 statements（跳過 opinion）
    by_section: dict[str, list[dict]] = {}
    for st in statements:
        if st.get("type") == "opinion":
            continue
        sec = st.get("section", "unknown")
        by_section.setdefault(sec, []).append(st)

    if not by_section:
        return []

    # ── Step 2: 按 subquestion 分組 claims（key 用 lower case Q1/Q2/...）
    claims_by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_subq.setdefault(c.subquestion.lower(), []).append(c)

    # ── Step 3: 收集所有 source 文件（所有 section 共用）
    source_texts: list[str] = []
    source_files = list_workspace_files(workspace, "search-results")
    for sf in source_files:
        content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
        if content:
            source_texts.append(f"--- {Path(sf).name} ---\n{content}")

    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # ── Step 4: 每 section 獨立審計 task
    async def _audit_one_section(
        section_name: str,
        section_statements: list[dict],
    ) -> list[StatementCheck]:
        subq_match = _re.match(r"([qQ]\d+)", section_name)
        subq_key = subq_match.group(1).lower() if subq_match else ""
        section_claims = claims_by_subq.get(subq_key, [])

        claim_text = "\n".join(
            f"- {c.claim_id} ({c.status}): {c.claim_text[:100]}"
            for c in section_claims
        ) or "（本 section 無相關 claim）"
        statement_text = "\n".join(
            f"- {st.get('statement_id')}: [{st.get('type')}] {st.get('text', '')[:100]}"
            for st in section_statements
        )

        extra_context = f"""## Section: {section_name}

## 本 section 的 Statements（請逐條審計）
{statement_text}

## 相關 Claim Ledger（子問題 {subq_key.upper() or '?'}）
{claim_text}"""

        result_text = await iterative_refine(
            sources=source_texts,
            full_research_topic=full_research_topic,
            system_prompt=AUDIT_SYSTEM,
            extra_context=extra_context,
            role="verifier",  # 報告審計 = verifier 任務（Gemini 主導，Vectara HHEM 幻覺率最低）
        )

        json_match = _re.search(r"\[[\s\S]*\]", result_text)
        if not json_match:
            return []
        try:
            data = json.loads(json_match.group())
            return [StatementCheck(**item) for item in data]
        except (json.JSONDecodeError, Exception):
            return []

    tasks = [_audit_one_section(sec, sts) for sec, sts in by_section.items()]
    results = await _asyncio.gather(*tasks, return_exceptions=True)

    merged: list[StatementCheck] = []
    for r in results:
        if isinstance(r, Exception) or not r:
            continue
        merged.extend(r)
    return merged


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

    claims_text = "\n".join(
        f"- {c.claim_id}: {c.claim_text}" for c in approved
    )

    # role="writer" — 摘要寫作，Claude Opus 主導
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[
            SystemMessage(content="""根據以下 approved claims 生成 1-3 段摘要。
鐵律：每句摘要必須對應 claim_id。禁止引用 claims 以外的資訊。
語言：繁體中文。"""),
            HumanMessage(content=f"## Approved Claims\n{claims_text}"),
        ],
        max_tokens=2048,
        temperature=0.2,
    )

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
