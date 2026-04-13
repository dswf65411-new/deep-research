"""Phase 0: Clarify + Research Planning.

Functions:
  generate_questions  — LLM generates clarifying questions (up to max_questions)
  validate_answers    — check answer format/completeness, return missing indices
  judge_clarity       — independent LLM (clean context) decides if topic is clear enough
  phase0_plan         — generates research plan (graph node)
  phase0_plan_standalone — same logic, callable outside graph (for skill mode)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import get_prompt
from deep_research.context import synthesize_research_topic
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)
from deep_research.tools.workspace import (
    create_workspace,
    init_execution_log,
    init_gap_log,
    init_source_registry,
    write_workspace_file,
)

# Depth → budget mapping
DEPTH_CONFIG = {
    "quick": {"budget": 30, "subquestions": "1-2", "iterations": 1, "min_budget_per_sq": 3},
    "standard": {"budget": 60, "subquestions": "2-5", "iterations": 2, "min_budget_per_sq": 6},
    "deep": {"budget": 150, "subquestions": "5-10", "iterations": 5, "min_budget_per_sq": 12},
}

# Default question limit per round
DEFAULT_MAX_QUESTIONS = 10

# QA 壓縮閾值：超過此數時，舊輪 QA 壓成主題列表
QA_COMPACT_THRESHOLD = 15
QA_KEEP_LATEST = 5


def _compact_clarifications(qas: list[dict]) -> str:
    """Format prior clarifications as context — compact when too many.

    Anti-pattern 防護（LLM 專注原則）：
      - 場景：multi-round clarification 累積到 20+ 條 QA 時，全展開塞進 LLM
              context，模型只需「不要重複問」這個訊號，不需逐字看完整答案。
      - 策略：QA <= 15 條全展開；> 15 條時，舊 N-5 條壓成「主題行」
              （Q 截 30 字 + A 截 20 字），最新 5 條保留完整供深挖。
    """
    if not qas:
        return ""

    if len(qas) <= QA_COMPACT_THRESHOLD:
        body = "\n".join(
            f"{i}. **問：** {qa['question']}\n   **答：** {qa['answer']}"
            for i, qa in enumerate(qas, 1)
        )
        return (
            "\n\n## 已取得的澄清資訊（前幾輪的結果）\n"
            f"{body}\n"
            "\n請不要重複問已經回答過的問題。只問新的、之前沒涵蓋到的面向。\n"
        )

    # 壓縮：舊的只留主題行，最新 5 條完整
    old_qas = qas[:-QA_KEEP_LATEST]
    recent_qas = qas[-QA_KEEP_LATEST:]

    old_lines = []
    for i, qa in enumerate(old_qas, 1):
        q_short = (qa["question"][:30] + "…") if len(qa["question"]) > 30 else qa["question"]
        a_short = (qa["answer"][:20] + "…") if len(qa["answer"]) > 20 else qa["answer"]
        old_lines.append(f"{i}. {q_short} → {a_short}")

    recent_lines = []
    start_idx = len(old_qas) + 1
    for i, qa in enumerate(recent_qas, start_idx):
        recent_lines.append(
            f"{i}. **問：** {qa['question']}\n   **答：** {qa['answer']}"
        )

    return (
        "\n\n## 已取得的澄清資訊（前幾輪的結果）\n"
        f"### 舊輪主題（共 {len(old_qas)} 條，已壓縮為主題列表）\n"
        + "\n".join(old_lines)
        + f"\n\n### 最新 {len(recent_qas)} 條完整 QA\n"
        + "\n".join(recent_lines)
        + "\n\n請不要重複問已經回答過的主題（含上方壓縮列表）。只問新的、之前沒涵蓋到的面向。\n"
    )


# ---------------------------------------------------------------------------
# Clarification: generate questions
# ---------------------------------------------------------------------------

async def generate_questions(
    topic: str,
    existing_clarifications: list[dict],
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    round_num: int = 1,
) -> tuple[list[str], str]:
    """Call LLM to generate clarifying questions.

    Args:
        topic: research topic
        existing_clarifications: Q&A pairs from prior rounds
        max_questions: cap on number of questions this round
        round_num: which clarification round (for prompt context)

    Returns:
        (questions, reasoning)
    """
    context = _compact_clarifications(existing_clarifications)

    round_note = ""
    if round_num > 1:
        round_note = f"\n這是第 {round_num} 輪澄清。請根據前幾輪的回答，挖掘更深層的細節或補充遺漏的面向。\n"

    system_msg = SystemMessage(content=f"""你是深度研究規劃器的澄清模組。

根據使用者的研究主題，判斷是否需要澄清。盡量多問多釐清 — 澄清是提升研究品質最重要的環節。

如果主題已經完全明確（所有面向都已涵蓋），才回覆空的 questions 陣列。
{round_note}
生成最多 {max_questions} 個問題。每個問題都要說明「為什麼需要知道這個」。

考慮以下面向（只問需要的）：
1. 研究目的：做決策？寫報告？學習？解決具體問題？
2. 期望產出：比較表？推薦結論？客觀呈現？深度技術分析？
3. 範圍邊界：包含/排除什麼？時間範圍？地域限制？
4. 背景知識：使用者已經了解什麼？已經排除什麼選項？
5. 特定偏好：技術棧限制、預算範圍、團隊規模？
6. 成功標準：什麼樣的研究結果對使用者最有價值？
7. 利害關係人：研究結果的受眾是誰？需要說服誰？
8. 時效性：需要多新的資料？是否有截止日期？
9. 深度 vs 廣度：偏好全面概覽還是某幾個面向的深入分析？
10. 已知限制：有什麼已知的約束條件？
{context}

回覆格式（嚴格 JSON）：
{{"questions": ["問題1（為什麼需要知道：原因）", "問題2（為什麼需要知道：原因）"], "reasoning": "這輪為什麼要問這些"}}

如果不需要再問：
{{"questions": [], "reasoning": "主題已完全明確，原因是..."}}""")

    human_msg = HumanMessage(content=f"研究主題：{topic}")
    # role="writer" — 規劃 / 提問類，Claude Opus 主導
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[system_msg, human_msg],
        max_tokens=4096,
        temperature=0.3,
    )

    try:
        text = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {"questions": [], "reasoning": "parse_error"}
    except (json.JSONDecodeError, AttributeError):
        parsed = {"questions": [], "reasoning": "parse_error"}

    questions = parsed.get("questions", [])[:max_questions]
    return questions, parsed.get("reasoning", "")


# ---------------------------------------------------------------------------
# Validation: check answers are complete
# ---------------------------------------------------------------------------

def validate_answers(
    questions: list[str],
    answers: dict | list | str,
) -> tuple[list[dict], list[int]]:
    """Validate user answers against questions.

    Returns:
        (valid_pairs, missing_indices)
        - valid_pairs: list of {"question": q, "answer": a} for valid answers
        - missing_indices: indices of questions with empty/missing answers
    """
    valid = []
    missing = []

    if isinstance(answers, dict):
        for i, q in enumerate(questions):
            ans = answers.get(str(i), answers.get(i, ""))
            ans = str(ans).strip() if ans else ""
            if ans:
                valid.append({"question": q, "answer": ans})
            else:
                missing.append(i)
    elif isinstance(answers, list):
        for i, q in enumerate(questions):
            ans = str(answers[i]).strip() if i < len(answers) and answers[i] else ""
            if ans:
                valid.append({"question": q, "answer": ans})
            else:
                missing.append(i)
    elif isinstance(answers, str) and answers.strip():
        # Single string = answer to all questions combined
        valid.append({"question": "; ".join(questions), "answer": answers.strip()})
    else:
        missing = list(range(len(questions)))

    return valid, missing


# ---------------------------------------------------------------------------
# LLM Judge: Three-layer clarity evaluation
#
# Architecture based on research findings:
#   Layer 1 — Analytic Rubric: 7 binary dimensions, evidence-anchored (Autorubric, Rulers)
#   Layer 2 — PoLL: Up to 3 model families vote in parallel (Verga et al. 2024)
#   Layer 3 — Adaptive Escalation: targeted follow-up for failed dimensions only
#
# References:
#   - Autorubric (2025): per-criterion binary eval, 87% accuracy, κ=0.642
#   - Rulers (2025): evidence-anchored scoring, QWK 0.7276 vs 0.5566
#   - PoLL (Verga et al. 2024): 3 diverse models, κ 0.763-0.906, 7-8x cheaper
#   - EvalGen (UIST 2024): criteria-output interdependence
# ---------------------------------------------------------------------------

CLARITY_DIMENSIONS = [
    {
        "id": "purpose",
        "name": "研究目的",
        "required": True,
        "question": "使用者的研究目的是否具體到可以判斷研究成功還是失敗？",
        "pass_example": "「為公司客服系統選擇 AI 框架」→ 有具體場景和目標",
        "fail_example": "「了解 AI」→ 太模糊，無法判斷成敗",
    },
    {
        "id": "scope",
        "name": "範圍界定",
        "required": True,
        "question": "研究的邊界（時間範圍、地域範圍、技術範圍）是否明確？是否有可能無限發散？",
        "pass_example": "「2024 年後的開源框架，排除付費 SaaS」→ 有明確邊界",
        "fail_example": "沒有任何邊界條件的說明 → 可能無限發散",
    },
    {
        "id": "output_format",
        "name": "產出格式",
        "required": True,
        "question": "使用者期望什麼形式的研究結果？",
        "pass_example": "「需要比較表 + 最終推薦 + 風險評估」→ 交付物明確",
        "fail_example": "不知道要比較表、推薦結論、還是客觀呈現",
    },
    {
        "id": "success_criteria",
        "name": "成功標準",
        "required": True,
        "question": "有沒有明確的評估維度或標準來判斷研究成果好壞？",
        "pass_example": "「從效能、成本、社群活躍度三個維度比較」→ 有評估框架",
        "fail_example": "沒有評估維度或標準 → 無法判斷好壞",
    },
    {
        "id": "consistency",
        "name": "一致性",
        "required": True,
        "question": "問答記錄中的所有回答是否邏輯一致、沒有自相矛盾？",
        "pass_example": "所有回答邏輯一致，沒有矛盾",
        "fail_example": "說「不限預算」但又說「越便宜越好」→ 存在矛盾",
    },
    {
        "id": "constraints",
        "name": "已知約束",
        "required": False,
        "question": "使用者有沒有說明硬性限制（技術棧、預算、團隊規模、合規要求）？",
        "pass_example": "「必須支持 Python，預算 1000 美元以內」→ 約束明確",
        "fail_example": "未提及任何限制（可能有但沒說）",
    },
    {
        "id": "depth_breadth",
        "name": "深度 vs 廣度",
        "required": False,
        "question": "使用者偏好全面概覽還是聚焦特定面向的深入分析？",
        "pass_example": "「深入比較前 3 名的技術細節」→ 偏好明確",
        "fail_example": "未表達偏好 → 不知道該做廣還是深",
    },
]


def _build_rubric_system_prompt(dims: list[dict] | None = None) -> str:
    """Build the rubric evaluation system prompt for a given dimension subset.

    傳入 None 時用全部 7 dim（向後相容）；正式呼叫應傳子集，每組 3-4 dim 避免
    LLM 對中段 dim 注意力下降（Lost in the Middle）。
    """
    target = dims if dims is not None else CLARITY_DIMENSIONS
    dims_text = ""
    for i, dim in enumerate(target, 1):
        req_label = "必要" if dim["required"] else "加分"
        dims_text += f"""
### {i}. {dim['name']} ({dim['id']}) 【{req_label}】
判定：{dim['question']}
- PASS：{dim['pass_example']}
- FAIL：{dim['fail_example']}
"""

    dim_ids_json = ", ".join(
        f'{{"id": "{d["id"]}", "verdict": "...", "evidence": "...", "reason": "...", "question": "..."}}'
        for d in target
    )

    return f"""你是獨立的研究品質評審（Judge）。你與提問的 LLM 完全獨立 — 你沒有看到它的推理過程，你只看到研究主題和問答記錄。

## Evidence-Anchored 規則（鐵律）
1. 每個維度的判定必須引述使用者的原文作為依據
2. 若使用者原文中找不到支持「已充分」的證據 → 必須判定 FAIL
3. 不可推測使用者可能的意圖，只根據已明確提供的資訊判斷
4. 每個維度獨立評估 — 不要讓某個維度的判定影響其他維度
5. FAIL 的維度必須提供一個針對性的追問問題（說明「為什麼需要知道」）

## 評估維度（本批共 {len(target)} 個，逐一獨立評估）
{dims_text}
## 回覆格式（嚴格 JSON，不要加任何其他文字）
{{"dimensions": [{dim_ids_json}]}}

每個維度的欄位：
- id: 維度 ID（必須與上方一致）
- verdict: "PASS" 或 "FAIL"
- evidence: 使用者原文中支持此判定的直接引述（FAIL 則說明缺少什麼）
- reason: 判定理由（一句話）
- question: 若 FAIL，追問問題（為什麼需要知道：原因）；若 PASS 則空字串 ""
"""


# Group A：required 維度（核心，前 4 個）
# Group B：optional 維度（後 3 個）
# 拆兩組各別 call，每次只評 3-4 dim，避免 LiM 影響中段判斷品質
_DIM_GROUP_A = [d for d in CLARITY_DIMENSIONS if d["required"]]
_DIM_GROUP_B = [d for d in CLARITY_DIMENSIONS if not d["required"]]


# Cached prompts (built once per group)
_RUBRIC_PROMPT_A: str | None = None
_RUBRIC_PROMPT_B: str | None = None


def _get_rubric_prompt(group: str) -> str:
    """Return cached prompt for group A or B."""
    global _RUBRIC_PROMPT_A, _RUBRIC_PROMPT_B
    if group == "A":
        if _RUBRIC_PROMPT_A is None:
            _RUBRIC_PROMPT_A = _build_rubric_system_prompt(_DIM_GROUP_A)
        return _RUBRIC_PROMPT_A
    if _RUBRIC_PROMPT_B is None:
        _RUBRIC_PROMPT_B = _build_rubric_system_prompt(_DIM_GROUP_B)
    return _RUBRIC_PROMPT_B


async def _evaluate_dim_group(
    provider: str,
    topic: str,
    qa_text: str,
    group: str,
    tier: str,
) -> list[dict]:
    """評估一組 dim（3-4 個），回傳 list of dim verdict dict。失敗回空 list。"""
    try:
        llm = get_llm(tier=tier, max_tokens=4096, temperature=0.0, provider=provider)
        system_msg = SystemMessage(content=_get_rubric_prompt(group))
        human_msg = HumanMessage(content=(
            f"## 研究主題\n{topic}\n\n"
            f"## 已完成的澄清問答\n{qa_text or '（尚未進行任何澄清）'}"
        ))
        response = await safe_ainvoke(llm, [system_msg, human_msg])
        text = response.content.strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return []
        parsed = json.loads(json_match.group())
        return parsed.get("dimensions", []) or []
    except Exception as e:
        logger.warning("judge_clarity: provider %s group %s failed: %s", provider, group, e)
        return []


async def _evaluate_single_judge(
    provider: str,
    topic: str,
    qa_text: str,
    tier: str = "fast",
) -> dict | None:
    """One provider evaluates all dimensions via rubric — split into 2 groups.

    Group A: required 維度（核心 4 個）
    Group B: optional 維度（3 個）
    並發兩次 call，每次只評 3-4 dim。比一次評 7 dim 顯著降低 Lost in the Middle
    對中段維度的影響。
    """
    import asyncio as _asyncio
    a_results, b_results = await _asyncio.gather(
        _evaluate_dim_group(provider, topic, qa_text, "A", tier),
        _evaluate_dim_group(provider, topic, qa_text, "B", tier),
    )
    dims = list(a_results) + list(b_results)
    if not dims:
        logger.warning("judge_clarity: provider %s returned empty dimensions for both groups", provider)
        return None
    return {"provider": provider, "dimensions": dims}


def _aggregate_panel_votes(panel_results: list[dict]) -> dict[str, dict]:
    """Layer 3: Majority vote per dimension across panel.

    Returns {dim_id: {"verdict", "pass_count", "fail_count", "total",
                       "unanimous", "reasons", "questions"}}
    """
    aggregated = {}

    for dim in CLARITY_DIMENSIONS:
        dim_id = dim["id"]
        pass_count = 0
        fail_count = 0
        reasons: list[str] = []
        questions: list[str] = []

        for result in panel_results:
            # Find this dimension in the provider's response
            dim_vote = next(
                (d for d in result["dimensions"] if d.get("id") == dim_id),
                None,
            )
            if dim_vote is None:
                # Provider didn't return this dimension — treat as FAIL (conservative)
                fail_count += 1
                continue

            verdict = dim_vote.get("verdict", "FAIL").upper().strip()
            if verdict == "PASS":
                pass_count += 1
            else:
                fail_count += 1

            reason = dim_vote.get("reason", "")
            if reason:
                reasons.append(f"[{result['provider']}] {reason}")

            question = dim_vote.get("question", "")
            if question and verdict != "PASS":
                questions.append(question)

        total = pass_count + fail_count
        # Majority vote; ties → FAIL (conservative)
        final_verdict = "PASS" if pass_count > fail_count else "FAIL"

        aggregated[dim_id] = {
            "verdict": final_verdict,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total": total,
            "unanimous": (pass_count == total or fail_count == total),
            "reasons": reasons,
            "questions": questions,
        }

    return aggregated


def _build_judge_verdict(
    aggregated: dict[str, dict],
    providers_used: list[str],
) -> tuple[bool, str, list[str]]:
    """Determine final clarity verdict + build reasoning + collect suggestions."""
    required_fails = []
    optional_fails = []

    for dim in CLARITY_DIMENSIONS:
        dim_id = dim["id"]
        result = aggregated.get(dim_id, {"verdict": "FAIL"})
        if result["verdict"] == "FAIL":
            if dim["required"]:
                required_fails.append(dim)
            else:
                optional_fails.append(dim)

    is_clear = len(required_fails) == 0

    # --- Build reasoning string ---
    n_providers = len(providers_used)
    provider_names = ", ".join(providers_used)
    lines = [f"## 澄清充分性評估（{n_providers} 模型: {provider_names}）\n"]

    for dim in CLARITY_DIMENSIONS:
        dim_id = dim["id"]
        result = aggregated.get(dim_id, {
            "verdict": "FAIL", "pass_count": 0, "fail_count": 0, "total": 0,
            "reasons": [], "questions": [],
        })
        req_label = "必要" if dim["required"] else "加分"
        verdict = result["verdict"]
        marker = "[PASS]" if verdict == "PASS" else "[FAIL]"
        votes = f"{result['pass_count']}/{result['total']}"

        lines.append(f"  {marker} {dim['name']} ({dim_id}) [{req_label}] — {votes}")
        # Show first reason for context
        if result["reasons"]:
            lines.append(f"        {result['reasons'][0]}")

    lines.append("")
    if is_clear:
        lines.append("結論：所有必要維度通過。")
        if optional_fails:
            names = ", ".join(d["name"] for d in optional_fails)
            lines.append(f"  （{names} 未明確，但不影響研究進行）")
    else:
        fail_names = ", ".join(d["name"] for d in required_fails)
        lines.append(
            f"結論：{len(required_fails)} 個必要維度未通過（{fail_names}），"
            f"建議針對性追問。"
        )

    reasoning = "\n".join(lines)

    # --- Collect targeted questions from failed required dimensions ---
    suggested_questions: list[str] = []
    seen_prefixes: set[str] = set()
    for dim in required_fails:
        for q in aggregated[dim["id"]].get("questions", []):
            prefix = q[:40]
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                suggested_questions.append(q)

    return (is_clear, reasoning, suggested_questions)


async def judge_clarity(
    topic: str,
    clarifications: list[dict],
) -> tuple[bool, str, list[str]]:
    """Three-layer clarity judge: Rubric + PoLL + Adaptive Escalation.

    Layer 1 (Analytic Rubric): Decomposes "is the topic clear?" into 7 binary
        dimensions, each independently evaluated with evidence-anchored scoring.
    Layer 2 (PoLL): Sends rubric to up to 3 different model families in parallel,
        aggregates via majority vote per dimension.
    Layer 3 (Adaptive Escalation): All required dims pass → clear;
        failed dims → targeted follow-up questions (not broad re-ask).

    Returns:
        (is_clear, reasoning, suggested_questions)
    """
    from deep_research.llm import get_available_providers, get_provider

    # Format Q&A text
    qa_text = ""
    for i, qa in enumerate(clarifications, 1):
        qa_text += f"{i}. 問：{qa['question']}\n   答：{qa['answer']}\n\n"

    # Layer 2: Get available providers for PoLL (up to 3 diverse families)
    providers = get_available_providers()[:3]

    # Evaluate with all providers in parallel
    tasks = [
        _evaluate_single_judge(p, topic, qa_text, tier="fast")
        for p in providers
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    valid_results = [r for r in raw_results if isinstance(r, dict)]

    if not valid_results:
        # All fast-tier calls failed — single retry with strong tier, current provider
        logger.warning("judge_clarity: all panel calls failed, retrying with strong tier")
        fallback = await _evaluate_single_judge(
            get_provider(), topic, qa_text, tier="strong",
        )
        if fallback:
            valid_results = [fallback]
        else:
            # Total failure — default to clear to not block the user
            return (True, "（所有評估模型均失敗，預設通過）", [])

    providers_used = [r["provider"] for r in valid_results]

    # Layer 3: Aggregate votes + Adaptive Escalation
    aggregated = _aggregate_panel_votes(valid_results)
    return _build_judge_verdict(aggregated, providers_used)


# ---------------------------------------------------------------------------
# Graph node: phase0_plan
# ---------------------------------------------------------------------------

async def phase0_plan(state: ResearchState) -> dict:
    """Generate research plan from topic + clarifications (graph node).

    同時統整 topic + refs + clarifications → full_research_topic，
    作為整個研究流程的固定 context（prompt prefix caching 友善）。
    """
    topic = state["topic"]
    depth = state.get("depth", "deep")
    config = DEPTH_CONFIG[depth]
    budget = state.get("search_budget", config["budget"])
    clarifications = state.get("clarifications", [])
    refs = state.get("refs", [])

    instructions = get_prompt("phase0-clarify.md")
    workspace_path = create_workspace(topic)

    # Step 1: 先統整 research brief（topic + refs + clarifications → 結構化任務書）
    # 這份任務書消除矛盾、補充隱含需求，品質遠高於 raw QA 對
    full_research_topic = await synthesize_research_topic(topic, refs, clarifications)

    # Step 2: 用統整後的 research brief 生成計畫（而非 raw clarifications）
    plan_content = await _generate_plan(
        topic, depth, budget, config, clarifications, instructions,
        research_brief=full_research_topic,
    )

    # Write workspace files
    _write_workspace_files(workspace_path, topic, budget, clarifications, plan_content)

    # 也將 full_research_topic 寫入 workspace 供參考
    write_workspace_file(workspace_path, "research-brief.md", full_research_topic)

    return {
        "plan": plan_content,
        "full_research_topic": full_research_topic,
        "depth": depth,
        "search_budget": budget,
        "search_count": 0,
        "workspace_path": workspace_path,
        "iteration_count": 0,
        "coverage_status": {},
        "execution_log": [f"Phase 0 完成：workspace={workspace_path}，研究任務書已統整"],
    }


# ---------------------------------------------------------------------------
# Standalone plan generation (for skill mode)
# ---------------------------------------------------------------------------

async def phase0_plan_standalone(
    topic: str,
    depth: str,
    budget: int,
    clarifications: list[dict],
    workspace_path: str,
) -> str:
    """Plan generation callable outside the graph. Returns plan content."""
    config = DEPTH_CONFIG[depth]
    instructions = get_prompt("phase0-clarify.md")

    plan_content = await _generate_plan(
        topic, depth, budget, config, clarifications, instructions
    )

    _write_workspace_files(workspace_path, topic, budget, clarifications, plan_content)

    return plan_content


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_plan_system_prompt(depth: str, budget: int, config: dict) -> str:
    """Plan 生成專用 system prompt — 從 phase0-clarify.md 抽取出 Step 9 的必要規則。

    原本 `_generate_plan` 把整份 phase0-clarify.md（~2500 tokens）塞 system_msg，
    但 Step 1-8（澄清提問 / 題型分流 / perspective discovery / 搜尋策略流程等）
    和「只產出 plan markdown」無關 — 純 distractor。這裡 inline ~500 tokens 精簡版，
    只保留 plan 生成需要的格式範本 + 硬性規則。
    """
    subq_cap = config["subquestions"]
    iter_cap = config["iterations"]
    current_year = datetime.now().year
    return f"""你是深度研究規劃器。根據研究主題和任務書，產出 markdown 格式的完整研究計畫。

## 時間背景
- 當前年份：{current_year}
- freshness_sla 的「最新」和「近期」均指 {current_year} 年以內
- 搜尋 query 中的年份請使用 {current_year}（例如 "best tools {current_year}"、"最新 {current_year} 推薦"）
- 不得在 query 中使用 {current_year - 1} 或更早年份作為「最新」

## 必須覆蓋的八個要素
1. 題型分流（Adversarial / Temporal / Funnel / Multi-Stakeholder）
2. Query Enrichment（PICO + 來源優先級 + 防幻覺錨點）
3. 利害關係人視角（至少 1 advocate + 1 critic）
4. 子問題 DAG（含 facets、依賴、執行順序）— 數量上限 {subq_cap}
5. 搜尋策略（第一輪最小 query 集 + 後續觸發規則）
6. 預算分配（每子問題配額 × 迭代數），總預算 {budget} 次
7. 幻覺高風險區域（數字型 / 因果型 / 趨勢型 / 比較型）
8. 納入/排除標準

## 輸出格式（嚴格遵循）
```markdown
# 研究計畫

## 結構化 Header
- topic: {{主題}}
- mode: {{Adversarial / Temporal / Funnel / Multi-Stakeholder / 組合}}
- depth: {depth}
- budget: {budget}
- freshness_sla:
  - numeric: {{N}} 個月
  - policy: {{N}} 個月
  - background: {{N}} 個月
  - historical_exempt: true/false
- subquestions: {{N}} 個
- perspectives: {{N}} 個
- total_coverage_units: {{N}} 個（required: {{M}}）

## Query Enrichment
{{PICO + 來源優先級 + 防幻覺錨點}}

## 利害關係人視角
{{視角清單 + 各自關注和搜尋角度}}

## 子問題 DAG
{{子問題 + facets + 依賴 + 執行順序}}

## 搜尋策略
{{第一輪最小 query 集 + 後續觸發規則}}

## 預算分配
{{分配表}}

## 幻覺高風險區域
{{哪些論點需特別驗證}}

## 納入/排除標準
- 納入：{{語言、時間、地域、來源類型}}
- 排除：{{排除項}}
```

## 硬性規則
- 子問題數量 <= {subq_cap}
- 迭代輪次 <= {iter_cap}
- 每個子問題必須 advocate + critic 雙視角覆蓋
- 語言：繁體中文（技術術語保留原文）
- 研究深度：{depth}"""


async def _generate_plan(
    topic: str,
    depth: str,
    budget: int,
    config: dict,
    clarifications: list[dict],
    instructions: str,
    research_brief: str | None = None,
) -> str:
    """Call LLM to generate research plan.

    Args:
        research_brief: 統整後的研究任務書（優先使用）。
            若提供，作為主要 context；raw clarifications 僅作為補充附錄。
            若未提供（standalone 模式），退化為直接使用 raw clarifications。
        instructions: 保留參數向後相容，但實際不再整份塞入 system_msg
            （原本塞 phase0-clarify.md 全文 ~2500 tokens 是 distractor）。
    """
    _ = instructions  # 顯式棄用，保留簽名避免 caller 破壞

    if research_brief:
        # 統整後的 brief 已消除矛盾、補充隱含需求，品質更高
        context_section = f"\n\n## 研究任務書（統整後）\n\n{research_brief}"
    elif clarifications:
        # Fallback: standalone 模式沒有 brief，用 raw QA
        context_section = "\n\n## 使用者澄清資訊\n"
        for qa in clarifications:
            context_section += f"- **問：** {qa['question']}\n  **答：** {qa['answer']}\n"
    else:
        context_section = ""

    system_msg = SystemMessage(
        content=_build_plan_system_prompt(depth, budget, config) + context_section
    )
    human_msg = HumanMessage(content=f"研究主題：{topic}")
    # role="writer" — 規劃任務書，Claude Opus 主導
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[system_msg, human_msg],
        max_tokens=8192,
        temperature=0.3,
    )
    return response.content


def _write_workspace_files(
    workspace_path: str,
    topic: str,
    budget: int,
    clarifications: list[dict],
    plan_content: str,
) -> None:
    """Write all Phase 0 workspace files."""
    write_workspace_file(workspace_path, "phase0-plan.md", plan_content)
    init_source_registry(workspace_path)
    init_execution_log(workspace_path, topic, budget)
    init_gap_log(workspace_path)

    if clarifications:
        clarify_log = "# 澄清記錄\n\n"
        for i, qa in enumerate(clarifications, 1):
            clarify_log += f"## Q{i}: {qa['question']}\n\n{qa['answer']}\n\n"
        write_workspace_file(workspace_path, "clarifications.md", clarify_log)

    coverage_content = _generate_coverage_checklist(plan_content)
    write_workspace_file(workspace_path, "coverage.chk", coverage_content)


def _generate_coverage_checklist(plan: str) -> str:
    """Extract a basic coverage checklist skeleton from the plan text.

    Handles three common plan formats:
      1. "Q1: description" / "Q1：description" (English-style, e.g. "Q1: iPhone 錄音方案")
      2. "子問題 N: description" / "子問題 N：description" (Chinese with colon)
      3. Numbered list under "子問題 DAG": "1. **title**" or "1.  **title (Execution Order: N)**"

    Deduplicates by Q ID (first occurrence wins) and truncates long
    descriptions (e.g. lines containing embedded \\n ASCII diagrams).

    Last resort: parse "- subquestions: N 個" from the plan header and generate
    N placeholder sections so budget guard never gets a degenerate 1-SQ list.
    """
    lines = ["# Coverage Checklist\n"]

    # Format 1: Q1/Q2 style
    raw_matches = re.findall(r"\b(Q\d+)\s*[:：]\s*(.+)", plan)

    if not raw_matches:
        # Format 2: 子問題 N: style (possibly wrapped in **)
        sq_raw = re.findall(r"子問題\s*(\d+)\s*[:：]\s*([^*\n]+)", plan)
        if sq_raw:
            raw_matches = [(f"Q{n}", desc.strip()) for n, desc in sq_raw]

    if not raw_matches:
        # Format 3: numbered list items within "子問題 DAG" section
        dag_match = re.search(r"## 子問題 DAG\s*\n(.*?)(?=\n## |\Z)", plan, re.DOTALL)
        if dag_match:
            dag_content = dag_match.group(1)
            # Match "N.  **title (Execution Order: N)**" or "N. **title**"
            numbered = re.findall(r"^\s*(\d+)\.\s+\*{0,2}([^*\n(]+)", dag_content, re.MULTILINE)
            if numbered:
                raw_matches = [(f"Q{n}", title.strip()) for n, title in numbered]

    if raw_matches:
        seen: dict[str, str] = {}
        for qid, desc in raw_matches:
            if qid not in seen:
                # Truncate at embedded literal \n or after 80 chars
                desc_clean = desc.strip().split("\\n")[0][:80].strip()
                seen[qid] = desc_clean
        for qid, desc in seen.items():
            lines.append(f"\n## {qid}: {desc}")
            lines.append("- [ ] advocate — not_started")
            lines.append("- [ ] critic — not_started")
    else:
        # Last resort: read subquestions count from plan header
        sq_count_match = re.search(r"subquestions:\s*(\d+)", plan)
        sq_count = int(sq_count_match.group(1)) if sq_count_match else 1
        for i in range(1, sq_count + 1):
            lines.append(f"\n## Q{i}: (待 Phase 1a 填入)")
            lines.append("- [ ] advocate — not_started")
            lines.append("- [ ] critic — not_started")

    return "\n".join(lines) + "\n"
