"""Phase 0: Clarify + Research Planning.

Functions:
  generate_questions  — LLM generates clarifying questions (up to max_questions)
  validate_answers    — check answer format/completeness, return missing indices
  judge_clarity       — independent LLM (clean context) decides if topic is clear enough
  phase0_plan         — generates research plan (graph node)
  phase0_plan_standalone — same logic, callable outside graph (for skill mode)
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import get_prompt
from deep_research.context import synthesize_research_topic
from deep_research.llm import get_llm
from deep_research.state import ResearchState
from deep_research.tools.workspace import (
    create_workspace,
    init_execution_log,
    init_gap_log,
    init_source_registry,
    write_workspace_file,
)

# Depth → budget mapping
DEPTH_CONFIG = {
    "quick": {"budget": 30, "subquestions": "1-2", "iterations": 1},
    "standard": {"budget": 60, "subquestions": "2-5", "iterations": 2},
    "deep": {"budget": 150, "subquestions": "5-10", "iterations": 5},
}

# Default question limit per round
DEFAULT_MAX_QUESTIONS = 10


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
    llm = get_llm(tier="strong", max_tokens=4096, temperature=0.3)

    context = ""
    if existing_clarifications:
        context = "\n\n## 已取得的澄清資訊（前幾輪的結果）\n"
        for i, qa in enumerate(existing_clarifications, 1):
            context += f"{i}. **問：** {qa['question']}\n   **答：** {qa['answer']}\n"
        context += "\n請不要重複問已經回答過的問題。只問新的、之前沒涵蓋到的面向。\n"

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
    response = await llm.ainvoke([system_msg, human_msg])

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
# LLM Judge: is the topic clear enough?
# ---------------------------------------------------------------------------

async def judge_clarity(
    topic: str,
    clarifications: list[dict],
) -> tuple[bool, str, list[str]]:
    """Independent LLM judge (clean context) evaluates if clarification is sufficient.

    This judge has NO access to the questioner LLM's reasoning or plan.
    It only sees the topic + Q&A pairs and evaluates from scratch.

    Returns:
        (is_clear, reasoning, suggested_questions)
        - is_clear: True if ready for deep research
        - reasoning: why it's clear or not
        - suggested_questions: if not clear, what else to ask
    """
    llm = get_llm(tier="strong", max_tokens=4096, temperature=0.0)

    qa_text = ""
    for i, qa in enumerate(clarifications, 1):
        qa_text += f"{i}. 問：{qa['question']}\n   答：{qa['answer']}\n\n"

    system_msg = SystemMessage(content="""你是獨立的研究品質評審（Judge）。你與提問的 LLM 完全獨立 — 你沒有看到它的推理過程，你只看到研究主題和問答記錄。

你的角色是嚴格評估：「根據目前的資訊，能否產出高品質的深度研究？」

## 嚴格檢查清單（每一項都要逐一評估）

1. **目的明確性：** 使用者為什麼做這個研究？目的是否具體到可以判斷研究成功還是失敗？
   - ❌ 「了解 AI」 → 太模糊
   - ✅ 「為公司的客服系統選擇 AI 框架」 → 有具體場景

2. **範圍界定：** 研究的邊界在哪？是否有可能無限發散？
   - ❌ 沒有時間範圍、地域範圍、技術範圍
   - ✅ 「2024 年後的開源框架，排除付費 SaaS」

3. **產出格式：** 使用者期望什麼形式的結果？
   - ❌ 不知道要比較表、推薦結論、還是客觀呈現
   - ✅ 「需要比較表 + 最終推薦 + 風險評估」

4. **成功標準：** 什麼叫「好的研究結果」？怎麼判斷好壞？
   - ❌ 沒有評估維度或標準
   - ✅ 「從效能、成本、社群活躍度三個維度比較」

5. **互相矛盾：** 問答中是否有自相矛盾的地方？
   - 例如：說「不限預算」但又說「最便宜的」

6. **隱含假設：** 是否有未驗證的假設？
   - 例如：假設某技術適用但沒確認場景

7. **受眾與決策者：** 研究結果給誰看？誰做決策？
   - 這影響報告的深度和表達方式

8. **時效性要求：** 需要多新的資料？有沒有截止日期？

9. **已知約束：** 有什麼硬性限制（技術、預算、團隊規模、合規）？

10. **深度 vs 廣度偏好：** 全面概覽還是聚焦特定面向？

## 評判規則

- 不需要全部完美，但核心 4 項（目的、範圍、產出、成功標準）必須明確
- 如果有互相矛盾，必須追問
- 如果有模糊定義（如「好用」「效能好」），必須追問具體標準
- 寧可多問一輪也不要帶著模糊的需求開始研究
- 生成的追問問題要具體、有針對性，說明「為什麼需要知道」

## 回覆格式（嚴格 JSON）

如果足夠清楚：
{"clear": true, "reasoning": "逐項說明為什麼每個核心項目都已滿足", "suggested_questions": []}

如果還需要澄清：
{"clear": false, "reasoning": "指出具體哪些項目不足、為什麼不足", "suggested_questions": ["具體的追問問題1（為什麼需要知道：原因）", "具體的追問問題2（為什麼需要知道：原因）"]}""")

    human_msg = HumanMessage(content=f"""## 研究主題
{topic}

## 已完成的澄清問答
{qa_text if qa_text else "（尚未進行任何澄清）"}""")

    response = await llm.ainvoke([system_msg, human_msg])

    try:
        text = response.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            return True, "parse_error, defaulting to clear", []
    except (json.JSONDecodeError, AttributeError):
        return True, "parse_error, defaulting to clear", []

    return (
        parsed.get("clear", True),
        parsed.get("reasoning", ""),
        parsed.get("suggested_questions", []),
    )


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

    plan_content = await _generate_plan(
        topic, depth, budget, config, clarifications, instructions
    )

    # 統整研究任務書（topic + refs + clarifications → full_research_topic）
    # 這份任務書會作為所有後續 phase 的 fixed context
    full_research_topic = await synthesize_research_topic(topic, refs, clarifications)

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

async def _generate_plan(
    topic: str,
    depth: str,
    budget: int,
    config: dict,
    clarifications: list[dict],
    instructions: str,
) -> str:
    """Call LLM to generate research plan."""
    clarify_context = ""
    if clarifications:
        clarify_context = "\n\n## 使用者澄清資訊\n"
        for qa in clarifications:
            clarify_context += f"- **問：** {qa['question']}\n  **答：** {qa['answer']}\n"

    llm = get_llm(tier="strong", max_tokens=8192, temperature=0.3)

    system_msg = SystemMessage(content=f"""{instructions}

你是深度研究規劃器。根據使用者的研究主題，生成完整的研究計畫。

研究深度：{depth}
搜尋預算：{budget} 次
子問題數量上限：{config['subquestions']}
迭代上限：{config['iterations']} 輪
{clarify_context}

輸出格式：嚴格遵循 phase0-clarify.md 中 Step 9 定義的 markdown 格式。
語言：繁體中文（技術術語保留原文）。
""")

    human_msg = HumanMessage(content=f"研究主題：{topic}")
    response = await llm.ainvoke([system_msg, human_msg])
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
    """Extract a basic coverage checklist skeleton from the plan text."""
    lines = ["# Coverage Checklist\n"]
    q_matches = re.findall(r"(Q\d+)\s*[:：]\s*(.+)", plan)
    if q_matches:
        for qid, desc in q_matches:
            lines.append(f"\n## {qid}: {desc.strip()}")
            lines.append("- [ ] advocate — not_started")
            lines.append("- [ ] critic — not_started")
    else:
        lines.append("\n## Q1: (待 Phase 1a 填入)")
        lines.append("- [ ] advocate — not_started")
        lines.append("- [ ] critic — not_started")

    return "\n".join(lines) + "\n"
