"""Phase 1a: Search + Deep-Read + Verbatim Transcription.

Agent node — has search tools and operates autonomously within budget.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from deep_research.llm import get_llm
from deep_research.state import ResearchState
from deep_research.tools.search import get_search_tools
from deep_research.tools.workspace import (
    read_workspace_file,
    write_workspace_file,
    append_workspace_file,
)

from deep_research.config import get_prompt


async def phase1a_search(state: ResearchState) -> dict:
    """Execute search phase as a ReAct agent with direct API tools.

    Reads the research plan, generates queries, searches, deep-reads,
    and writes results to workspace files.
    """
    workspace_path = state["workspace_path"]
    plan = state.get("plan", "")
    depth = state.get("depth", "deep")
    budget = state.get("search_budget", 150)
    used = state.get("search_count", 0)
    iteration = state.get("iteration_count", 0)

    # Read phase instruction
    instructions = get_prompt("phase1a-search.md")

    # Read current workspace state
    coverage = read_workspace_file(workspace_path, "coverage.chk") or ""
    gap_log = read_workspace_file(workspace_path, "gap-log.md") or ""
    exec_log = read_workspace_file(workspace_path, "execution-log.md") or ""

    # Build the agent's system prompt
    system_prompt = f"""{instructions}

## 當前狀態
- 研究深度：{depth}
- 搜尋預算：已用 {used}/{budget}（剩餘 {budget - used}）
- 迭代輪次：第 {iteration + 1} 輪
- Workspace：{workspace_path}

## Coverage Checklist（當前）
{coverage}

## Gap Log（當前）
{gap_log}

## 重要規則
1. 每次搜尋後更新 execution-log.md 的搜尋計數
2. 所有深讀結果存為 workspace/search-results/Q{{n}}/S{{id}}.md
3. 更新 source-registry.md
4. QUOTE 和 NUMBER 必須有唯一 ID
5. URL 必須先用 urlhealth 驗證再深讀
6. 搜尋計數不得超過剩餘預算 {budget - used}

語言：繁體中文（技術術語保留原文）。

完成搜尋後，輸出一個 JSON 摘要：
```json
{{"searches_used": N, "sources_found": N, "quotes_extracted": N}}
```
"""

    task_msg = f"""請根據研究計畫執行第 {iteration + 1} 輪搜尋。

研究計畫摘要：
{plan}

請按照 phase1a-search.md 的流程執行搜尋、深讀、逐字抄錄，並將結果寫入 workspace。
"""

    # Create the search agent with direct HTTP API tools
    llm = get_llm(tier="strong", max_tokens=16384, temperature=0.2)

    tools = get_search_tools()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=task_msg)]},
    )

    # Parse the agent's output to extract search count
    last_msg = result["messages"][-1].content if result.get("messages") else ""
    searches_used = _extract_search_count(last_msg, budget - used)

    return {
        "search_count": used + searches_used,
        "execution_log": [
            f"Phase 1a 第 {iteration + 1} 輪：搜尋 {searches_used} 次（累計 {used + searches_used}/{budget}）"
        ],
    }


def _extract_search_count(agent_output: str, max_allowed: int) -> int:
    """Parse the agent's JSON summary to get search count."""
    import json
    import re

    # Try to find JSON block in agent output
    json_match = re.search(r'\{[^{}]*"searches_used"\s*:\s*(\d+)[^{}]*\}', agent_output)
    if json_match:
        count = int(json_match.group(1))
        return min(count, max_allowed)

    # Fallback: count based on heuristic
    return min(5, max_allowed)
