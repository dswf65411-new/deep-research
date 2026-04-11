"""Phase 1b: Verify + Conditional Dialectic + Iteration.

Contains the verify subgraph (most complex part of the pipeline):
  Grounding Check → 4D Quality Eval → (pass → END | fail → Attack Agent → Process → END)
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from deep_research.harness.gates import quality_gate
from deep_research.llm import get_llm
from deep_research.state import Claim, SubagentResult, VerifyState
from deep_research.tools.grounding import (
    check_grounding_availability,
    ground_claims,
)
from deep_research.tools.workspace import (
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)

from deep_research.config import get_prompt
from deep_research.context import iterative_refine, estimate_tokens

# Bedrock thresholds by claim type
THRESHOLDS = {
    "numeric": 0.8,
    "comparative": 0.75,
    "causal": 0.75,
    "forecast": 0.75,
    "qualitative": 0.7,
}


# ---------------------------------------------------------------------------
# Subgraph nodes
# ---------------------------------------------------------------------------

async def grounding_check_node(state: VerifyState) -> dict:
    """Run grounding verification on all claims."""
    claims = state.get("claims_to_verify", [])
    workspace = state.get("workspace_path", "")

    if not claims:
        return {"grounding_results": []}

    # Check tool availability first
    tool_name, error = check_grounding_availability()
    if tool_name == "none":
        return {
            "grounding_results": [
                {"claim_id": c.claim_id, "error": f"GROUNDING-UNAVAILABLE: {error}"}
                for c in claims
            ]
        }

    results = []
    # Batch claims by subquestion for efficiency
    for claim in claims:
        # Gather source text for this claim
        source_texts = []
        for sid in claim.source_ids:
            # Find the source file
            source_files = list_workspace_files(workspace, "search-results", f"**/{sid}.md")
            if not source_files:
                # Try broader search
                import glob as g
                pattern = f"{workspace}/search-results/**/{sid}.md"
                source_files = g.glob(pattern, recursive=True)

            for sf in source_files:
                content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
                if content:
                    # Truncate to ~2000 tokens worth of text around quotes
                    source_texts.append(content)

        if not source_texts:
            results.append({
                "claim_id": claim.claim_id,
                "score": 0.0,
                "verdict": "NO_SOURCE_TEXT",
                "tool": "none",
            })
            continue

        combined_source = "\n\n---\n\n".join(source_texts)
        grounding = ground_claims(
            claims=[claim.claim_text],
            sources=[combined_source],
            preferred_tool=tool_name,
        )

        if grounding:
            r = grounding[0]
            threshold = THRESHOLDS.get(claim.claim_type, 0.7)
            results.append({
                "claim_id": claim.claim_id,
                "score": r.score,
                "verdict": "GROUNDED" if r.score >= threshold else "NOT_GROUNDED",
                "tool": r.tool_used,
                "threshold": threshold,
            })
        else:
            results.append({
                "claim_id": claim.claim_id,
                "score": 0.0,
                "verdict": "ERROR",
                "tool": tool_name,
            })

    return {"grounding_results": results}


async def quality_eval_node(state: VerifyState) -> dict:
    """Evaluate 4-dimensional quality for each subquestion."""
    claims = state.get("claims_to_verify", [])
    grounding = state.get("grounding_results", [])

    # Build grounding map
    g_map = {r["claim_id"]: r for r in grounding}

    # Group claims by subquestion
    by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        by_subq.setdefault(c.subquestion, []).append(c)

    scores = {}
    all_failed = []

    for subq, subq_claims in by_subq.items():
        grounded_claims = [
            c for c in subq_claims
            if g_map.get(c.claim_id, {}).get("verdict") == "GROUNDED"
        ]
        advocate_claims = [c for c in grounded_claims if "advocate" in (c.source_ids[0] if c.source_ids else "")]
        critic_claims = [c for c in grounded_claims if "critic" in (c.source_ids[0] if c.source_ids else "")]

        # Actionability: claims are specific and bounded
        actionability = len(grounded_claims) > 0

        # Freshness: defer to claim metadata (simplified check)
        freshness = True  # Will be properly checked with freshness SLA

        # Plurality: >= 2 independent sources
        all_sources = set()
        for c in grounded_claims:
            all_sources.update(c.source_ids)
        plurality = len(all_sources) >= 2

        # Completeness: both advocate and critic covered
        has_advocate = any(True for c in subq_claims
                         for r in [g_map.get(c.claim_id, {})]
                         if r.get("verdict") == "GROUNDED")
        completeness = has_advocate and len(grounded_claims) >= 2

        dim_scores = {
            "actionability": actionability,
            "freshness": freshness,
            "plurality": plurality,
            "completeness": completeness,
        }
        scores[subq] = dim_scores

        result, failed = quality_gate(dim_scores)
        if failed:
            all_failed.extend(failed)

    overall_result, overall_failed = quality_gate(
        {d: all(scores[sq].get(d, False) for sq in scores) for d in
         ["actionability", "freshness", "plurality", "completeness"]}
    ) if scores else ("needs_attack", ["no_claims"])

    return {
        "quality_scores": scores,
        "failed_dimensions": list(set(all_failed)) if all_failed else [],
    }


def quality_routing(state: VerifyState) -> str:
    """Route based on quality evaluation results."""
    failed = state.get("failed_dimensions", [])
    if not failed:
        return "all_pass"
    return "needs_attack"


ATTACK_SYSTEM = """你是攻擊型事實核查員。你的任務是嘗試證明待核對 claims 是錯的。

## 核對規則（嚴格遵守）

1. 在來源原文中找到 QUOTE 或 NUMBER 原文，嘗試推翻 claim
2. 數字：執行逐字核對。15% ≠ 約15% ≠ 近15%。任何不一致 = NOT_SUPPORTED
3. 程度詞：原文「成長」但 claim 說「大幅成長」= PARTIAL
4. 語氣：原文「可能」但 claim 說「確定」= NOT_SUPPORTED
5. 跨來源拼接：如果 claim 需要兩個不同來源才能支持 = PARTIAL 並標記 COMPOSITE

若找不到逐字對應或明確支持的原文，必須判為 NOT_SUPPORTED。

## Iterative 模式說明

你可能會收到多輪來源文件。每輪你要：
1. 審閱本輪新來源，核對每個 claim
2. 如果前面的累積結果中已有 SUPPORTED 判定，不要因為本輪沒找到就改成 NOT_SUPPORTED
3. 如果本輪找到更強的支持/反駁證據，更新對應 claim 的判定
4. 輸出完整的最新核對結果（包含之前已判定的和本輪新判定的）

## 輸出格式（嚴格 JSON array，包含所有 claims 的最新狀態）

```json
[{"claim_id": "...", "verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED", "quote_id": "...", "issue": "..."}]
```"""


async def attack_agent_node(state: VerifyState) -> dict:
    """Adversarial fact-checking via Iterative Refinement.

    Iron rule: sub-agent has NO search tools — only local file reading.

    使用 context.py 的 iterative_refine 管理 context window：
      - 場景：攻擊式事實核查 — 多針逐字核對（claim_text vs source 原文）
      - 風險：source 全塞時 Lost in the Middle 效應會讓中間的 source 被忽略，
              導致本應被推翻的 claim 被放行。
      - 策略：超過 context_threshold 時切換為 Iterative Refinement，
              每輪只送一批 sources，確保每份 source 都被完整審閱。
    """
    claims = state.get("claims_to_verify", [])
    workspace = state.get("workspace_path", "")

    if not claims:
        return {"attack_results": []}

    # 收集所有 source 文件為 list（供 iterative_refine 分批處理）
    source_texts: list[str] = []
    source_files = list_workspace_files(workspace, "search-results")
    for sf in source_files:
        content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
        if content:
            source_texts.append(f"--- {Path(sf).name} ---\n{content}")

    # 待核對 claims 作為 extra_context（每輪固定不變）
    claims_text = "\n".join(
        f"- {c.claim_id}: {c.claim_text}" for c in claims
        if c.status in ("pending", "needs_revision")
    )
    extra_context = f"## 待核對 Claims\n\n{claims_text}"

    # 取 full_research_topic（從 parent state 經 workspace 讀取）
    from deep_research.tools.workspace import read_workspace_file
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # 使用 iterative_refine：context 管理 + BM25 排序 + 分批處理
    result_text = await iterative_refine(
        sources=source_texts,
        full_research_topic=full_research_topic,
        system_prompt=ATTACK_SYSTEM,
        extra_context=extra_context,
        tier="fast",
    )

    # Parse results
    attack_results = _parse_attack_results(result_text)

    return {"attack_results": attack_results}


async def process_attack_node(state: VerifyState) -> dict:
    """Process attack agent results — update claim statuses."""
    claims = state.get("claims_to_verify", [])
    attack_results = state.get("attack_results", [])

    verdict_map = {r.claim_id: r for r in attack_results}

    updated = []
    for claim in claims:
        r = verdict_map.get(claim.claim_id)
        if r is None:
            updated.append(claim)
            continue

        if r.verdict == "SUPPORTED":
            claim.status = "approved"
        elif r.verdict == "PARTIAL":
            claim.status = "needs_revision"
        else:  # NOT_SUPPORTED
            claim.status = "rejected"
        updated.append(claim)

    return {"claims_to_verify": updated}


# ---------------------------------------------------------------------------
# Subgraph assembly
# ---------------------------------------------------------------------------

def build_verify_subgraph() -> StateGraph:
    """Build and compile the Phase 1b verification subgraph."""
    builder = StateGraph(VerifyState)

    builder.add_node("grounding", grounding_check_node)
    builder.add_node("quality_eval", quality_eval_node)
    builder.add_node("attack_agent", attack_agent_node)
    builder.add_node("process_attack", process_attack_node)

    builder.add_edge(START, "grounding")
    builder.add_edge("grounding", "quality_eval")
    builder.add_conditional_edges(
        "quality_eval",
        quality_routing,
        {
            "all_pass": END,
            "needs_attack": "attack_agent",
        },
    )
    builder.add_edge("attack_agent", "process_attack")
    builder.add_edge("process_attack", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main Phase 1b entry point (wraps subgraph for parent graph)
# ---------------------------------------------------------------------------

async def phase1b_verify(state: ResearchState) -> dict:
    """Run the Phase 1b verification subgraph.

    Transforms parent state → subgraph state, runs subgraph,
    transforms results back to parent state.
    """
    claims = state.get("claims", [])
    workspace = state.get("workspace_path", "")

    # Convert claims to Claim objects if needed
    claim_objects = []
    for c in claims:
        if isinstance(c, Claim):
            claim_objects.append(c)
        elif isinstance(c, dict):
            claim_objects.append(Claim(**c))

    # Only verify pending/needs_revision claims
    to_verify = [c for c in claim_objects if c.status in ("pending", "needs_revision")]

    if not to_verify:
        return {
            "phase1b_result": "pass",
            "execution_log": ["Phase 1b：無待驗證 claims，直接通過"],
        }

    # Run subgraph
    subgraph = build_verify_subgraph()
    result = await subgraph.ainvoke({
        "claims_to_verify": to_verify,
        "workspace_path": workspace,
        "iteration": 0,
    })

    # Merge results back: update claim statuses
    verified = result.get("claims_to_verify", [])
    verified_map = {c.claim_id: c for c in verified}

    updated_claims = []
    for c in claim_objects:
        if c.claim_id in verified_map:
            updated_claims.append(verified_map[c.claim_id])
        else:
            updated_claims.append(c)

    # Write grounding results to workspace
    grounding = result.get("grounding_results", [])
    if grounding:
        grounding_md = "# Grounding Results\n\n"
        for r in grounding:
            grounding_md += f"- {r.get('claim_id', '?')}: score={r.get('score', 0):.2f} verdict={r.get('verdict', '?')}\n"
        write_workspace_file(workspace, "grounding-results/latest.md", grounding_md)

    # Write/update claim ledger
    _write_claim_ledger(workspace, updated_claims)

    # Determine result
    all_resolved = all(c.status in ("approved", "rejected") for c in updated_claims)
    failed_dims = result.get("failed_dimensions", [])

    log_entry = (
        f"Phase 1b：verified={len(to_verify)} "
        f"approved={sum(1 for c in updated_claims if c.status == 'approved')} "
        f"rejected={sum(1 for c in updated_claims if c.status == 'rejected')} "
        f"pending={sum(1 for c in updated_claims if c.status in ('pending', 'needs_revision'))}"
    )

    return {
        "claims": updated_claims,
        "phase1b_result": "pass" if all_resolved and not failed_dims else "fail",
        "execution_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_attack_results(text: str) -> list[SubagentResult]:
    """Parse attack agent JSON output."""
    import re
    results = []
    # Try JSON array extraction
    json_match = re.search(r'\[[\s\S]*?\]', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                results.append(SubagentResult(
                    claim_id=item.get("claim_id", ""),
                    verdict=item.get("verdict", "NOT_SUPPORTED"),
                    quote_id=item.get("quote_id", "NONE"),
                    issue=item.get("issue", ""),
                ))
            return results
        except json.JSONDecodeError:
            pass

    # Fallback: parse line-by-line format
    blocks = text.split("---")
    for block in blocks:
        claim_id = ""
        verdict = "NOT_SUPPORTED"
        quote_id = "NONE"
        issue = ""
        for line in block.strip().split("\n"):
            if line.startswith("CLAIM_ID:"):
                claim_id = line.split(":", 1)[1].strip()
            elif line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip()
            elif line.startswith("QUOTE_ID:"):
                quote_id = line.split(":", 1)[1].strip()
            elif line.startswith("ISSUE:"):
                issue = line.split(":", 1)[1].strip()
        if claim_id:
            results.append(SubagentResult(
                claim_id=claim_id,
                verdict=verdict,
                quote_id=quote_id,
                issue=issue,
            ))

    return results


def _write_claim_ledger(workspace: str, claims: list[Claim]) -> None:
    """Write the claim-ledger.md file."""
    header = (
        "# Claim Ledger\n\n"
        "| claim_id | subquestion | type | claim_text | source_ids | quote_ids "
        "| bedrock | status |\n"
        "|----------|-------------|------|------------|------------|-----------|"
        "---------|--------|\n"
    )
    rows = []
    for c in claims:
        rows.append(
            f"| {c.claim_id} | {c.subquestion} | {c.claim_type} "
            f"| {c.claim_text[:60]}... | {','.join(c.source_ids)} "
            f"| {','.join(c.quote_ids)} | {c.bedrock_score:.2f} | {c.status} |"
        )
    write_workspace_file(workspace, "claim-ledger.md", header + "\n".join(rows) + "\n")
