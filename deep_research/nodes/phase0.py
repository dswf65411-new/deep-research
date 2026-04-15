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
from deep_research.harness.url_validator import (
    annotate_invalid,
    invalid_items,
    validate_plan_text,
)
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

# Whisper P3-1 — split clarify into staged rounds.
# Round 1 asks only the top scope/priority questions so the user isn't buried
# under ten ambiguous items; the Judge-driven follow-ups (round 2+) go deep
# on whatever still isn't clear.
FIRST_ROUND_CORE_CAP = 3

# QA compaction threshold: when exceeded, older rounds are compressed into a topic list
QA_COMPACT_THRESHOLD = 15
QA_KEEP_LATEST = 5


def _compact_clarifications(qas: list[dict]) -> str:
    """Format prior clarifications as context — compact when too many.

    Anti-pattern guard (LLM focused-task principle):
      - Scenario: when multi-round clarification accumulates 20+ QA entries,
              dumping them all into LLM context wastes attention — the model
              only needs the "don't re-ask" signal, not every verbatim answer.
      - Strategy: QA <= 15 expanded in full; > 15 collapses older N-5 entries
              into topic lines (Q truncated to 30 chars, A to 20 chars), while
              the latest 5 stay fully expanded for deeper follow-up.
    """
    if not qas:
        return ""

    if len(qas) <= QA_COMPACT_THRESHOLD:
        body = "\n".join(
            f"{i}. **Q:** {qa['question']}\n   **A:** {qa['answer']}"
            for i, qa in enumerate(qas, 1)
        )
        return (
            "\n\n## Clarifications collected so far (from prior rounds)\n"
            f"{body}\n"
            "\nDo not re-ask questions already answered. Only ask new aspects "
            "not previously covered.\n"
        )

    # Compaction: older entries collapsed to topic lines, latest 5 kept in full
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
            f"{i}. **Q:** {qa['question']}\n   **A:** {qa['answer']}"
        )

    return (
        "\n\n## Clarifications collected so far (from prior rounds)\n"
        f"### Older rounds ({len(old_qas)} entries, collapsed to topic list)\n"
        + "\n".join(old_lines)
        + f"\n\n### Latest {len(recent_qas)} full QA entries\n"
        + "\n".join(recent_lines)
        + "\n\nDo not re-ask topics already answered (including the collapsed list above). "
        "Only ask new aspects not previously covered.\n"
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

    # P3-1: round 1 is deliberately narrow — just enough to nail scope &
    # priority — so the user isn't overwhelmed. The Judge's follow-ups in
    # later rounds handle depth. ``effective_max`` becomes the cap for both
    # the prompt and the downstream slice.
    if round_num == 1:
        effective_max = min(max_questions, FIRST_ROUND_CORE_CAP)
        round_note = (
            f"\nThis is clarification round 1 — the *scoping* round. Ask the "
            f"{effective_max} single most important questions that lock down "
            "research scope, decision priority, and success criteria. Do NOT "
            "ask for tech-stack details, budget, team size, or other narrow "
            "constraints yet — those are for later rounds if the judge deems "
            "them needed.\n"
        )
    else:
        effective_max = max_questions
        round_note = (
            f"\nThis is clarification round {round_num} — the *depth* round. "
            "Round 1 already locked down scope and priority. Now dig into the "
            "specific aspects the judge flagged as still unclear, or ask "
            "concrete follow-ups that unblock downstream planning.\n"
        )

    system_msg = SystemMessage(content=f"""You are the clarification module of a deep-research planner.

Given the user's research topic, decide whether clarification is needed. Ask generously — clarification is the single most important lever for research quality.

Only return an empty `questions` array when the topic is fully clear on every aspect.
{round_note}
Generate at most {effective_max} questions. Each question must state "why do we need to know this".

Consider the following aspects (only ask what's needed):
1. Research purpose: making a decision? writing a report? learning? solving a concrete problem?
2. Expected output: comparison table? recommendation? objective presentation? in-depth technical analysis?
3. Scope boundaries: what is included/excluded? time range? geographic constraints?
4. Background knowledge: what does the user already know? which options have been ruled out?
5. Specific preferences: tech stack constraints, budget range, team size?
6. Success criteria: what kind of result would be most valuable to the user?
7. Stakeholders: who is the audience for the result? who needs to be convinced?
8. Recency: how fresh must the data be? is there a deadline?
9. Depth vs breadth: broad overview, or deep analysis of a few aspects?
10. Known constraints: any known limitations?
{context}

Reply format (strict JSON):
{{"questions": ["Question 1 (why we need to know: reason)", "Question 2 (why we need to know: reason)"], "reasoning": "why this round asks these"}}

If no further questions are needed:
{{"questions": [], "reasoning": "Topic is fully clear because..."}}""")

    human_msg = HumanMessage(content=f"Research topic: {topic}")
    # role="writer" — planning / question generation, Claude Opus leads
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

    questions = parsed.get("questions", [])[:effective_max]
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
        "name": "research purpose",
        "required": True,
        "question": "Is the user's research purpose concrete enough to tell success from failure?",
        "pass_example": "\"Pick an AI framework for our customer-service system\" → concrete scenario and goal",
        "fail_example": "\"Understand AI\" → too vague, no way to judge success",
    },
    {
        "id": "scope",
        "name": "scope",
        "required": True,
        "question": "Are the research boundaries (time, geography, technical scope) explicit? Could it diverge indefinitely?",
        "pass_example": "\"Open-source frameworks after 2024, excluding paid SaaS\" → clear boundaries",
        "fail_example": "No boundary statement at all → could diverge without end",
    },
    {
        "id": "output_format",
        "name": "output format",
        "required": True,
        "question": "What form of research result does the user expect?",
        "pass_example": "\"Need a comparison table + final recommendation + risk assessment\" → deliverables are clear",
        "fail_example": "Unclear whether a comparison table, a recommendation, or objective presentation is expected",
    },
    {
        "id": "success_criteria",
        "name": "success criteria",
        "required": True,
        "question": "Are there explicit evaluation dimensions or criteria for judging the result?",
        "pass_example": "\"Compare along performance, cost, and community activity\" → has an evaluation framework",
        "fail_example": "No evaluation dimensions or criteria → no way to tell good from bad",
    },
    {
        "id": "consistency",
        "name": "consistency",
        "required": True,
        "question": "Are all answers in the Q&A record logically consistent, with no self-contradiction?",
        "pass_example": "All answers logically consistent, no contradictions",
        "fail_example": "Says \"no budget limit\" but also \"cheaper the better\" → contradictory",
    },
    {
        "id": "constraints",
        "name": "known constraints",
        "required": False,
        "question": "Has the user stated hard constraints (tech stack, budget, team size, compliance)?",
        "pass_example": "\"Must support Python, budget under US$1000\" → constraints are clear",
        "fail_example": "No constraints mentioned (may exist but not stated)",
    },
    {
        "id": "depth_breadth",
        "name": "depth vs breadth",
        "required": False,
        "question": "Does the user prefer a broad overview or deep analysis of specific aspects?",
        "pass_example": "\"Deep dive into the technical details of the top 3\" → preference is clear",
        "fail_example": "No preference stated → unclear whether to go broad or deep",
    },
]


def _build_rubric_system_prompt(dims: list[dict] | None = None) -> str:
    """Build the rubric evaluation system prompt for a given dimension subset.

    Passing None uses all 7 dims (backward compatible); production calls should
    pass a subset of 3-4 dims per group to avoid Lost-in-the-Middle attention
    drop on dims in the middle of a longer list.
    """
    target = dims if dims is not None else CLARITY_DIMENSIONS
    dims_text = ""
    for i, dim in enumerate(target, 1):
        req_label = "required" if dim["required"] else "bonus"
        dims_text += f"""
### {i}. {dim['name']} ({dim['id']}) [{req_label}]
Judge: {dim['question']}
- PASS: {dim['pass_example']}
- FAIL: {dim['fail_example']}
"""

    dim_ids_json = ", ".join(
        f'{{"id": "{d["id"]}", "verdict": "...", "evidence": "...", "reason": "...", "question": "..."}}'
        for d in target
    )

    return f"""You are an independent research-quality Judge. You are fully independent of the asking LLM — you do not see its reasoning, only the research topic and the Q&A record.

## Evidence-Anchored rules (hard rules)
1. Every dimension's verdict must quote the user's original text as evidence.
2. If the user's text contains no evidence supporting "sufficient" → the verdict must be FAIL.
3. Do not infer user intent; judge only based on information that was explicitly provided.
4. Evaluate each dimension independently — do not let one dimension's verdict influence another.
5. Any FAIL dimension must produce a targeted follow-up question (including "why we need to know").

## Evaluation dimensions ({len(target)} in this batch; judge each independently)
{dims_text}
## Reply format (strict JSON, no other text)
{{"dimensions": [{dim_ids_json}]}}

Fields per dimension:
- id: dimension ID (must match above)
- verdict: "PASS" or "FAIL"
- evidence: direct quote from user's original text supporting the verdict (for FAIL, state what is missing)
- reason: one-line justification
- question: for FAIL, the follow-up question (with "why we need to know: reason"); for PASS, empty string ""
"""


# Group A: required dimensions (core, first 4)
# Group B: optional dimensions (last 3)
# Two separate calls, each judging only 3-4 dims, to avoid LiM degrading the
# middle dimensions' judgment quality.
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
    """Evaluate one group of dims (3-4). Returns a list of dim verdict dicts; empty on failure."""
    try:
        llm = get_llm(tier=tier, max_tokens=4096, temperature=0.0, provider=provider)
        system_msg = SystemMessage(content=_get_rubric_prompt(group))
        human_msg = HumanMessage(content=(
            f"## Research topic\n{topic}\n\n"
            f"## Completed clarification Q&A\n{qa_text or '(no clarification yet)'}"
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

    Group A: required dimensions (4 core)
    Group B: optional dimensions (3)
    Two concurrent calls, each judging only 3-4 dims. Substantially reduces
    Lost-in-the-Middle impact on middle dims vs judging all 7 at once.
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
    lines = [f"## Clarification sufficiency evaluation ({n_providers} models: {provider_names})\n"]

    for dim in CLARITY_DIMENSIONS:
        dim_id = dim["id"]
        result = aggregated.get(dim_id, {
            "verdict": "FAIL", "pass_count": 0, "fail_count": 0, "total": 0,
            "reasons": [], "questions": [],
        })
        req_label = "required" if dim["required"] else "bonus"
        verdict = result["verdict"]
        marker = "[PASS]" if verdict == "PASS" else "[FAIL]"
        votes = f"{result['pass_count']}/{result['total']}"

        lines.append(f"  {marker} {dim['name']} ({dim_id}) [{req_label}] — {votes}")
        # Show first reason for context
        if result["reasons"]:
            lines.append(f"        {result['reasons'][0]}")

    lines.append("")
    if is_clear:
        lines.append("Conclusion: all required dimensions passed.")
        if optional_fails:
            names = ", ".join(d["name"] for d in optional_fails)
            lines.append(f"  ({names} unclear, but does not block the research)")
    else:
        fail_names = ", ".join(d["name"] for d in required_fails)
        lines.append(
            f"Conclusion: {len(required_fails)} required dimensions failed ({fail_names}); "
            f"targeted follow-up recommended."
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
        qa_text += f"{i}. Q: {qa['question']}\n   A: {qa['answer']}\n\n"

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
            return (True, "(all evaluation models failed; defaulting to pass)", [])

    providers_used = [r["provider"] for r in valid_results]

    # Layer 3: Aggregate votes + Adaptive Escalation
    aggregated = _aggregate_panel_votes(valid_results)
    return _build_judge_verdict(aggregated, providers_used)


# ---------------------------------------------------------------------------
# Graph node: phase0_plan
# ---------------------------------------------------------------------------

async def phase0_plan(state: ResearchState) -> dict:
    """Generate research plan from topic + clarifications (graph node).

    Also integrates topic + refs + clarifications → full_research_topic,
    which acts as the stable context for the whole research pipeline
    (prompt prefix caching friendly).
    """
    topic = state["topic"]
    depth = state.get("depth", "deep")
    config = DEPTH_CONFIG[depth]
    budget = state.get("search_budget", config["budget"])
    clarifications = state.get("clarifications", [])
    refs = state.get("refs", [])

    instructions = get_prompt("phase0-clarify.md")
    workspace_path = create_workspace(topic)

    # Step 1: integrate into a research brief (topic + refs + clarifications → structured brief).
    # The brief resolves contradictions and surfaces implicit requirements — much higher
    # quality than raw QA pairs.
    full_research_topic = await synthesize_research_topic(topic, refs, clarifications)

    # Step 2: generate the plan using the integrated research brief (not the raw clarifications).
    plan_content = await _generate_plan(
        topic, depth, budget, config, clarifications, instructions,
        research_brief=full_research_topic,
    )

    # Step 2b: verify that every arxiv ID / URL the plan cites really exists (guard against LLM hallucination).
    # The failure workspace from 2026-04-14 had 4 future-dated arxiv IDs in the plan; this step catches them at the source.
    plan_content, validation_log = await _validate_and_annotate_plan(
        plan_content, workspace_path
    )

    # Write workspace files
    _write_workspace_files(workspace_path, topic, budget, clarifications, plan_content)

    # Also write full_research_topic to the workspace for reference
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
        "execution_log": [
            f"Phase 0 complete: workspace={workspace_path}, research brief integrated",
            *validation_log,
        ],
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
    """System prompt dedicated to plan generation — extracts the essential rules from
    Step 9 of phase0-clarify.md.

    Originally `_generate_plan` stuffed the entire phase0-clarify.md (~2500 tokens)
    into the system message, but Steps 1-8 (clarifying questions, topic triage,
    perspective discovery, search strategy flow, etc.) are unrelated to the task of
    producing plan markdown — pure distractors. This inlines a ~500-token distilled
    version keeping only the format template plus hard rules that plan generation needs.
    """
    subq_cap = config["subquestions"]
    iter_cap = config["iterations"]
    current_year = datetime.now().year
    return f"""You are a deep-research planner. Given the research topic and brief, produce a complete research plan in markdown.

## Time context
- Current year: {current_year}
- "latest" and "recent" in freshness_sla refer to within {current_year}
- Use {current_year} for the year in search queries (e.g. "best tools {current_year}", "top tools {current_year} recommendation")
- Do not use {current_year - 1} or earlier years as "latest" in queries

## Nine required elements
1. **Premise audit** — list the 2-3 assumptions embedded in the topic itself that, if wrong, would invalidate the whole research. For each: (a) restate the assumption explicitly, (b) propose 1 concrete search query that attacks it, (c) describe what evidence would falsify it. This section must appear BEFORE subquestions so the DAG can absorb any falsifier findings.
2. Topic triage (Adversarial / Temporal / Funnel / Multi-Stakeholder)
3. Query Enrichment (PICO + source priority + anti-hallucination anchors)
4. Stakeholder perspectives (at least 1 advocate + 1 critic)
5. Subquestion DAG (facets, dependencies, execution order) — at most {subq_cap}
6. Search strategy (minimum query set for round 1 + triggers for later rounds)
7. Budget allocation (quota per subquestion × iterations), total budget {budget}
8. High-hallucination-risk areas (numeric / causal / trend / comparative)
9. Inclusion / exclusion criteria

## Output format (strict)
```markdown
# Research Plan

## Structured Header
- topic: {{topic}}
- mode: {{Adversarial / Temporal / Funnel / Multi-Stakeholder / combination}}
- depth: {depth}
- budget: {budget}
- freshness_sla:
  - numeric: {{N}} months
  - policy: {{N}} months
  - background: {{N}} months
  - historical_exempt: true/false
- subquestions: {{N}}
- perspectives: {{N}}
- total_coverage_units: {{N}} (required: {{M}})

## Premise audit
{{2-3 assumptions baked into the topic, each with: (a) assumption restated, (b) falsifier query, (c) what evidence would invalidate}}

## Query Enrichment
{{PICO + source priority + anti-hallucination anchors}}

## Stakeholder perspectives
{{perspective list + each one's concerns and search angles}}

## Subquestion DAG
{{subquestions + facets + dependencies + execution order}}

## Search Strategy
{{minimum query set for round 1 + triggers for later rounds}}

## Budget Allocation
{{allocation table}}

## High-hallucination-risk areas
{{which claims need extra verification}}

## Inclusion / Exclusion criteria
- Include: {{language, time, geography, source types}}
- Exclude: {{exclusions}}
```

## Hard rules
- Subquestions <= {subq_cap}
- Iterations <= {iter_cap}
- Every subquestion must cover both advocate and critic perspectives
- Language: English (keep technical terms in their original form)
- Research depth: {depth}"""


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
        research_brief: The integrated research brief (preferred).
            If provided, used as the primary context; raw clarifications act as a
            supplementary appendix only. If not provided (standalone mode), falls
            back to using raw clarifications directly.
        instructions: Kept for backward compatibility, but the full text is no
            longer stuffed into system_msg (dumping the whole phase0-clarify.md
            ~2500 tokens was a distractor).
    """
    _ = instructions  # explicitly deprecated; signature kept to avoid breaking callers

    if research_brief:
        # The integrated brief already resolves contradictions and surfaces implicit
        # requirements — higher quality than raw QA.
        context_section = f"\n\n## Research brief (integrated)\n\n{research_brief}"
    elif clarifications:
        # Fallback: standalone mode has no brief — use raw QA
        context_section = "\n\n## User clarifications\n"
        for qa in clarifications:
            context_section += f"- **Q:** {qa['question']}\n  **A:** {qa['answer']}\n"
    else:
        context_section = ""

    system_msg = SystemMessage(
        content=_build_plan_system_prompt(depth, budget, config) + context_section
    )
    human_msg = HumanMessage(content=f"Research topic: {topic}")
    # role="writer" — brief / plan generation, Claude Opus leads
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[system_msg, human_msg],
        max_tokens=8192,
        temperature=0.3,
    )
    return response.content


async def _validate_and_annotate_plan(
    plan_content: str,
    workspace_path: str,
) -> tuple[str, list[str]]:
    """Check every arxiv ID / URL the planner produced and tag hallucinated ones.

    Runs against:
      - https://export.arxiv.org/api/query (arxiv IDs)
      - HEAD requests for any URL in the plan

    Known-bad references are replaced with a ``[REMOVED: hallucinated ...]``
    marker so downstream phases can't accidentally use them as seeds.
    Unreachable (timeout / 500) URLs are left alone — absence of evidence
    is not evidence of absence.

    Validation results are written to
    ``<workspace>/plan-url-validation.json`` and a human-readable summary
    is appended to ``execution-log.md``. Cache lives in
    ``<workspace>/.arxiv-validation-cache.json`` (per-workspace; cross-
    workspace caching is P3 scope).
    """
    from pathlib import Path as _Path

    cache_path = _Path(workspace_path) / ".arxiv-validation-cache.json"
    log_entries: list[str] = []

    try:
        validation = await validate_plan_text(plan_content, cache_path=cache_path)
    except Exception as exc:
        logger.warning("phase0 plan validation skipped due to error: %s", exc)
        log_entries.append(f"Phase 0 plan validation skipped ({exc})")
        return plan_content, log_entries

    bad = invalid_items(validation)
    annotated = annotate_invalid(plan_content, validation)

    try:
        write_workspace_file(
            workspace_path,
            "plan-url-validation.json",
            json.dumps(
                {
                    "arxiv": validation.get("arxiv", {}),
                    "urls": validation.get("urls", {}),
                    "summary": bad,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
    except Exception:
        logger.exception("phase0: failed to write plan-url-validation.json")

    if bad["hallucinated_arxiv"] or bad["hallucinated_urls"]:
        log_entries.append(
            f"⚠️ Phase 0 plan validation: removed {len(bad['hallucinated_arxiv'])} hallucinated arxiv IDs, "
            f"{len(bad['hallucinated_urls'])} hallucinated URLs"
        )
        if bad["hallucinated_arxiv"]:
            log_entries.append(
                "  - hallucinated arxiv: " + ", ".join(bad["hallucinated_arxiv"][:10])
            )
        if bad["hallucinated_urls"]:
            log_entries.append(
                "  - hallucinated URLs: " + ", ".join(bad["hallucinated_urls"][:5])
            )
    else:
        arxiv_count = len(validation.get("arxiv", {}))
        url_count = len(validation.get("urls", {}))
        if arxiv_count or url_count:
            log_entries.append(
                f"✅ Phase 0 plan validation: {arxiv_count} arxiv IDs all validation passed, "
                f"0 hallucinated out of {url_count} URLs ({len(bad['unreachable_urls'])} temporarily unreachable)"
            )

    return annotated, log_entries


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
        clarify_log = "# Clarifications\n\n"
        for i, qa in enumerate(clarifications, 1):
            clarify_log += f"## Q{i}: {qa['question']}\n\n{qa['answer']}\n\n"
        write_workspace_file(workspace_path, "clarifications.md", clarify_log)

    coverage_content = _generate_coverage_checklist(plan_content)
    write_workspace_file(workspace_path, "coverage.chk", coverage_content)


def _generate_coverage_checklist(plan: str) -> str:
    """Extract a basic coverage checklist skeleton from the plan text.

    Handles these common plan formats:
      1. "Q1: description" / "Q1: description" (English-style)
      2. "Subquestion N: description" (English with colon)
      3. Numbered list under "Subquestion DAG": "1. **title**" or "1.  **title (Execution Order: N)**"

    Deduplicates by Q ID (first occurrence wins) and truncates long
    descriptions (e.g. lines containing embedded \\n ASCII diagrams).

    Last resort: parse "- subquestions: N" from the plan header and generate
    N placeholder sections so budget guard never gets a degenerate 1-SQ list.
    """
    lines = ["# Coverage Checklist\n"]

    # Format 1: Q1/Q2 style
    raw_matches = re.findall(r"\b(Q\d+)\s*:\s*(.+)", plan)

    if not raw_matches:
        # Format 2: "Subquestion N:" style (possibly wrapped in **)
        sq_raw = re.findall(
            r"subquestion\s*(\d+)\s*:\s*([^*\n]+)",
            plan,
            re.IGNORECASE,
        )
        if sq_raw:
            raw_matches = [(f"Q{n}", desc.strip()) for n, desc in sq_raw]

    if not raw_matches:
        # Format 3: numbered list items within "Subquestion DAG" section
        dag_match = re.search(
            r"##\s+subquestion\s+dag\s*\n(.*?)(?=\n## |\Z)",
            plan,
            re.DOTALL | re.IGNORECASE,
        )
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
            lines.append(f"\n## Q{i}: (to be filled by Phase 1a)")
            lines.append("- [ ] advocate — not_started")
            lines.append("- [ ] critic — not_started")

    return "\n".join(lines) + "\n"
