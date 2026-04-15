"""Heavy Mode — n=3 parallel rollouts with selector-picks-best.

Adapted from Tongyi DeepResearch's ParallelMuse pattern
(`WebAgent/ParallelMuse/compressed_reasoning_aggregation.py:26-68`):

  - REPORT_PROMPT compresses each trajectory into a single-page report.
  - INTEGRATE_PROMPT selects the single most credible answer; MERGING IS
    FORBIDDEN. The pitfall the Tongyi authors call out is that merged
    answers become "X, Y, or Z" non-answers — we inherit that rule.

Trigger (in graph.py, `route_after_review`):
  revision_count has already hit MAX_REVISIONS **and** the critic still says
  accept=False. Instead of shipping the latest draft unchanged, fan out to
  n=3 rollouts and pick the best per-section.

Cost model: 3× the report-generation LLM calls plus 1 selector call per
section — no new search / grounding cost, so budget impact is bounded and
predictable. The node is latched via ``heavy_mode_triggered`` to prevent
re-entry (it is a last-resort pass, not a loop).

Why this is adjacent to phase2 rather than phase1:
  Tongyi's original Heavy Mode spawns n=3 *full agent trajectories*. In our
  pipeline that would mean re-searching the web three times (3× Serper
  spend, 3× Bedrock grounding). We would only be willing to pay that on
  truly high-stakes queries; phase2-level heavy mode is the cheapest
  adaptation that still buys multi-rollout selection on the output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.llm import safe_ainvoke_chain
from deep_research.prompts_shared import FOCUSED_EXEC_PROMPT
from deep_research.state import Claim, ResearchState
from deep_research.tools.workspace import read_workspace_file, write_workspace_file

logger = logging.getLogger(__name__)


# Number of parallel rollouts. Tongyi uses 3; staying at 3 keeps the cost
# proportional to the Critic-revise loop (which also runs at most ~3 LLM
# passes) rather than ballooning into a 5-10× rewrite.
HEAVY_MODE_N = 3

# Temperatures for the n rollouts. 0.3 preserves the existing draft's spine,
# 0.7 is the writer's usual mid-range, 1.0 forces more exploration. Must have
# exactly HEAVY_MODE_N entries.
_ROLLOUT_TEMPERATURES: tuple[float, ...] = (0.3, 0.7, 1.0)
assert len(_ROLLOUT_TEMPERATURES) == HEAVY_MODE_N, (
    "heavy_mode: _ROLLOUT_TEMPERATURES must match HEAVY_MODE_N"
)


_REWRITE_SYSTEM = FOCUSED_EXEC_PROMPT + """You are rewriting a single section of a deep-research report. The section was drafted but the critic flagged concrete issues. Produce a new version that addresses those issues while staying faithful to the approved claims.

## Hard rules
1. Do NOT invent new facts. You may cite only claim_ids present in the "Approved claims" block — no other claim_ids are allowed anywhere in your output.
2. Every factual sentence must end with its supporting claim_id in square brackets, e.g. [Q1-C3]. A sentence without a claim_id must be general framing only (transition, context, restating the subquestion).
3. The section must directly answer its subquestion. If the draft wandered off-topic, pull it back.
4. Length: at least 200 chars of prose below the header, no mid-sentence truncation.
5. Keep the same section header (##/### level and the subquestion it addresses) as the original draft.

## Output format
Return ONLY the rewritten section markdown. No preamble, no explanation, no fences. Your first line is the section header; your last line is the final prose sentence.
"""


_SELECTOR_SYSTEM = """You are a judge picking the single best rewrite of a research-report section. You are given a subquestion, a list of critic issues the rewrite must address, and n numbered candidate rewrites.

## Selection rules
1. Pick exactly ONE candidate. Do NOT merge or blend candidates — that is explicitly forbidden (inherited from Tongyi ParallelMuse INTEGRATE_PROMPT).
2. Prefer the candidate that most directly fixes the listed critic issues. A beautifully written candidate that ignores the issues loses to a plainer one that addresses them.
3. Penalize candidates that cite claim_ids outside the approved list, or that have fewer than 30% of the SQ's approved claims cited.
4. Penalize candidates that wander off the subquestion.

## Output — STRICT JSON, no markdown:
{
  "winner_index": <integer in [0, n-1]>,
  "rationale": "<≤40 words naming the decisive factor>"
}
"""


async def heavy_mode_rollout(state: ResearchState) -> dict:
    """Rewrite critic-rejected sections as n=3 rollouts, pick best per-section.

    Assumes ``report-sections/*.md`` contains the most recent (rejected)
    draft — that draft becomes rollout #0 and the remaining HEAVY_MODE_N-1
    rollouts are produced by a rewrite LLM at different temperatures. Then a
    selector LLM picks the best candidate per-section and overwrites the
    canonical file.

    On any infrastructure failure (LLM outage, parse miss, empty sections
    dir) the node is a no-op — it logs the reason and returns with
    ``heavy_mode_triggered=True`` so the graph still advances to phase3.
    """
    workspace = state["workspace_path"]
    verdict = state.get("review_verdict") or {}
    per_sq_issues: dict[str, list[str]] = dict(verdict.get("per_sq_issues", {}) or {})

    sections_dir = Path(workspace) / "report-sections"
    section_files = sorted(sections_dir.glob("*.md")) if sections_dir.exists() else []
    if not section_files:
        logger.info("heavy_mode: no sections on disk; skipping")
        return {
            "heavy_mode_triggered": True,
            "execution_log": ["Heavy mode skipped: report-sections/ empty"],
        }

    claims_raw = state.get("claims", [])
    claim_objs = [c if isinstance(c, Claim) else Claim(**c) for c in claims_raw]
    approved_by_sq: dict[str, list[Claim]] = {}
    for c in claim_objs:
        if getattr(c, "status", None) == "approved":
            approved_by_sq.setdefault(c.subquestion, []).append(c)

    # Only rewrite sections the critic actually flagged. If per_sq_issues is
    # empty (critic rejected without pointing at a SQ) we rewrite every
    # section, since the critic couldn't localise the problem.
    targets = list(per_sq_issues.keys()) if per_sq_issues else [
        _infer_sq_from_filename(f.name) for f in section_files
    ]
    targets = [t for t in targets if t]  # drop "" from filenames we can't parse

    winners: list[str] = []
    no_op_sqs: list[str] = []
    rewrite_tasks: list[asyncio.Task] = []
    rewrite_meta: list[tuple[str, Path, list[str]]] = []

    for sq in targets:
        section_file = _find_section_file(section_files, sq)
        if section_file is None:
            # Section flagged but file missing — nothing to rewrite, log and move on.
            no_op_sqs.append(sq)
            continue
        issues = per_sq_issues.get(sq, ["critic did not localise; rewrite for coherence"])
        rewrite_tasks.append(
            asyncio.create_task(
                _generate_rollouts(
                    section_text=section_file.read_text(encoding="utf-8"),
                    sq=sq,
                    issues=issues,
                    approved=approved_by_sq.get(sq, []),
                )
            )
        )
        rewrite_meta.append((sq, section_file, issues))

    rollouts_per_section = await asyncio.gather(*rewrite_tasks, return_exceptions=True)

    for (sq, section_file, issues), result in zip(rewrite_meta, rollouts_per_section, strict=True):
        if isinstance(result, Exception) or not result:
            # Rewrite failed — leave the original draft in place, log, continue.
            logger.warning("heavy_mode: rewrite failed for %s (%r); keeping draft", sq, result)
            winners.append(f"{sq}: rewrite-failed→kept-draft")
            continue

        candidates: list[str] = result  # type: ignore[assignment]
        winner_idx = await _select_best(sq=sq, issues=issues, candidates=candidates)
        winner_text = candidates[winner_idx]
        rel_path = section_file.relative_to(Path(workspace)).as_posix()
        write_workspace_file(workspace, rel_path, winner_text)
        winners.append(f"{sq}:rollout#{winner_idx}")

    # Log the rollout decisions for auditing — matches gpt-researcher's
    # review-log pattern but in its own file so the two are distinguishable.
    log_lines = [
        "",
        "",
        "## Heavy Mode rollout",
        f"- rollouts per section: {HEAVY_MODE_N} (temperatures {list(_ROLLOUT_TEMPERATURES)})",
        f"- sections rewritten: {len(winners) or 0}",
        f"- winners: {winners or '(none)'}",
    ]
    if no_op_sqs:
        log_lines.append(f"- flagged-but-missing sections (kept as-is): {no_op_sqs}")
    from deep_research.tools.workspace import append_workspace_file
    append_workspace_file(workspace, "review-log.md", "\n".join(log_lines) + "\n")

    return {
        "heavy_mode_triggered": True,
        "execution_log": [
            f"Heavy mode: rewrote {len(winners)} section(s), winners={winners}"
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _generate_rollouts(
    *,
    section_text: str,
    sq: str,
    issues: list[str],
    approved: list[Claim],
) -> list[str]:
    """Produce HEAVY_MODE_N rewrites of one section, one per temperature.

    Rollout 0 is the existing draft unchanged (so a no-op winner keeps the
    revise-loop's last output). Rollouts 1..n-1 go through the rewrite LLM
    at the remaining temperatures.
    """
    approved_block = "\n".join(
        f"- [{c.claim_id}] {c.claim_text}" for c in approved
    ) or "(no approved claims for this SQ — rewrite must keep prose general and avoid claim_ids)"
    issues_block = "\n".join(f"- {i}" for i in issues)

    user_msg = f"""## Subquestion
{sq}

## Critic issues to address
{issues_block}

## Approved claims (only these claim_ids may appear in your output)
{approved_block}

## Previous draft (rewrite this)
{section_text}
"""

    tasks: list[asyncio.Task] = []
    for temperature in _ROLLOUT_TEMPERATURES[1:]:
        tasks.append(
            asyncio.create_task(
                _safe_rewrite(
                    system_prompt=_REWRITE_SYSTEM,
                    user_msg=user_msg,
                    temperature=temperature,
                )
            )
        )
    rewrites = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[str] = [section_text]  # rollout 0 = current draft
    for r in rewrites:
        if isinstance(r, Exception) or not isinstance(r, str) or not r.strip():
            # Surface the failure by substituting the original draft — the
            # selector still gets HEAVY_MODE_N candidates and just sees a dup.
            candidates.append(section_text)
        else:
            candidates.append(r)
    return candidates


async def _safe_rewrite(
    *, system_prompt: str, user_msg: str, temperature: float
) -> str:
    """One rewrite call. Returns the raw content string; raises on LLM error."""
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ],
        max_tokens=4096,
        temperature=temperature,
    )
    return (getattr(response, "content", "") or "").strip()


async def _select_best(
    *, sq: str, issues: list[str], candidates: list[str]
) -> int:
    """Ask the selector LLM which candidate best addresses the critic issues.

    Falls back to candidate 0 (the original draft) when the selector LLM
    fails or returns an out-of-range index. Keeping the pre-heavy-mode draft
    on failure is the safe default: we at least get the revise-loop output.
    """
    numbered = "\n\n".join(
        f"### Candidate {i}\n{c}" for i, c in enumerate(candidates)
    )
    user_msg = f"""## Subquestion
{sq}

## Critic issues
{chr(10).join(f'- {i}' for i in issues)}

## Candidates (n={len(candidates)})
{numbered}
"""
    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[
                SystemMessage(content=_SELECTOR_SYSTEM),
                HumanMessage(content=user_msg),
            ],
            max_tokens=512,
            temperature=0.0,
        )
        raw = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.warning("heavy_mode: selector LLM failed for %s (%s); keeping draft", sq, e)
        return 0

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        logger.info("heavy_mode: selector produced no JSON for %s; keeping draft", sq)
        return 0
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        logger.info("heavy_mode: selector JSON parse failed for %s; keeping draft", sq)
        return 0

    idx = parsed.get("winner_index")
    if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
        logger.info(
            "heavy_mode: selector returned out-of-range index %r for %s; keeping draft",
            idx, sq,
        )
        return 0
    return idx


_SQ_FROM_FILENAME = re.compile(r"^(q\d+)_section\.md$", re.IGNORECASE)


def _infer_sq_from_filename(filename: str) -> str:
    """``q1_section.md`` → ``Q1``; unrecognised names return ``""``."""
    m = _SQ_FROM_FILENAME.match(filename)
    return m.group(1).upper() if m else ""


def _find_section_file(files: list[Path], sq: str) -> Path | None:
    """Match ``Q1`` to a file like ``q1_section.md`` (case-insensitive)."""
    target = sq.lower()
    for f in files:
        if f.name.lower().startswith(f"{target}_"):
            return f
    return None
