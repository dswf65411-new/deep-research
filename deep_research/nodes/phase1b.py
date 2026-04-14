"""Phase 1b: Verify + Conditional Dialectic + Iteration.

Contains the verify subgraph (most complex part of the pipeline):
  Grounding Check → 4D Quality Eval → (pass → END | fail → Attack Agent → Process → END)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from deep_research.harness.gates import quality_gate
from deep_research.llm import get_available_providers, get_llm, get_provider, safe_ainvoke, safe_ainvoke_chain
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

async def _llm_ground_one_claim(
    claim: "Claim",
    source_texts: list[str],
    semaphore: asyncio.Semaphore,
) -> dict:
    """LLM-based grounding for a single claim (used as fallback).

    Runs concurrently under the provided semaphore to avoid rate-limit spikes.
    Returns a grounding result dict compatible with ground_claims() output.
    """
    combined = "\n\n---\n\n".join(source_texts[:3])
    # Truncate to avoid huge context — grounding needs local context, not full doc
    if len(combined) > 6000:
        combined = combined[:6000] + "\n...[truncated]"

    threshold = THRESHOLDS.get(claim.claim_type, 0.7)

    prompt = f"""You are a fact checker. Determine whether the following Claim is directly supported by the source text.

## Important notes
- The Claim may be written in Traditional Chinese (or another target language), while the source text may be in English or another language. This is expected.
- Judge across languages: if the Claim is a faithful description of the source text (even when phrased in a different language), treat it as GROUNDED.
- Example: English source "Best speech-to-text app" -> Traditional Chinese Claim with equivalent meaning -> treat as supported.
- Do not lower the score solely because the languages differ.

## Source text
{combined}

## Claim to verify (type: {claim.claim_type})
{claim.claim_text}

## Scoring rubric
- 1.0: the text contains an explicit sentence that directly supports this claim (even if phrased in a different language)
- 0.7-0.9: the text provides sufficient indirect support or allows reasonable derivation
- 0.5-0.69: the text provides partial, related support
- 0.0-0.49: no support can be found in the text, or the text contradicts the claim

Only output JSON, no other text: {{"score": 0.0 to 1.0}}"""

    async with semaphore:
        try:
            response = await safe_ainvoke_chain(
                role="verifier",
                messages=[HumanMessage(content=prompt)],
                max_tokens=60,
                temperature=0.0,
            )
            text = response.content if hasattr(response, "content") else str(response)
            m = re.search(r'\{[^}]+\}', text)
            if m:
                data = json.loads(m.group())
                score = min(max(float(data.get("score", 0.5)), 0.0), 1.0)
            else:
                score = 0.5
        except Exception as exc:
            logger.warning("LLM grounding fallback failed for %s: %s", claim.claim_id, exc)
            # Conservative: don't penalize when LLM itself fails
            score = 0.5

    verdict = "GROUNDED" if score >= threshold else "NOT_GROUNDED"
    return {
        "claim_id": claim.claim_id,
        "score": score,
        "verdict": verdict,
        "tool": "llm",
        "threshold": threshold,
    }


async def _march_blind_recheck(
    claim: "Claim",
    source_texts: list[str],
    alternate_provider: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Independent grounding verification by a different-vendor LLM (MARCH pattern).

    MARCH (arxiv 2603.24579) observes that LLMs from the same family share
    correlated hallucinations — a single model agreeing with its own claim is
    weak evidence. The blind checker sees the same claim + source but NOT the
    solver's reasoning, and must independently judge. If primary and blind
    checker disagree on a GROUNDED verdict, the claim is downgraded.

    Uses the same rubric as ``_llm_ground_one_claim`` so that only the model
    identity differs. Returns a dict with ``{claim_id, score, verdict,
    tool: "march-<provider>"}`` or None if the alternate LLM itself failed
    (we do not penalize the claim for alternate-provider outages).
    """
    combined = "\n\n---\n\n".join(source_texts[:3])
    if len(combined) > 6000:
        combined = combined[:6000] + "\n...[truncated]"

    threshold = THRESHOLDS.get(claim.claim_type, 0.7)

    prompt = f"""You are an independent fact checker. Judge whether the Claim is directly supported by the Source text. You have not seen any other model's reasoning about this claim — judge from source only.

## Notes
- The Claim may be in Traditional Chinese while the source is in English (or vice versa). Cross-language matches count as grounded if meaning is faithful.

## Source text
{combined}

## Claim (type: {claim.claim_type})
{claim.claim_text}

## Rubric
- 1.0: source explicitly states the claim
- 0.7-0.9: source indirectly supports or allows reasonable derivation
- 0.5-0.69: partial support
- 0.0-0.49: no support or contradicted

Only output JSON: {{"score": 0.0 to 1.0}}"""

    async with semaphore:
        try:
            llm = get_llm(tier="strong", max_tokens=60, temperature=0.0, provider=alternate_provider)
            response = await safe_ainvoke(llm, [HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            m = re.search(r'\{[^}]+\}', text)
            if not m:
                return None
            score = min(max(float(json.loads(m.group()).get("score", 0.5)), 0.0), 1.0)
        except Exception as exc:
            logger.info("MARCH recheck via %s failed for %s: %s", alternate_provider, claim.claim_id, exc)
            return None

    verdict = "GROUNDED" if score >= threshold else "NOT_GROUNDED"
    return {
        "claim_id": claim.claim_id,
        "score": score,
        "verdict": verdict,
        "tool": f"march-{alternate_provider}",
        "threshold": threshold,
    }


def _strip_html_for_grounding(content: str) -> str:
    """Strip HTML tags and decode entities from raw fetched content for grounding.

    Raw files (_raw.md) may contain full HTML with CSS/JS noise. The LLM grounding
    prompt is truncated at 6000 chars, so CSS/JS preamble would crowd out actual text.
    This strips HTML tags so the LLM sees plain text, improving grounding accuracy.
    """
    import re as _re
    import html as _html_module
    # Remove script and style blocks entirely
    content = _re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', content, flags=_re.DOTALL | _re.IGNORECASE)
    # Remove all other HTML tags
    content = _re.sub(r'<[^>]+>', ' ', content)
    # Decode HTML entities (e.g. &amp; → &, &lt; → <)
    content = _html_module.unescape(content)
    # Collapse whitespace
    content = _re.sub(r'[ \t]+', ' ', content)
    content = _re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


async def _gather_claim_sources(
    claim: "Claim", workspace: str
) -> list[str]:
    """Collect source text for grounding.

    Strategy:
    1. Try *_raw.md (full fetched content). If it contains HTML, strip the tags
       so the LLM grounding sees clean text rather than CSS/JS noise.
    2. Fall back to *.md (metadata + extracted quotes) if raw is unavailable.

    The search-results directory has two files per source:
      S001_raw.md  — full fetched page content (may be raw HTML, up to 25K chars)
      S001.md      — metadata + extracted verified quotes (clean, < 2K chars)
    """
    import glob as _glob

    source_texts: list[str] = []
    for sid in claim.source_ids:
        # Prefer _raw.md (full content, HTML-stripped); fall back to .md (metadata)
        for suffix in (f"{sid}_raw.md", f"{sid}.md"):
            source_files = list_workspace_files(workspace, "search-results", f"**/{suffix}")
            if not source_files:
                pattern = f"{workspace}/search-results/**/{suffix}"
                source_files = _glob.glob(pattern, recursive=True)
            if source_files:
                for sf in source_files:
                    raw = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
                    if not raw:
                        continue
                    # Strip HTML if this is a raw fetch file
                    if suffix.endswith("_raw.md") and "<html" in raw.lower():
                        cleaned = _strip_html_for_grounding(raw)
                        # Fall back to metadata if stripping left too little content
                        content = cleaned if len(cleaned.strip()) >= 200 else raw
                    else:
                        content = raw
                    if content and len(content.strip()) >= 200:
                        source_texts.append(content)
                break  # found raw — skip metadata fallback
    return source_texts


async def grounding_check_node(state: VerifyState) -> dict:
    """Run grounding verification on all claims."""
    claims = state.get("claims_to_verify", [])
    workspace = state.get("workspace_path", "")

    if not claims:
        return {"grounding_results": []}

    # Always use LLM grounding.
    # CLI tools (bedrock/minicheck/nemo) test with English→English but our pipeline
    # generates Chinese claim_text against English sources, giving ~0.00 scores.
    # LLM grounding handles cross-language matching correctly.
    tool_name, error = "none", "forced LLM path for cross-language support"
    use_llm_fallback = True
    logger.info("grounding_check_node: using LLM grounding (Gemini fallback, supports cross-language)")

    if use_llm_fallback:
        # LLM fallback path: first gather source texts concurrently, then run LLM grounding concurrently
        sem = asyncio.Semaphore(5)  # at most 5 concurrent LLM calls, to avoid 429

        # Gather source texts for all claims concurrently
        source_tasks = [_gather_claim_sources(claim, workspace) for claim in claims]
        all_sources = await asyncio.gather(*source_tasks)

        # Run LLM grounding concurrently for claims that have sources
        llm_tasks = []
        no_source_indices: list[int] = []
        grounding_indices: list[int] = []

        for i, (claim, srcs) in enumerate(zip(claims, all_sources)):
            if not srcs:
                no_source_indices.append(i)
            else:
                grounding_indices.append(i)
                llm_tasks.append(_llm_ground_one_claim(claim, srcs, sem))

        llm_results = await asyncio.gather(*llm_tasks)

        # Merge results in original order
        results: list[dict] = [None] * len(claims)  # type: ignore[list-item]
        for idx in no_source_indices:
            results[idx] = {
                "claim_id": claims[idx].claim_id,
                "score": 0.0,
                "verdict": "NO_SOURCE_TEXT",
                "tool": "none",
            }
        for task_i, claim_i in enumerate(grounding_indices):
            results[claim_i] = llm_results[task_i]

        logger.info(
            "LLM grounding fallback complete: %d claims, %d have source, %d NO_SOURCE_TEXT",
            len(claims), len(grounding_indices), len(no_source_indices),
        )

        # ── MARCH blind recheck: second opinion from a different vendor ─────
        # For every claim the primary model marked GROUNDED, an independent
        # LLM from a different family re-verifies. Disagreement downgrades.
        # Skipped when only one provider has an API key.
        providers = get_available_providers()
        alternates = [p for p in providers if p != get_provider()]
        if alternates and any(r and r.get("verdict") == "GROUNDED" for r in results):
            alt = alternates[0]
            logger.info("MARCH blind recheck starting via alternate provider: %s", alt)
            recheck_tasks = []
            recheck_indices: list[int] = []
            for i, r in enumerate(results):
                if not r or r.get("verdict") != "GROUNDED":
                    continue
                try:
                    claim_i = next(idx for idx, c in enumerate(claims) if c.claim_id == r["claim_id"])
                except StopIteration:
                    continue
                srcs = all_sources[claim_i]
                if not srcs:
                    continue
                recheck_indices.append(i)
                recheck_tasks.append(_march_blind_recheck(claims[claim_i], srcs, alt, sem))

            recheck_results = await asyncio.gather(*recheck_tasks, return_exceptions=True)

            downgrade_count = 0
            for i, rr in zip(recheck_indices, recheck_results):
                if isinstance(rr, BaseException) or rr is None:
                    results[i]["march_status"] = "unchecked"
                    continue
                results[i]["march_score"] = rr["score"]
                results[i]["march_provider"] = alt
                if rr["verdict"] == "GROUNDED":
                    results[i]["march_status"] = "agreed"
                else:
                    results[i]["march_status"] = "disagreed"
                    results[i]["verdict"] = "NOT_GROUNDED"
                    results[i]["downgrade_reason"] = f"MARCH disagreement (alt={alt} score={rr['score']:.2f})"
                    downgrade_count += 1

            logger.info(
                "MARCH recheck complete: %d claims rechecked, %d downgraded due to disagreement",
                len(recheck_indices), downgrade_count,
            )
        else:
            logger.info(
                "MARCH recheck skipped (providers=%s, alternates=%s)",
                providers, alternates,
            )

        return {"grounding_results": [r for r in results if r is not None]}

    # ── CLI grounding path (bedrock / minicheck / nemo) ──────────────────────
    results = []
    for claim in claims:
        source_texts = await _gather_claim_sources(claim, workspace)

        if not source_texts:
            results.append({
                "claim_id": claim.claim_id,
                "score": 0.0,
                "verdict": "NO_SOURCE_TEXT",
                "tool": "none",
            })
            continue

        combined_source = "\n\n---\n\n".join(source_texts)
        grounding_res = ground_claims(
            claims=[claim.claim_text],
            sources=[combined_source],
            preferred_tool=tool_name,
        )

        if grounding_res:
            r = grounding_res[0]
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
    """Evaluate quality for each subquestion.

    Execution order (order matters):
    1. Grounding verdict -> set claim.status (approved / rejected / pending)
    2. Relevance filter -> LLM judges whether approved claims are on-topic; off-topic -> rejected
    3. Dim-score computation -> based on filtered approved claims (reflects true quality)
    """
    claims = state.get("claims_to_verify", [])
    grounding = state.get("grounding_results", [])
    workspace = state.get("workspace_path", "")

    # Build grounding map
    g_map = {r["claim_id"]: r for r in grounding}

    # ── Step 1: Set initial status from grounding verdict ─────────────────
    # Must happen before relevance check so we only check grounded (approved) claims.
    for c in claims:
        g = g_map.get(c.claim_id, {})
        verdict = g.get("verdict", "")
        if verdict == "GROUNDED":
            c.status = "approved"
        elif verdict in ("NOT_GROUNDED", "ERROR"):
            c.status = "rejected"
        # NO_SOURCE_TEXT → stays pending (attack agent may have source context)

    # ── Step 2: Relevance filter ──────────────────────────────────────────
    # Grounding only checks "does the text exist in source?"; not "does it answer the SQ?".
    # Reject off-topic approved claims (addresses, bios, boilerplate that happen to be in source).
    irrelevant_ids = await _run_relevance_checks(claims, workspace)
    for c in claims:
        if c.claim_id in irrelevant_ids:
            logger.info(
                "quality_eval: relevance filter rejected off-topic claim %s: %s",
                c.claim_id,
                c.claim_text[:60],
            )
            c.status = "rejected"

    # ── Step 3: Compute quality dimensions (post-relevance-filter) ────────
    by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        by_subq.setdefault(c.subquestion, []).append(c)

    scores = {}
    all_failed = []

    for subq, subq_claims in by_subq.items():
        # Dimension scores based on claims that passed both grounding AND relevance
        approved_claims = [c for c in subq_claims if c.status == "approved"]

        # Actionability: at least one approved claim
        actionability = len(approved_claims) > 0

        # Freshness: simplified (real SLA check deferred to future iteration)
        freshness = True

        # Plurality: >= 2 independent sources among approved claims
        all_sources: set[str] = set()
        for c in approved_claims:
            all_sources.update(c.source_ids)
        plurality = len(all_sources) >= 2

        # Completeness: >= 2 approved claims for this SQ
        completeness = len(approved_claims) >= 2

        # Relevance dimension: False if any claim in this SQ was off-topic
        sq_irrelevant = irrelevant_ids & {c.claim_id for c in subq_claims}
        relevance = len(sq_irrelevant) == 0

        dim_scores = {
            "actionability": actionability,
            "freshness": freshness,
            "plurality": plurality,
            "completeness": completeness,
            "relevance": relevance,  # informational; not enforced by quality_gate
        }
        scores[subq] = dim_scores

        # quality_gate enforces only the 4 primary dimensions
        result, failed = quality_gate(
            {k: v for k, v in dim_scores.items() if k != "relevance"}
        )
        if failed:
            all_failed.extend(failed)

    quality_gate(
        {d: all(scores[sq].get(d, False) for sq in scores) for d in
         ["actionability", "freshness", "plurality", "completeness"]}
    ) if scores else ("needs_attack", ["no_claims"])

    return {
        "claims_to_verify": claims,
        "quality_scores": scores,
        "failed_dimensions": list(set(all_failed)) if all_failed else [],
    }


def quality_routing(state: VerifyState) -> str:
    """Route based on quality evaluation results."""
    failed = state.get("failed_dimensions", [])
    if not failed:
        return "all_pass"
    return "needs_attack"


# ---------------------------------------------------------------------------
# Relevance filter helpers — LLM judges if claim answers the subquestion
# ---------------------------------------------------------------------------

_RELEVANCE_SYSTEM = """You are a research-claim relevance reviewer. For each claim, determine whether it actually answers the research subquestion.

## Criteria
- relevant=true: the claim's information directly or indirectly helps answer the subquestion (features, performance, pricing, limits, comparisons, etc.)
- relevant=false: the claim is **completely unrelated** to the subquestion (company physical addresses, phone numbers, customer-support promo copy, unrelated product ads, author bios, cookie notices, unrelated NAS/hardware setup tutorials, etc.)

## Output format (strict JSON array)
[{"claim_id": "...", "relevant": true/false, "reason": "one-line explanation"}]

Only output JSON, no other text."""


def _extract_sq_descriptions(checklist_text: str) -> dict[str, str]:
    """Parse the Q{n} -> subquestion-description mapping from coverage.chk.

    coverage.chk format:
        ## Q1: Market overview and initial screening
        ## Q2: zh-TW speech-to-text accuracy...
    """
    if not checklist_text:
        return {}
    result: dict[str, str] = {}
    for m in re.finditer(r'^##\s+(Q\d+):\s+(.+)', checklist_text, re.MULTILINE):
        sq_id = m.group(1)
        desc = m.group(2).strip()
        if sq_id not in result and desc:
            result[sq_id] = desc
    return result


async def _batch_relevance_check(
    sq_id: str,
    sq_description: str,
    approved_claims: list[Claim],
    semaphore: asyncio.Semaphore,
) -> dict[str, bool]:
    """Check relevance of approved claims for one subquestion.

    Runs one LLM call per SQ (batch, not per-claim).
    Returns {claim_id: is_relevant}.
    Conservative on failure: all claims assumed relevant (no false rejections).
    """
    if not approved_claims:
        return {}

    claims_text = "\n".join(
        f"- {c.claim_id}: {c.claim_text}"
        for c in approved_claims
    )

    user_msg = f"""## Research subquestion
{sq_id}: {sq_description}

## Claims (judge one by one whether each answers the subquestion)
{claims_text}

Every claim must be given a verdict:"""

    async with semaphore:
        try:
            response = await safe_ainvoke_chain(
                role="verifier",
                messages=[
                    SystemMessage(content=_RELEVANCE_SYSTEM),
                    HumanMessage(content=user_msg),
                ],
                max_tokens=600,
                temperature=0.0,
            )
            text = response.content if hasattr(response, "content") else str(response)
            m = re.search(r'\[[\s\S]*?\]', text)
            if not m:
                logger.warning("relevance check: no JSON array found for %s", sq_id)
                return {c.claim_id: True for c in approved_claims}
            parsed = json.loads(m.group())
            return {
                r["claim_id"]: bool(r.get("relevant", True))
                for r in parsed
                if isinstance(r, dict) and "claim_id" in r
            }
        except Exception as exc:
            logger.warning("_batch_relevance_check failed for %s: %s", sq_id, exc)
            # Conservative: on any error, assume all relevant (avoid false rejections)
            return {c.claim_id: True for c in approved_claims}


async def _run_relevance_checks(
    claims: list[Claim],
    workspace: str,
) -> set[str]:
    """Run relevance checks for all approved claims.

    Reads coverage.chk to get SQ descriptions.
    Skips SQs with no meaningful description (too risky to reject without context).
    Returns set of claim_ids that are off-topic.
    """
    checklist_text = ""
    if workspace:
        checklist_text = read_workspace_file(workspace, "coverage.chk") or ""
    sq_descriptions = _extract_sq_descriptions(checklist_text)

    if not sq_descriptions:
        logger.debug("relevance check: no SQ descriptions found, skipping")
        return set()

    # Group approved claims by subquestion
    approved_by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        if c.status == "approved":
            approved_by_subq.setdefault(c.subquestion, []).append(c)

    if not approved_by_subq:
        return set()

    # Only run for SQs with a real description (>= 5 chars)
    sem = asyncio.Semaphore(3)  # max 3 concurrent LLM calls
    tasks = []
    for sq, subq_claims in approved_by_subq.items():
        desc = sq_descriptions.get(sq, "")
        if not desc or len(desc) < 5:
            logger.debug("relevance check: no description for %s, skipping", sq)
            continue
        tasks.append(_batch_relevance_check(sq, desc, subq_claims, sem))

    if not tasks:
        return set()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    irrelevant_ids: set[str] = set()
    for res in results:
        if isinstance(res, Exception):
            logger.warning("relevance check task error: %s", res)
            continue
        for claim_id, is_relevant in res.items():
            if not is_relevant:
                irrelevant_ids.add(claim_id)

    if irrelevant_ids:
        logger.info("relevance filter: %d off-topic claims rejected", len(irrelevant_ids))

    return irrelevant_ids


ATTACK_SYSTEM = """You are an adversarial fact checker. Your task is to try to prove the claims under review are wrong.

## Verification rules (strictly follow)

1. Locate the QUOTE or NUMBER in the source text and try to refute the claim
2. Numbers: do verbatim comparison. 15% != about 15% != nearly 15%. Any inconsistency = NOT_SUPPORTED
3. Degree words: source says "growth" but claim says "significant growth" = PARTIAL
4. Tone: source says "may" but claim says "certainly" = NOT_SUPPORTED
5. Cross-source stitching: if the claim needs two different sources to be supported = PARTIAL with COMPOSITE tag

If you cannot find a verbatim match or a source that explicitly supports the claim, you must rule NOT_SUPPORTED.

## Iterative mode notes

You may receive multiple rounds of source documents. In each round you should:
1. Review the new sources for this round and verify each claim
2. If the accumulated results already contain a SUPPORTED verdict, do not downgrade it to NOT_SUPPORTED merely because this round did not find supporting evidence
3. If this round turns up stronger supporting/refuting evidence, update that claim's verdict accordingly
4. Output the complete latest verification results (including previously decided claims and newly decided ones)

## Output format (strict JSON array containing the latest status of all claims)

```json
[{"claim_id": "...", "verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED", "quote_id": "...", "issue": "..."}]
```"""


async def attack_agent_node(state: VerifyState) -> dict:
    """Adversarial fact-checking via Iterative Refinement — per-subquestion parallel.

    Iron rule: sub-agent has NO search tools — only local file reading.

    Anti-pattern guard (LLM-focus principle):
      - Before: stuff all claims + all search-results into a single LLM for review.
              Claims span multiple subquestions and sources span multiple subquestions;
              the LLM sees a pile of unrelated claims that don't match the source it has,
              which triggers the "too many tasks + context noise" double Lost-in-the-Middle problem.
      - Now: group by claim.subquestion. Each group reviews only its own subquestion's claims
              against the source files under search-results/{subq}/, running concurrently
              via asyncio.gather. Each LLM instance sees "focused on a single subquestion",
              so attacks are more precise.
    """
    claims = state.get("claims_to_verify", [])
    workspace = state.get("workspace_path", "")

    if not claims:
        return {"attack_results": []}

    # Group pending claims by subquestion
    from collections import defaultdict
    import asyncio

    by_subq: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        if c.status in ("pending", "needs_revision"):
            by_subq[c.subquestion or "_unknown"].append(c)

    if not by_subq:
        return {"attack_results": []}

    # Read full_research_topic (from parent state via workspace)
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    async def _attack_one_subq(subq: str, subq_claims: list[Claim]) -> list:
        """Review a single subquestion's claims — read only that subquestion's sources."""
        # First locate the source files under this subq (supports both q1 / Q1 naming)
        source_files: list[str] = []
        if subq and subq != "_unknown":
            for pat in (
                f"{subq}/**/*.md",
                f"{subq.lower()}/**/*.md",
                f"{subq.upper()}/**/*.md",
            ):
                found = list_workspace_files(workspace, "search-results", pat)
                if found:
                    source_files = found
                    break

        # Fallback: subq empty or no files in that directory -> use all search-results (avoid no-op)
        if not source_files:
            source_files = list_workspace_files(workspace, "search-results")

        # Use only metadata files (S001.md), not _raw.md (large HTML)
        # Metadata already contains verified verbatim quotes, enough for the attack agent to check
        meta_files = [f for f in source_files if not f.endswith("_raw.md")]
        if meta_files:
            source_files = meta_files

        source_texts: list[str] = []
        for sf in source_files:
            content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
            if content:
                source_texts.append(f"--- {Path(sf).name} ---\n{content}")

        # Put only this group's claims into extra_context
        claims_text = "\n".join(
            f"- {c.claim_id}: {c.claim_text}" for c in subq_claims
        )
        extra_context = f"## Claims to verify ({subq})\n\n{claims_text}"

        result_text = await iterative_refine(
            sources=source_texts,
            full_research_topic=full_research_topic,
            system_prompt=ATTACK_SYSTEM,
            extra_context=extra_context,
            role="verifier",  # adversarial fact-check = verifier task (Gemini-led)
        )
        return _parse_attack_results(result_text)

    # One LLM instance per subquestion, concurrent
    group_results = await asyncio.gather(
        *[_attack_one_subq(sq, sc) for sq, sc in by_subq.items()],
        return_exceptions=True,
    )

    all_results: list = []
    for r in group_results:
        if isinstance(r, Exception):
            # One subq failing doesn't affect other groups, but we retain an empty
            # result so downstream knows
            continue
        all_results.extend(r)

    return {"attack_results": all_results}


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
            "execution_log": ["Phase 1b: no claims to verify, passing through"],
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

    # Backfill bedrock_score + citation_verdict onto claims
    if grounding:
        g_map = {r["claim_id"]: r for r in grounding}
        for c in updated_claims:
            r = g_map.get(c.claim_id)
            if r:
                c.bedrock_score = r.get("score", 0.0)
                c.citation_verdict = r.get("verdict", "")

    # Write/update claim ledger
    _write_claim_ledger(workspace, updated_claims)

    # Determine result
    all_resolved = all(c.status in ("approved", "rejected") for c in updated_claims)
    failed_dims = result.get("failed_dimensions", [])

    log_entry = (
        f"Phase 1b: verified={len(to_verify)} "
        f"approved={sum(1 for c in updated_claims if c.status == 'approved')} "
        f"rejected={sum(1 for c in updated_claims if c.status == 'rejected')} "
        f"pending={sum(1 for c in updated_claims if c.status in ('pending', 'needs_revision'))}"
    )

    return {
        "claims": updated_claims,
        "phase1b_result": "pass" if all_resolved and not failed_dims else "fail",
        "quality_scores": result.get("quality_scores", {}),
        "execution_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Fallback trigger (parent-graph node, called after phase1b)
# ---------------------------------------------------------------------------

async def trigger_fallback_node(state: "ResearchState") -> dict:
    """Determine which SQs need focused re-search based on grounding metrics.

    Conditions for flagging a SQ:
      - grounded_ratio < 0.3   (too few claims survived grounding)
      - avg_bedrock   < 0.4   (claims weakly supported on average)
      - false_dims    >= 2    (multiple quality dimensions failed)

    Additionally: minimum search budget guard — for deep/standard research,
    ensure at least 40%/50% of the budget is used before quality can pass
    (prevents 1-round early exit with only ~9 searches out of 150).

    If fallback_count >= 2: emit [BLOCKER] and let graph proceed to phase2.
    Otherwise: populate needs_refetch and increment fallback_count.
    """
    from deep_research.state import ResearchState  # avoid circular import at module level
    from deep_research.tools.workspace import append_workspace_file

    quality_scores: dict = state.get("quality_scores", {})
    claims = state.get("claims", [])
    fallback_count: int = state.get("fallback_count", 0)
    workspace: str = state.get("workspace_path", "")

    # Normalise claims to Claim objects
    claim_objects: list[Claim] = []
    for c in claims:
        if isinstance(c, Claim):
            claim_objects.append(c)
        elif isinstance(c, dict):
            try:
                claim_objects.append(Claim(**c))
            except Exception:
                pass

    needs_refetch: list[str] = []
    blocker_msgs: list[str] = []

    # ── Minimum search budget guard ──────────────────────────────────────────
    # Prevent pipeline from exiting Phase 1a too early (e.g. 9/150 searches).
    # Uses large needs_refetch (all SQs) as signal; phase1a detects this and
    # applies a higher budget cap for budget-enforcement rounds (not the 25-cap
    # used for quality-failure focused refetch).
    search_count: int = state.get("search_count", 0)
    search_budget: int = state.get("search_budget", 150)
    depth: str = state.get("depth", "deep")
    _MIN_PCT = {"deep": 0.40, "standard": 0.50, "quick": 1.01}  # quick: never enforced
    min_searches = int(search_budget * _MIN_PCT.get(depth, 0.40))
    budget_ok = search_count >= min_searches

    if not budget_ok and fallback_count < 2:
        # Pull all planned SQs from coverage.chk section headers (## Q1:, ## Q2:, ...)
        # NOTE: old regex extracted checklist items, not SQ IDs → fixed to section headers
        from deep_research.tools.workspace import read_workspace_file as _rwf
        coverage_txt = _rwf(workspace, "coverage.chk") or ""
        import re as _re
        # Extract unique Q IDs from ## Q{n}: section headers (preserves order, deduplicates)
        planned_sqs = list(dict.fromkeys(_re.findall(r'^## (Q\d+):', coverage_txt, _re.MULTILINE)))
        if len(planned_sqs) <= 1:
            # Fallback: coverage.chk has only the placeholder Q1 — use claim subquestions
            sq_from_claims = sorted({c.subquestion for c in claim_objects if c.subquestion})
            if len(sq_from_claims) > len(planned_sqs):
                planned_sqs = sq_from_claims
        if len(planned_sqs) <= 1:
            # Last resort: read subquestions count from phase0-plan.md header
            plan_txt = _rwf(workspace, "phase0-plan.md") or ""
            m = _re.search(r"subquestions:\s*(\d+)", plan_txt)
            if m:
                sq_count = int(m.group(1))
                if sq_count > len(planned_sqs):
                    planned_sqs = [f"Q{i+1}" for i in range(sq_count)]
        log_msg = (
            f"trigger_fallback: search budget insufficient ({search_count}/{min_searches}), "
            f"forcing continuation across {len(planned_sqs)} SQs (not counted as quality failure)"
        )
        logger.info(log_msg)
        return {
            "needs_refetch": planned_sqs,  # large list → phase1a uses full budget, not 25-cap
            "fallback_count": fallback_count,  # intentionally NOT incremented
            "execution_log": [log_msg],
        }
    # ──────────────────────────────────────────────────────────────────────────

    for subq, dim_scores in quality_scores.items():
        false_dims = sum(1 for v in dim_scores.values() if not v)
        subq_claims = [c for c in claim_objects if c.subquestion == subq]
        if not subq_claims:
            continue

        # Exclude NO_SOURCE_TEXT / UNAVAILABLE / "" (validation skipped states, not ungrounded)
        # Only compute ratio across claims that truly completed grounding, to avoid
        # falsely triggering fallback when tools are unavailable.
        _unverifiable = {"NO_SOURCE_TEXT", "UNAVAILABLE", "ERROR", ""}
        verifiable = [c for c in subq_claims if c.citation_verdict not in _unverifiable]
        if verifiable:
            grounded = [c for c in verifiable if c.citation_verdict == "GROUNDED"]
            grounded_ratio = len(grounded) / len(verifiable)
        else:
            # Nothing verifiable -> set ratio to 1.0 (do not trigger refetch; avoid infinite loop)
            grounded_ratio = 1.0

        # avg_bedrock: exclude claims defaulted to 0.0 (grounding tool not executed)
        scored = [c for c in subq_claims if c.bedrock_score > 0.0]
        avg_bedrock = (
            sum(c.bedrock_score for c in scored) / len(scored)
            if scored else 1.0  # no score -> do not trigger the bedrock condition
        )

        # Check trigger conditions
        triggered = grounded_ratio < 0.3 or avg_bedrock < 0.4 or false_dims >= 2
        if not triggered:
            continue

        reason_parts = []
        if grounded_ratio < 0.3:
            reason_parts.append(f"grounded_ratio={grounded_ratio:.2f}")
        if avg_bedrock < 0.4:
            reason_parts.append(f"avg_bedrock={avg_bedrock:.2f}")
        if false_dims >= 2:
            reason_parts.append(f"false_dims={false_dims}")
        reason = ", ".join(reason_parts)

        if fallback_count >= 2:
            # Max retries reached — emit BLOCKER
            msg = f"[BLOCKER: {subq} grounding insufficient ({reason}), max refetches reached]"
            blocker_msgs.append(msg)
            logger.warning(msg)
        else:
            needs_refetch.append(subq)
            logger.info(f"trigger_fallback: {subq} refetch ({reason})")

    # Append BLOCKER entries to gap-log.md for visibility
    if blocker_msgs and workspace:
        lines = ["\n\n## Refetch BLOCKER (max retries reached, forced exit)"]
        lines.extend(f"- {m}" for m in blocker_msgs)
        append_workspace_file(workspace, "gap-log.md", "\n".join(lines) + "\n")

    new_fallback_count = fallback_count + 1 if needs_refetch else fallback_count

    log_msg = (
        f"trigger_fallback: needs_refetch={needs_refetch or 'none'}, "
        f"fallback_count={new_fallback_count}"
        + (f", blockers={len(blocker_msgs)}" if blocker_msgs else "")
    )

    return {
        "needs_refetch": needs_refetch,
        "fallback_count": new_fallback_count,
        "blockers": blocker_msgs,
        "execution_log": [log_msg],
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
