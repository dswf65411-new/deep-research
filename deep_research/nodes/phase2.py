"""Phase 2: Integration + Conflict Resolution.

Workflow node — reads approved claims, resolves contradictions,
writes report sections with confidence levels.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.claim_dedup import is_near_duplicate
from deep_research.harness.validators import validate_claims_for_phase2, validate_numeric_claims
from deep_research.harness.source_tier import tier_rank
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
from deep_research.state import Claim, ResearchState, Source
from deep_research.tools.workspace import (
    append_workspace_file,
    read_workspace_file,
    write_workspace_file,
)

from deep_research.config import get_prompt
from deep_research.context import iterative_refine

logger = logging.getLogger(__name__)

# Pattern for scanning claim_id quotes in section text (e.g. Q1-C1, Q12-C23)
_CLAIM_ID_RE = re.compile(r"\bQ\d+-C\d+\b")

_DOMAIN_BIAS_THRESHOLD = 0.30  # Same as phase1a._log_domain_bias


def _extract_domain(url: str) -> str:
    """Extract hostname from URL, stripping the www. prefix."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.") if host else ""
    except Exception:
        return ""


def _detect_biased_domains(sources: list[Source]) -> set[str]:
    """Compute domain distribution across all sources and return the set of domains whose share exceeds 30%."""
    from collections import Counter
    domains = [
        _extract_domain(s.url)
        for s in sources
        if s.url and s.url_status != "UNREACHABLE"
    ]
    domains = [d for d in domains if d]
    if not domains:
        return set()
    total = len(domains)
    counts = Counter(domains)
    return {d for d, n in counts.items() if n / total > _DOMAIN_BIAS_THRESHOLD}


def _dedup_approved_claims(claims: list[Claim]) -> list[Claim]:
    """Remove near-duplicate approved claims within each subquestion.

    Keeps the first occurrence (by claim_id order) when two claims share
    similarity >= 0.92. Cross-subquestion dedup is intentionally skipped to
    avoid removing legitimately similar facts that belong to different SQs.
    """
    by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        by_subq.setdefault(c.subquestion, []).append(c)

    result: list[Claim] = []
    for subq_claims in by_subq.values():
        accepted: list[Claim] = []
        for claim in subq_claims:
            if any(is_near_duplicate(claim.claim_text, prev.claim_text) for prev in accepted):
                logger.debug(
                    "phase2 dedup: skipped near-duplicate claim %s", claim.claim_id
                )
                continue
            accepted.append(claim)
        result.extend(accepted)

    return result


def _build_fallback_section(
    subq: str,
    claims: list[Claim],
    biased_domains: set[str],
    sid_to_domain: dict[str, str],
    error: str,
) -> str:
    """Emergency section used when the LLM writer fails for a subquestion.

    Preserves each claim verbatim with its ``[Qx-Cy]`` tag so Phase 3's
    statement ledger can still build a traceability chain. Clearly marked so
    readers know the narrative integration step was skipped — this is NOT a
    synthesised report paragraph.

    Why: without this, a transient LLM failure wipes the entire report body
    (observed 2026-04-14: 113 approved claims but zero sections on disk).
    A bullet dump is uglier than a polished paragraph but far better than
    silent failure.
    """
    header = (
        f"## {subq} — section integration failed, using fallback format\n\n"
        f"> **Phase 2 writer failed**: {error[:200]}\n"
        f"> Below are the approved claims for this subquestion, listed verbatim (**no LLM integration, no inference**).\n"
        f"> Confidence level: LOW (for audit traceability only; not treated as complete analysis)\n\n"
    )
    lines: list[str] = []
    for c in claims:
        bias_tag = ""
        if biased_domains:
            hit = {
                sid_to_domain[sid]
                for sid in c.source_ids
                if sid in sid_to_domain and sid_to_domain[sid] in biased_domains
            }
            if hit:
                bias_tag = f"  [BIASED_SOURCE: {', '.join(sorted(hit))}]"
        # Inline citation [Qx-Cy] after each claim so Phase 3 ledger can
        # extract claim_ids via the same regex it uses for normal sections.
        lines.append(f"- {c.claim_text} [{c.claim_id}]{bias_tag}")
    return header + "\n".join(lines) + "\n"


def _scan_phantom_claim_ids(section: str, approved_ids: set[str]) -> list[str]:
    """Return the list of claim_ids that appear in the section text but are not in approved_ids.

    The phase2 LLM is instructed to only quote approved claims. Occasionally the LLM
    hallucinates IDs (typos / fabricated IDs that do not exist). The phase3 audit
    catches chain breaks, but catching them earlier saves work.

    Note:
        phase2 itself does not perform span-based validation: it is a "writer"
        rather than an "extractor". The original-text quotes were already
        validated by the index check in phase1a; section text will be split
        into statements and index-validated again by verify_indexed_items in
        phase3. This function performs only a lightweight referential integrity
        check on claim_ids and does not touch the text content itself.
    """
    found = set(_CLAIM_ID_RE.findall(section or ""))
    return sorted(found - approved_ids)


async def phase2_integrate(state: ResearchState) -> dict:
    """Integrate approved claims into report sections."""
    workspace = state["workspace_path"]
    claims = state.get("claims", [])

    # Clear stale report-sections from any previous pipeline run on this workspace.
    # Phase 3 reads ALL *.md files from report-sections/, so leftovers from a prior
    # run would contaminate the new report.
    import glob as _glob
    _old_sections = _glob.glob(str(Path(workspace) / "report-sections" / "*.md"))
    for _f in _old_sections:
        try:
            Path(_f).unlink()
            logger.info("phase2: removed stale section file %s", Path(_f).name)
        except OSError:
            pass

    # Convert to Claim objects
    claim_objects = []
    for c in claims:
        if isinstance(c, Claim):
            claim_objects.append(c)
        elif isinstance(c, dict):
            claim_objects.append(Claim(**c))

    # Iron rule: only approved claims with quote_ids
    approved = validate_claims_for_phase2(claim_objects)

    # Near-duplicate dedup: within the same subquestion, claims with similarity >= 0.92
    # keep only the first occurrence.
    # Prevents cross-round follow-up searches from feeding semantically duplicate claims
    # to the LLM (distractor + token waste).
    approved = _dedup_approved_claims(approved)

    # Iron rule: numeric claims must have number_tag
    violations = validate_numeric_claims(approved)
    blockers = []
    if violations:
        blockers.extend(violations)

    # Read phase instructions
    instructions = get_prompt("phase2-integrate.md")

    # Read gap log (source-registry is no longer injected into LLM context —
    # it acts as a distractor; when the LLM integrates a subquestion section it
    # does not need to see the source list for other subquestions)
    gap_log = read_workspace_file(workspace, "gap-log.md") or ""

    # Fetch full_research_topic
    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # Group claims by subquestion
    by_subq: dict[str, list[Claim]] = {}
    for c in approved:
        by_subq.setdefault(c.subquestion, []).append(c)

    # Collect source text for each subquestion (for iterative_refine batched processing).
    # Pass in sources so _gather_source_texts can sort by tier (T1-T3 first).
    all_sources: list[Source] = [
        s if isinstance(s, Source) else Source(**s)
        for s in state.get("sources", [])
        if s
    ]
    subq_sources = _gather_source_texts(workspace, approved, sources=all_sources)

    # Domain concentration detection: find domains with excessive share (same as phase1a's _log_domain_bias)
    biased_domains: set[str] = _detect_biased_domains(all_sources)
    # Build source_id -> domain mapping for claim tagging
    sid_to_domain: dict[str, str] = {
        s.source_id: _extract_domain(s.url)
        for s in all_sources
        if s.url
    }

    # BLOCKER list (written by trigger_fallback_node)
    all_blockers: list[str] = state.get("blockers", [])

    async def _write_one_section(
        subq: str,
        subq_claims: list[Claim],
    ) -> tuple[str, str, list[str]]:
        """Write report section for one subquestion.

        Returns (subq, section_text, local_blockers).
        Runs concurrently with other sections via asyncio.gather.
        """
        local_blockers: list[str] = []

        def _claim_line(c: Claim) -> str:
            base = f"- {c.claim_id} [{c.claim_type}] (bedrock={c.bedrock_score:.2f}): {c.claim_text}"
            if biased_domains:
                claim_domains = {
                    sid_to_domain[sid]
                    for sid in c.source_ids
                    if sid in sid_to_domain and sid_to_domain[sid] in biased_domains
                }
                if claim_domains:
                    tag = ", ".join(sorted(claim_domains))
                    base += f"  [BIASED_SOURCE: {tag}]"
            return base

        claims_text = "\n".join(_claim_line(c) for c in subq_claims)

        def _persist_fallback(err: str) -> tuple[str, str, list[str]]:
            section = _build_fallback_section(
                subq, subq_claims, biased_domains, sid_to_domain, err
            )
            write_workspace_file(
                workspace, f"report-sections/{subq.lower()}_section.md", section
            )
            return subq, section, [
                f"phase2/{subq}: LLM writer failed; wrote fallback section: {err[:200]}"
            ]

        integrate_system = f"""{instructions}

You are the research report integrator. Based on the validated approved claims and source originals, generate a report section.

## Hard Rules
1. Use only the approved claims below; quoting any other information is forbidden.
2. **Every factual sentence must end with a claim_id marker in the format [Q1-C2].**
   - Example: "Whisper Large v3 achieves 8.3% Chinese WER. [Q1-C5]"
   - Example: "This feature is supported only on macOS 14 and later. [Q2-C3]"
   - When one sentence quotes multiple claims: "...[Q1-C5] [Q1-C7]"
3. Cross-claim inferences must be labelled [INFERENCE] and list the relevant claim_ids:
   - Example: "[INFERENCE] Therefore local tools have a privacy advantage. [Q1-C5] [Q2-C1]"
4. Numbers must be tagged ORIGINAL/NORMALIZED/DERIVED.
5. A confidence level must be assigned (HIGH / MEDIUM / CONFLICTING / LOW).
6. **Headings, lead-in sentences, transitional sentences, and concluding summary sentences do not need claim_ids** — these are report structure, not factual assertions.
7. **[BIASED_SOURCE] marker**: when a claim carries this marker, the source domain share exceeds 30% (likely self-promotion).
   - Unless an independent T1-T3 source corroborates the same fact, the claim must be flagged as CONFLICTING.
   - Add a parenthetical note in the report: "(this information comes from {{domain}}'s own source; treat with caution)".

## Iterative mode notes

You may receive source documents in multiple rounds. Each round you must:
1. Review this round's source originals and integrate valuable information into the draft.
2. Keep the draft structure intact; append new information to the end of the relevant section.
3. Output the complete latest report section (not a diff).

Language: English (keep technical terms verbatim).
Please follow the Phase 2 Step 6 format to generate the report section for {subq}."""

        extra = f"""## Subquestion: {subq}

## Approved Claims
{claims_text}"""

        source_texts = subq_sources.get(subq, [])

        try:
            if source_texts:
                section = await iterative_refine(
                    sources=source_texts,
                    full_research_topic=full_research_topic,
                    system_prompt=integrate_system,
                    extra_context=extra,
                    role="writer",
                )
            else:
                response = await safe_ainvoke_chain(
                    role="writer",
                    messages=[
                        SystemMessage(content=integrate_system),
                        HumanMessage(content=f"""{extra}\n\n(This subquestion has no search result originals; generate the report section using only the approved claims.)"""),
                    ],
                    max_tokens=16384,
                    temperature=0.2,
                )
                section = response.content
        except Exception as e:
            logger.exception("phase2/%s writer failed; falling back to raw claim dump", subq)
            return _persist_fallback(str(e))

        if not section or not section.strip():
            return _persist_fallback("LLM returned empty section")

        # Referential integrity check
        approved_ids = {c.claim_id for c in approved}
        phantom_ids = _scan_phantom_claim_ids(section, approved_ids)
        if phantom_ids:
            msg = f"phase2/{subq}: quoted unapproved claim_ids: {', '.join(phantom_ids)}"
            logger.warning(msg)
            local_blockers.append(msg)

        # BLOCKER disclaimer
        if any(f"[BLOCKER: {subq}" in b for b in all_blockers):
            disclaimer = (
                f"> **Insufficient data warning ({subq})**\n"
                f"> The grounding score for this subquestion fell short of the threshold and the maximum follow-up searches (2) have already been performed.\n"
                f"> The analysis below is based only on the limited data available; treat with caution — gaps may exist.\n\n"
            )
            section = disclaimer + section

        write_workspace_file(workspace, f"report-sections/{subq.lower()}_section.md", section)
        return subq, section, local_blockers

    # Write all subquestion sections in parallel (the original sequential for-loop is replaced with asyncio.gather)
    ordered_subqs = sorted(by_subq.items())
    results = await asyncio.gather(
        *[_write_one_section(sq, sq_claims) for sq, sq_claims in ordered_subqs],
        return_exceptions=True,
    )

    report_sections = []
    for r in results:
        if isinstance(r, BaseException):
            # Defensive: _write_one_section's own try/except should have
            # caught writer failures and returned a fallback tuple. An
            # exception reaching here means something outside the writer
            # path failed (file I/O, fallback builder bug). Surface it.
            msg = f"phase2 section task raised an unexpected exception: {r!r}"
            logger.error(msg)
            blockers.append(msg)
            continue
        _, section, local_blockers = r
        report_sections.append(section)
        blockers.extend(local_blockers)

    # Critical assertion — never leave the pipeline with an empty
    # report-sections/ directory when there were approved claims to write.
    sections_on_disk = list((Path(workspace) / "report-sections").glob("*.md"))
    if by_subq and not sections_on_disk:
        critical_msg = (
            f"[CRITICAL: phase2] {len(by_subq)} SQs have approved claims "
            f"({len(approved)} claims total), but report-sections/ is empty."
            f" Downstream phase3 will be unable to produce a detailed analysis section."
        )
        logger.critical(critical_msg)
        blockers.append(critical_msg)
        append_workspace_file(
            workspace,
            "gap-log.md",
            f"\n\n## CRITICAL — Phase 2 empty output\n- {critical_msg}\n",
        )

    return {
        "report_sections": report_sections,
        "blockers": blockers,
        "execution_log": [
            f"Phase 2 complete: {len(report_sections)} sections, {len(approved)} approved claims integrated"
            + (f" ({len(sections_on_disk)} files written)" if sections_on_disk else " (warning: report-sections is empty)")
        ],
    }


def _gather_source_texts(
    workspace: str,
    claims: list[Claim],
    sources: list[Source] | None = None,
) -> dict[str, list[str]]:
    """Read search result files relevant to approved claims, grouped by subquestion.

    Returns dict[subquestion, list[source_text]]; each source is a separate string.
    This format lets iterative_refine batch them (BM25 ranking + greedy packing).

    T1-T3 sources are placed before T4-T5 so iterative_refine feeds higher-quality
    content to the integrator first (BM25 fallback still applies within each batch).
    """
    # Build tier lookup: source_id → sort key (T1=0 … T5=4)
    tier_map: dict[str, int] = {}
    if sources:
        for s in sources:
            tier_map[s.source_id] = tier_rank(s.tier)

    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}  # subq → seen source_ids

    for claim in claims:
        subq = claim.subquestion
        if subq not in result:
            result[subq] = []
            seen[subq] = set()

        # Sort source_ids by tier quality (T1 first, T5 last)
        sorted_sids = sorted(claim.source_ids, key=lambda sid: tier_map.get(sid, 3))

        for sid in sorted_sids:
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
