"""Phase 2: Integration + Conflict Resolution.

Workflow node — reads approved claims, resolves contradictions,
writes report sections with confidence levels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.harness.claim_dedup import is_near_duplicate
from deep_research.harness.validators import validate_claims_for_phase2, validate_numeric_claims
from deep_research.harness.source_tier import tier_rank
from deep_research.llm import get_llm, safe_ainvoke, safe_ainvoke_chain
from deep_research.prompts_shared import FOCUSED_EXEC_PROMPT
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

    # Critic-revise loop hints (gpt-researcher style).
    # When phase2 is re-entered after phase2_review flagged issues, the
    # prior verdict is surfaced to the writer as a "revise these problems"
    # section in the system prompt. On the first pass both dicts are empty
    # and no hint is injected.
    prior_verdict: dict = state.get("review_verdict") or {}
    revision_count: int = state.get("revision_count", 0)
    per_sq_revision_hints: dict[str, list[str]] = prior_verdict.get("per_sq_issues") or {}
    top_level_revision_hints: list[str] = prior_verdict.get("issues") or []

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

        # Revision hint — only present on revisions (revision_count > 0).
        # Surfaces prior critic verdict so the writer knows exactly what to fix
        # instead of regenerating the same flawed section from scratch.
        revision_hint = ""
        if revision_count > 0:
            sq_issues = per_sq_revision_hints.get(subq, [])
            if sq_issues or top_level_revision_hints:
                parts = ["", "## Revision requested (round %d)" % revision_count]
                if sq_issues:
                    parts.append(f"The prior draft of {subq} was flagged for:")
                    parts.extend(f"- {i}" for i in sq_issues)
                if top_level_revision_hints:
                    parts.append("")
                    parts.append("General issues from the last critic pass:")
                    parts.extend(f"- {i}" for i in top_level_revision_hints)
                parts.append("")
                parts.append(
                    "Rewrite this section to address these problems. "
                    "Keep the same structure and claim_id citations."
                )
                revision_hint = "\n".join(parts)

        integrate_system = f"""{FOCUSED_EXEC_PROMPT}{instructions}

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
Please follow the Phase 2 Step 6 format to generate the report section for {subq}.{revision_hint}"""

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

    # Whisper P2-4: mark coverage.chk in-sync with the sections that actually
    # landed on disk, so phase3's coverage table and post-mortem auditing
    # don't read stale "[ ] not_started" entries for SQs that did complete.
    marked_sqs = _sync_coverage_checklist(workspace, sections_on_disk)

    # Whisper P3-3: surface advocate/critic collisions into gap-log so the
    # final report can't silently flatten both sides into a one-sided
    # narrative. Keeps scope tight by only flagging SQs that *actually* have
    # claims cited from both stakeholder roles.
    try:
        from deep_research.harness.stakeholder_collision import (
            collect_collisions, format_collision_section,
        )
        collisions = collect_collisions(approved, state.get("sources", []))
        section_text = format_collision_section(collisions)
        if section_text:
            append_workspace_file(workspace, "gap-log.md", "\n\n" + section_text)
    except Exception as exc:  # noqa: BLE001 — defensive; never block phase2
        logger.warning("stakeholder collision detector failed: %r", exc)

    # Whisper X-1: flag same-upstream mirror clusters so downstream grounding
    # doesn't count multiple copies of the same paper as independent
    # confirmation. Purely additive: only appends to gap-log.
    try:
        from deep_research.harness.source_mirror import (
            detect_mirror_groups, format_mirror_warnings,
        )
        mirror_groups = detect_mirror_groups(state.get("sources", []))
        mirror_text = format_mirror_warnings(mirror_groups)
        if mirror_text:
            append_workspace_file(workspace, "gap-log.md", "\n\n" + mirror_text)
    except Exception as exc:  # noqa: BLE001 — defensive; never block phase2
        logger.warning("source mirror detector failed: %r", exc)

    return {
        "report_sections": report_sections,
        "blockers": blockers,
        "execution_log": [
            f"Phase 2 complete: {len(report_sections)} sections, {len(approved)} approved claims integrated"
            + (f" ({len(sections_on_disk)} files written)" if sections_on_disk else " (warning: report-sections is empty)")
            + (f"; coverage.chk marked done for {', '.join(sorted(marked_sqs))}" if marked_sqs else "")
        ],
    }


# ---------------------------------------------------------------------------
# coverage.chk synchronisation (Whisper P2-4)
# ---------------------------------------------------------------------------

# Section filenames are ``q{N}_section.md`` (lowercase). We lift the Q id and
# flip the matching ``- [ ] advocate — ...`` / ``- [ ] critic — ...`` rows.
_SECTION_FILENAME_RE = re.compile(r"^(q\d+)(?:[_-]|$)", re.IGNORECASE)


def _extract_sq_id_from_section_filename(name: str) -> str | None:
    """Return the canonical ``Q{n}`` id for a section filename, else None."""
    m = _SECTION_FILENAME_RE.match(name)
    if not m:
        return None
    return m.group(1).upper()


def _sync_coverage_checklist(workspace: str, sections_on_disk: list[Path]) -> set[str]:
    """Rewrite coverage.chk so every SQ that produced a section is marked done.

    Idempotent: running it twice is safe (already-done rows are left alone).
    Returns the set of SQ ids that were flipped from not_started/in_progress
    to done, for logging.
    """
    if not sections_on_disk:
        return set()

    ws = Path(workspace)
    coverage_path = ws / "coverage.chk"
    if not coverage_path.exists():
        # phase0 should have written this; if it's missing, nothing to sync.
        return set()

    covered: set[str] = set()
    for p in sections_on_disk:
        sq = _extract_sq_id_from_section_filename(p.name)
        if sq:
            covered.add(sq)
    if not covered:
        return set()

    original_text = coverage_path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    current_sq: str | None = None
    marked: set[str] = set()

    # Match ``## Q7: description`` headers (case-insensitive, allow punctuation
    # or whitespace after the id).
    header_re = re.compile(r"^##\s+(Q\d+)\b", re.IGNORECASE)
    # Match ``- [ ] advocate — not_started`` / ``- [ ] critic — in_progress``
    # etc. Accept em-dash, hyphen, or ``-`` between role and state.
    row_re = re.compile(
        r"^(\s*-\s*)\[\s\]\s*(advocate|critic|perspective)\s*[—–-]\s*(\S+)\s*$",
        re.IGNORECASE,
    )

    for line in original_text.splitlines():
        header_m = header_re.match(line.strip())
        if header_m:
            current_sq = header_m.group(1).upper()
            new_lines.append(line)
            continue

        row_m = row_re.match(line)
        if row_m and current_sq and current_sq in covered:
            prefix, role, _state = row_m.group(1), row_m.group(2), row_m.group(3)
            new_lines.append(f"{prefix}[x] {role.lower()} — done")
            marked.add(current_sq)
            continue

        new_lines.append(line)

    if marked:
        coverage_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return marked


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


# =============================================================================
# Phase 2 Review — gpt-researcher-style Critic-revise loop
# =============================================================================
#
# Pattern: multi_agents/agents/editor.py:129-142 (EditorAgent._draft_article)
# runs a reviewer → revisor cycle capped at max_revisions=2. Each revisor round
# gets the critic's issues and rewrites targeted sections.
#
# Here the loop sits between phase2_integrate and phase3_report:
#   phase2_integrate → phase2_review → (accept? phase3 : bump_revision → phase2)
#
# State plumbing is in state.py (`review_verdict`, `revision_count`); the graph
# wiring lives in graph.py (route_after_review + bump_revision_count).

MAX_REVISIONS = 2  # gpt-researcher default; aligns with their editor.py config

_REVIEW_SYSTEM = FOCUSED_EXEC_PROMPT + """You are a research-report critic. You review drafted sections of a deep-research report and decide whether they ship, or need to be rewritten.

Your job is NOT to rewrite anything. Output a strict JSON verdict only.

## What passes ("accept": true)
A section passes when ALL of these hold:
1. It has substantive analysis — not just a bulleted claim dump (unless the dump is wrapped in the explicit "section integration failed, using fallback format" header, in which case treat it as a known-bad signal and reject).
2. It directly answers the subquestion it belongs to. A section that drifts into unrelated topics is a failure even if the writing is polished.
3. Every factual sentence ends with a claim_id marker like [Q1-C3].
4. It does not quote claim_ids that are outside the approved-claim list you are given.
5. It is not truncated: at least 200 chars of prose beyond the header, no mid-sentence cut-off.

## What fails ("accept": false)
Any of:
- Section starts with "section integration failed, using fallback format" (writer crashed — needs retry).
- Section is shorter than 200 chars of content, or ends mid-sentence.
- Section cites fewer than 30% of the approved claims available for that SQ (wasting grounding work).
- Section wanders off-topic (fewer than half of the factual sentences carry a claim_id matching this SQ).
- A subquestion flagged in the research brief as the user's primary question has no substantive section.

## Output — STRICT JSON, no markdown, no prose

{
  "accept": true | false,
  "issues": ["top-level issue 1", "top-level issue 2"],
  "per_sq_issues": {
    "Q1": ["specific issue for Q1"],
    "Q3": ["specific issue for Q3"]
  }
}

Rules:
- When `accept` is true, both `issues` and `per_sq_issues` may be empty.
- When `accept` is false, `per_sq_issues` MUST list at least one failing subquestion with at least one concrete, actionable issue. Vague complaints ("quality is low") are not useful — name the specific problem ("Q3 cites only 2 of 14 approved claims and repeats the same point twice").
"""


async def phase2_review(state: ResearchState) -> dict:
    """Critic pass over drafted report sections.

    Reads every ``report-sections/*.md`` file on disk (the source of truth;
    ``state["report_sections"]`` may be stale on a revise loop) and asks a
    verifier LLM for a strict JSON verdict. Writes the verdict to
    ``review-log.md`` for traceability.

    The revision counter itself is bumped by the ``bump_revision_count`` node
    in graph.py — this node only produces the verdict. When the verdict says
    ``accept: false`` but ``revision_count`` already equals ``MAX_REVISIONS``,
    graph's ``route_after_review`` still sends the pipeline to phase3 (no
    infinite loops); this node reports honestly regardless.
    """
    workspace = state["workspace_path"]
    current_count = state.get("revision_count", 0)

    sections_dir = Path(workspace) / "report-sections"
    section_files = sorted(sections_dir.glob("*.md")) if sections_dir.exists() else []

    if not section_files:
        # Nothing to review — don't block phase3; the empty-output case is
        # already logged as a blocker by phase2_integrate.
        verdict = {
            "accept": True,
            "issues": ["phase2 produced no sections; nothing to review"],
            "per_sq_issues": {},
        }
        _write_review_log(workspace, current_count, verdict, note="no sections to review")
        return {
            "review_verdict": verdict,
            "execution_log": [
                f"Phase 2 review (revision {current_count}): no sections — pass-through"
            ],
        }

    sections_text = "\n\n".join(
        f"## File: {f.name}\n\n{f.read_text(encoding='utf-8')}"
        for f in section_files
    )

    brief = read_workspace_file(workspace, "research-brief.md") or ""
    if len(brief) > 4000:
        brief = brief[:2000] + "\n\n...[middle omitted]...\n\n" + brief[-2000:]

    # Digest of approved claims so the critic can check coverage (e.g. "Q3
    # had 14 approved claims but the section cites only 2").
    claims_raw = state.get("claims", [])
    claim_objs = [c if isinstance(c, Claim) else Claim(**c) for c in claims_raw]
    approved = [c for c in claim_objs if getattr(c, "status", None) == "approved"]
    per_sq_summary: dict[str, int] = {}
    for c in approved:
        per_sq_summary[c.subquestion] = per_sq_summary.get(c.subquestion, 0) + 1
    claim_digest = "\n".join(
        f"- {sq}: {n} approved claims" for sq, n in sorted(per_sq_summary.items())
    ) or "(no approved claims)"

    human_content = f"""## Research brief
{brief or '(no brief available)'}

## Approved-claim counts by subquestion
{claim_digest}

## Drafted report sections
{sections_text}

## Revision round
{current_count} (max {MAX_REVISIONS}). When revision_count already equals {MAX_REVISIONS}, the graph will force-pass regardless of your verdict; still give an honest assessment so the review-log reflects reality.
"""

    try:
        response = await safe_ainvoke_chain(
            role="verifier",
            messages=[
                SystemMessage(content=_REVIEW_SYSTEM),
                HumanMessage(content=human_content),
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        content = getattr(response, "content", "") or ""
        verdict = _parse_review_verdict(content)
    except Exception as e:
        # Never let a critic failure block the pipeline — default-accept so
        # phase3 can proceed and the review-log records why.
        logger.warning("phase2_review: critic LLM call failed (%s); defaulting to accept=true", e)
        verdict = {
            "accept": True,
            "issues": [f"critic LLM call failed: {e!r}; defaulted to accept"],
            "per_sq_issues": {},
        }

    _write_review_log(workspace, current_count, verdict)

    return {
        "review_verdict": verdict,
        "execution_log": [
            f"Phase 2 review (revision {current_count}): accept={verdict.get('accept')}, "
            f"issues={len(verdict.get('issues', []))}, per_sq={len(verdict.get('per_sq_issues', {}))}"
        ],
    }


def _parse_review_verdict(text: str) -> dict:
    """Tolerant JSON parsing — LLM may wrap JSON in ```json fences or trail prose."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.MULTILINE
        ).strip()
    m = re.search(r"\{[\s\S]*\}", stripped)
    if not m:
        return {
            "accept": True,
            "issues": ["critic returned no JSON; defaulted to accept"],
            "per_sq_issues": {},
        }
    try:
        v = json.loads(m.group(0))
    except Exception:
        return {
            "accept": True,
            "issues": ["critic JSON parse failed; defaulted to accept"],
            "per_sq_issues": {},
        }
    return {
        "accept": bool(v.get("accept", True)),
        "issues": list(v.get("issues", []) or []),
        "per_sq_issues": dict(v.get("per_sq_issues", {}) or {}),
    }


def _write_review_log(
    workspace: str, revision: int, verdict: dict, note: str = ""
) -> None:
    """Append the verdict to review-log.md so every round of the revise loop
    is auditable. The file accumulates across revisions; grep for
    ``## Revision N`` to find a specific round.
    """
    lines = [
        f"\n\n## Revision {revision}",
        f"- accept: {verdict.get('accept')}",
    ]
    if note:
        lines.append(f"- note: {note}")
    if verdict.get("issues"):
        lines.append("- top-level issues:")
        lines.extend(f"  - {i}" for i in verdict["issues"])
    if verdict.get("per_sq_issues"):
        lines.append("- per-subquestion issues:")
        for sq, issues in verdict["per_sq_issues"].items():
            lines.append(f"  - {sq}:")
            lines.extend(f"    - {i}" for i in (issues or []))
    append_workspace_file(workspace, "review-log.md", "\n".join(lines) + "\n")
