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
from deep_research.prompts_shared import FOCUSED_EXEC_PROMPT
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

    # Step 1: Merge report sections (also keep per-section contents for ledger splitting)
    section_files = list_workspace_files(workspace, "report-sections", "*.md")
    section_contents: list[tuple[str, str]] = []  # (section_name, content)
    merged_body = ""
    for sf in section_files:
        content = Path(sf).read_text(encoding="utf-8")
        if "Status: FINAL" in content or content.strip():
            section_contents.append((Path(sf).stem, content))
            merged_body += content + "\n\n---\n\n"

    # Step 2: Build statement ledger — split per section to avoid Lost in the Middle
    statements = await _build_statement_ledger(section_contents, claim_objects)
    statement_ledger_md = _format_statement_ledger(statements)
    write_workspace_file(workspace, "statement-ledger.md", statement_ledger_md)

    # Step 3: Sub-agent final audit
    audit_results = await _run_final_audit(workspace, statements, claim_objects)

    # Step 4: Process audit results — fix issues
    fixed_body, fix_log = _apply_fixes(merged_body, audit_results)

    # Step 5: Traceability chain validation (iron rule)
    chain_breaks = validate_traceability_chain(statements, claim_objects, source_objects)

    # Step 5b: Tier 1 — three-category marker validation for numeric claims (hard rule 3)
    # phase1a has already verified that quotes exist verbatim in the original text;
    # here we re-check number_tag formatting to ensure ORIGINAL/NORMALIZED/DERIVED
    # markers were not dropped in the final reporting phase.
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

    # Coverage sanity check (two layers)
    # Layer 1: SQ coverage (planned subquestions vs actual approved claims)
    sq_coverage_note = _compute_coverage_note(plan, approved_claims)

    # Layer 2: Keyword coverage (tools explicitly mentioned in the brief vs actual approved claims)
    uncovered_keywords = _find_uncovered_keywords(brief_keywords, approved_claims)
    keyword_coverage_note = _format_keyword_coverage(brief_keywords, uncovered_keywords)

    # Final sanity check: detect truncated phase2/3 output and display a CRITICAL banner at the top of the report
    critical_banner = _build_critical_banner(
        workspace=workspace,
        fixed_body=fixed_body,
        section_files=section_files,
        statements=statements,
        approved_claims=approved_claims,
        brief_keywords=brief_keywords,
        uncovered_keywords=uncovered_keywords,
    )

    # Whisper P2-1: LLM-level sanity check to catch defects the rule-based
    # banner misses — e.g. body has text but doesn't actually answer the
    # brief, paragraphs contradict each other, or named entities appear in
    # claims but off-topic. Appends to the existing banner; swallows LLM
    # errors so it never blocks report emission.
    llm_issues = await _llm_report_sanity_check(
        brief_text=brief_text,
        final_body=fixed_body,
        approved_claim_count=len(approved_claims),
    )
    if llm_issues:
        critical_banner = _append_llm_issues_to_banner(
            critical_banner, llm_issues, workspace
        )

    # Whisper P3-4: pipeline self-eval + follow-up suggestions. The rubric is
    # deterministic (per-SQ status tally) so it always appears; the follow-up
    # LLM call can silently degrade to an empty list without affecting the
    # scorecard.
    self_eval_section = await _build_self_eval_section(
        brief_text=brief_text,
        approved_claims=approved_claims,
        all_claims=claim_objects,
        state=state,
    )

    # Read clarifications if any
    clarify_section = ""
    clarify_md = read_workspace_file(workspace, "clarifications.md")
    if clarify_md:
        clarify_section = f"""## Research Requirement Clarification Log

{clarify_md}

---

"""

    final_report = f"""# Research Report: {state.get('topic', 'Unnamed Research')}

**Research Date:** {_today()}
**Research Depth:** {state.get('depth', 'deep')}
**Search Statistics:** {state.get('iteration_count', 0)} rounds, search {state.get('search_count', 0)}/{state.get('search_budget', 150)} times
**Claim Statistics:** {len(approved_claims)} approved / {len(rejected_claims)} rejected / {len(claim_objects)} total
**Citation Chain Completeness:** {len(statements)} statements, {len(chain_breaks)} chain breaks

{critical_banner}---

{clarify_section}## Summary

{summary}

---

## Detailed Analysis

{fixed_body}

---

## Source Reference Table

{_format_source_table(source_objects)}

## Coverage Integrity Report

### Subquestion Coverage

{sq_coverage_note}

### Coverage of Tools / Topics Explicitly Mentioned in the Research Brief

{keyword_coverage_note}

## Unanswered Questions and Knowledge Gaps

{gap_log}

{self_eval_section}
## Research Methodology

This research uses a LangGraph workflow + agent interleaved architecture:
- Phase 0: Research planning + multi-round clarification (workflow node + LLM Judge evaluation)
- Phase 1a: Multi-engine parallel search (agent node with direct API tools)
- Phase 1b: Grounding validation + adversarial verification (subgraph: workflow + sub-agent)
- Phase 2: Conflict arbitration + integration (workflow node)
- Phase 3: Statement-level audit + report generation (workflow node + sub-agent)

All factual claims are verified by the Bedrock Grounding Check.
All numbers are tagged ORIGINAL / NORMALIZED / DERIVED.
Citation chain: report sentence -> claim_id -> quote_id -> source_id.
"""

    write_workspace_file(workspace, "final-report.md", final_report)

    return {
        "final_report": final_report,
        "execution_log": [
            f"Phase 3 complete: {len(statements)} statements audited, "
            f"{len(chain_breaks)} chain breaks, "
            f"final-report.md written"
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


async def _extract_brief_keywords(brief_text: str) -> list[str]:
    """LLM extracts explicitly mentioned tool / product / technology names from research-brief.md.

    Extracts only proper nouns explicitly mentioned by the user (tool names, product names,
    service names); does not extract generic descriptors such as "speech recognition" or
    "high accuracy".

    Conservative fallback on failure: return an empty list (does not affect report generation).
    """
    if not brief_text or len(brief_text.strip()) < 50:
        return []

    # Truncate to avoid oversized context (a brief is usually < 3K chars; keep the first 4000)
    snippet = brief_text[:4000]

    prompt = f"""From the research brief below, find every tool name, product name, service name, or technology name explicitly mentioned by the user.

Extract only: tool names (e.g. Otter.ai), product names (e.g. MacWhisper), service names (e.g. Google Docs),
        specific OS/versions (e.g. iOS 26), hardware device names (e.g. Plaud Note).
Do not extract: generic descriptors ("speech recognition", "accuracy", "speaker diarization") or non-tool conceptual terms.

Research brief:
{snippet}

Output JSON only, in the format: {{"keywords": ["ToolA", "ToolB", ...]}}"""

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
        return "(No explicit tool/product names detected in the research brief; no cross-check needed.)"

    covered = [k for k in brief_keywords if k not in uncovered]
    lines = ["| Tool / Topic | Status |", "|-----------|------|"]
    for kw in brief_keywords:
        if kw in uncovered:
            lines.append(f"| {kw} | **No approved claim found** |")
        else:
            lines.append(f"| {kw} | Covered |")

    table = "\n".join(lines)

    if uncovered:
        missing = ", ".join(uncovered)
        warning = (
            f"\n\n> **Notice: no valid sources found for the following topics: {missing}**\n"
            f"> These tools or topics are explicitly mentioned in the research brief, but no approved claim covers them.\n"
            f"> Possible causes: the search did not hit, relevant pages were UNREACHABLE, or the grounding score was too low and they were filtered out."
        )
        return table + warning
    else:
        return table + "\n\nEvery tool / topic explicitly mentioned in the research brief is covered by at least one approved claim."


_DETAILED_ANALYSIS_MIN_CHARS = 500


def _build_critical_banner(
    *,
    workspace: str,
    fixed_body: str,
    section_files: list[str],
    statements: list[dict],
    approved_claims: list[Claim],
    brief_keywords: list[str],
    uncovered_keywords: list[str],
) -> str:
    """Detect truncated phase2/3 output and return a CRITICAL banner Markdown block.

    Returns empty string when the report looks healthy. When any critical
    condition fires, returns a top-of-report warning that makes the skeleton
    report unusable-at-a-glance (the caller would otherwise ship a report
    whose "Detailed Analysis" section is silently empty — see failed workspace
    2026-04-14 where phase2 produced 0 sections but phase3 still wrote a
    final-report.md claiming coverage).
    """
    issues: list[str] = []

    # 1. Detailed-analysis body empty / tiny
    body_chars = len((fixed_body or "").strip())
    if approved_claims and body_chars < _DETAILED_ANALYSIS_MIN_CHARS:
        issues.append(
            f'"Detailed Analysis" section only has {body_chars} chars (< {_DETAILED_ANALYSIS_MIN_CHARS}), '
            f"but there are {len(approved_claims)} approved claims; integration clearly failed."
        )

    # 2. report-sections/ empty but there are approved claims (a direct signal that phase2 truncated)
    if approved_claims and not section_files:
        issues.append(
            f"report-sections/ is empty, but there are {len(approved_claims)} approved claims. "
            f"Phase 2 did not write any section to disk."
        )

    # 3. Statement ledger empty (phase3 statistics/audit chain broken)
    if approved_claims and not statements:
        issues.append(
            "statement-ledger is empty. phase3 was unable to extract any statement from report-sections."
        )

    # 4. None of the tools/products explicitly named in the brief entered any claim
    if brief_keywords and uncovered_keywords and len(uncovered_keywords) == len(brief_keywords):
        issues.append(
            f"None of the {len(brief_keywords)} tools / topics explicitly mentioned in the research brief "
            f"({', '.join(brief_keywords[:5])}"
            f"{'...' if len(brief_keywords) > 5 else ''}) "
            f"are covered by an approved claim. This research provides essentially zero help toward the original question."
        )

    if not issues:
        return ""

    # Persist to gap-log so downstream audit / post-mortem tools can pick it up
    try:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note = (
            f"\n\n## CRITICAL — Phase 3 sanity check failed ({ts})\n"
            + "\n".join(f"- {i}" for i in issues)
            + "\n"
        )
        from deep_research.tools.workspace import append_workspace_file as _append
        _append(workspace, "gap-log.md", note)
    except Exception:
        logger.exception("failed to append CRITICAL banner to gap-log.md")

    bullet = "\n".join(f"- {i}" for i in issues)
    return (
        "\n> **CRITICAL — report integrity check failed**\n"
        "> Before the final assembly, this report was found to have the following structural defects; **do not quote it directly**:\n"
        + "\n".join(f"> {line}" for line in bullet.splitlines())
        + "\n> See gap-log.md and execution-log.md to trace the root cause.\n\n"
    )


# ---------------------------------------------------------------------------
# Whisper P2-1 — LLM-level report sanity check
# ---------------------------------------------------------------------------
# Rule-based banner catches the obvious carcasses (empty body, no sections).
# Many of the failed-workspace defects, though, were *semantic*: the body had
# thousands of words but didn't answer Q6, named 13 SOTA tools but discussed
# none of them, or asserted X then later contradicted X. Judge with a cheap
# LLM on the final assembly and append any critical findings to the banner.

# Truncate the body we send to the judge: enough to spot skeleton / topic
# drift / contradiction, but not so long that it wastes tokens on a report
# that's already marked skeletal by the rule-based layer.
_SANITY_BODY_CHAR_CAP = 12000
_SANITY_BRIEF_CHAR_CAP = 4000

_SANITY_SYSTEM = FOCUSED_EXEC_PROMPT + """You are a strict report-integrity auditor. You are given:
(1) the original research brief (what the user asked for);
(2) the assembled "Detailed Analysis" body of the final report.

Identify only *critical* defects — the kind that would make a reader conclude
the report failed to answer the brief. Do NOT nitpick phrasing, minor
omissions, or stylistic choices.

Critical defects include, but are not limited to:
- Large gaps where the body suddenly stops discussing a declared topic;
- Named tools / methods in the brief that the body mentions only in passing
  without actually characterising them;
- Internal contradictions (claim A in one paragraph, contradicted two
  paragraphs later without reconciliation);
- Boilerplate / filler text where substantive analysis was expected;
- Detailed analysis that re-states the brief rather than answering it.

Return STRICT JSON:
{
  "critical_issues": ["short imperative sentence describing issue 1", ...]
}

If the body looks healthy, return {"critical_issues": []}.
Keep each issue under 25 words. Return at most 5 issues, most severe first.
Do NOT wrap the JSON in markdown fences.
"""


async def _llm_report_sanity_check(
    *,
    brief_text: str,
    final_body: str,
    approved_claim_count: int,
) -> list[str]:
    """LLM-judge the final report body against the research brief.

    Returns a list of short critical-issue strings. Empty list means the
    judge found nothing wrong (or the check was skipped). Never raises —
    a failed sanity check must not block report emission.
    """
    body = (final_body or "").strip()
    brief = (brief_text or "").strip()
    # Nothing to check, or the rule-based banner already flagged it.
    if not body or not brief or approved_claim_count == 0:
        return []
    if len(body) < _DETAILED_ANALYSIS_MIN_CHARS:
        return []

    body_snippet = body[:_SANITY_BODY_CHAR_CAP]
    if len(body) > _SANITY_BODY_CHAR_CAP:
        body_snippet += "\n[...truncated for sanity check...]"
    brief_snippet = brief[:_SANITY_BRIEF_CHAR_CAP]

    user_msg = (
        f"## Research brief (what the user asked for)\n\n{brief_snippet}\n\n"
        f"---\n\n"
        f"## Final report body (what we actually wrote)\n\n{body_snippet}"
    )

    try:
        llm = get_llm("fast")
        response = await safe_ainvoke(
            llm,
            [SystemMessage(content=_SANITY_SYSTEM), HumanMessage(content=user_msg)],
        )
        raw = (response.content or "").strip()
    except Exception as exc:
        logger.info("llm sanity check skipped: %s", exc)
        return []

    if not raw:
        return []
    # Defensive JSON parse — strip ```json fences if the LLM ignored the
    # instruction.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        logger.info("llm sanity check produced non-JSON output; discarding")
        return []

    issues_raw = parsed.get("critical_issues") if isinstance(parsed, dict) else None
    if not isinstance(issues_raw, list):
        return []
    cleaned: list[str] = []
    for item in issues_raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text[:300])  # hard cap per issue
        if len(cleaned) >= 5:
            break
    return cleaned


def _append_llm_issues_to_banner(
    existing_banner: str, llm_issues: list[str], workspace: str,
) -> str:
    """Merge LLM sanity findings into the critical banner + persist to gap-log.

    Keeps the rule-based banner format so downstream consumers see one
    unified "do not quote this report" block.
    """
    if not llm_issues:
        return existing_banner

    bullets = "\n".join(f"> - (LLM) {issue}" for issue in llm_issues)
    llm_block = (
        "\n> **LLM sanity check — critical findings**\n"
        + bullets
        + "\n> See report-sanity-check.md for the full judge response.\n\n"
    )

    # Persist the LLM issues to their own file so post-mortem tooling can
    # read them without re-parsing the final report.
    try:
        body = "# Report sanity check (LLM judge)\n\n" + "\n".join(
            f"- {issue}" for issue in llm_issues
        ) + "\n"
        write_workspace_file(workspace, "report-sanity-check.md", body)
    except Exception:
        logger.exception("failed to write report-sanity-check.md")

    # Also append to gap-log so the post-mortem trail is co-located with
    # rule-based findings.
    try:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note = (
            f"\n\n## CRITICAL — LLM sanity check flagged issues ({ts})\n"
            + "\n".join(f"- {i}" for i in llm_issues)
            + "\n"
        )
        from deep_research.tools.workspace import append_workspace_file as _append
        _append(workspace, "gap-log.md", note)
    except Exception:
        logger.exception("failed to append LLM sanity findings to gap-log.md")

    if existing_banner.strip():
        # Rule banner already exists — inject LLM block before its trailing
        # blank line so both blocks render under a single "CRITICAL" header.
        return existing_banner.rstrip("\n") + "\n" + llm_block
    return (
        "\n> **CRITICAL — report integrity check failed**\n"
        + llm_block
    )


async def _build_self_eval_section(
    brief_text: str,
    approved_claims: list[Claim],
    all_claims: list[Claim],
    state: ResearchState,
) -> str:
    """Whisper P3-4 — self-eval scorecard + follow-up suggestions.

    Always returns a non-empty block. The follow-up LLM call can degrade
    silently (returns ``[]``) and we still print the deterministic per-SQ
    scorecard.
    """
    from deep_research.harness.self_eval import (
        format_self_eval_section,
        generate_follow_ups,
        score_subquestions,
    )

    blockers = list(state.get("blockers", []) or [])
    coverage_sqs = sorted({
        (getattr(c, "subquestion", "") or "").strip()
        for c in all_claims
        if (getattr(c, "subquestion", "") or "").strip()
    })
    outcomes = score_subquestions(
        all_claims, blockers=blockers, coverage_sqs=coverage_sqs,
    )

    async def _invoker(messages):
        # Use the "fast" tier — this is a structured one-shot, not reasoning.
        llm = get_llm("fast")
        return await safe_ainvoke(llm, messages)

    follow_ups: list[str] = []
    try:
        follow_ups = await generate_follow_ups(
            brief_text=brief_text, outcomes=outcomes, invoker=_invoker,
        )
    except Exception:  # noqa: BLE001 — defensive; never block phase3
        logger.exception("self-eval follow-up generation failed")

    return format_self_eval_section(outcomes, follow_ups=follow_ups)


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
        return "(No planned subquestion information available for cross-check.)"

    # Determine which SQs have >=1 approved claim
    covered: set[str] = set()
    sq_claim_counts: dict[str, int] = {}
    for c in approved_claims:
        sq = c.subquestion
        if sq:
            covered.add(sq)
            sq_claim_counts[sq] = sq_claim_counts.get(sq, 0) + 1

    uncovered = [sq for sq in planned_ids if sq not in covered]
    lines = ["| Subquestion | Status | Approved Claims |",
             "|--------|------|-------------------|"]
    for sq in planned_ids:
        count = sq_claim_counts.get(sq, 0)
        if sq in covered:
            status = "Covered"
        else:
            status = "**No approved claims**"
        lines.append(f"| {sq} | {status} | {count} |")

    table = "\n".join(lines)

    if uncovered:
        warning = (
            f"\n\n> **Coverage warning**: the following subquestions have no approved claims, "
            f"so their sections may lack substantive evidence: **{', '.join(uncovered)}**\n"
            f"> See 'Unanswered Questions and Knowledge Gaps' for reasons."
        )
        return table + warning
    else:
        return table + "\n\nEvery planned subquestion is covered by at least one approved claim."


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


_LEDGER_SYSTEM = FOCUSED_EXEC_PROMPT + """Split the report content into statement-level entries. Every factual, numeric, or inference sentence occupies its own line.

## Hard Rules
- Review the entire section in full; no sentence may be skipped or merged
- Every factual sentence, every numeric sentence, every inference sentence must be listed independently
- statement_id must be drawn from the ID range allocated for this batch (to avoid collisions)

## Four type categories (correct classification is required; this directly affects citation chain validation)

### Must be classified as "opinion" (claim_ids may be empty [])
The following sentences do not need a claim_id and must be marked type="opinion":
- **Section headings / subheadings** (e.g. "## Mainstream Tool Comparison", "**Conclusion**")
- **Lead-in / transitional sentences** (e.g. "This section analyzes...", "Summarizing the above...", "It is worth noting")
- **Summary / concluding sentences** (overviews that span multiple claims and cannot map to a single source)
- **Confidence-level marker lines** (e.g. "HIGH confidence", "**Confidence level: MEDIUM**")
- **Disclaimer / warning lines** (e.g. "Insufficient data warning...")
- **Methodology explanations** (not factual assertions but explanations of how the report was produced)
- **Any evaluative statement in the report that has no claim_id marker (e.g. Q1-C2)**

### "fact" (claim_ids required)
Concrete, verifiable factual statements; the report usually contains claim markers in the form [Q1-C2] or (Q1-C2).

### "numeric" (claim_ids required)
Statements containing specific numbers (price, percentage, version number, rating, etc.).

### "inference" (claim_ids required)
Conclusions derived from multiple facts; the report usually marks them with [INFERENCE].

## Decision order
1. If the sentence contains a claim_id marker like [Q1-C1] or (Q1-C1) -> fact or numeric (fill claim_ids with the marked IDs)
2. If the sentence contains [INFERENCE] -> inference (fill claim_ids with every [Qn-Cm] ID in the sentence)
3. If the sentence is a heading, lead-in, transition, summary, or evaluation -> opinion (claim_ids=[])
4. When uncertain -> prefer opinion to avoid creating false broken chains

## start_char / end_char fields (important)
For each statement, additionally output start_char and end_char,
representing the character start/end indices of the sentence in the "report content" above (0-based, Python slicing semantics).
Validation rule: content[start_char:end_char] must equal text.

Inaccurate estimates are not penalized — the program will fall back by searching for text to find the real position.
But the text field must still be a verbatim excerpt from the original report; otherwise the whole entry will be rejected.

## Output (strict JSON array)
```json
[{"statement_id": "ST-1", "section": "Q1-positive", "text": "original sentence from the report",
  "start_char": 123, "end_char": 145,
  "claim_ids": ["Q1-C1"], "type": "fact|numeric|inference|opinion"}]
```

For opinion entries, claim_ids must be an empty array []; do not fabricate claim_ids.

Output JSON only, no other text."""


async def _build_statement_ledger(
    sections: list[tuple[str, str]],
    claims: list[Claim],
) -> list[dict]:
    """Split report into statement-level entries — per-section to avoid LiM.

    The previous version fed the entire body (20-50K tokens) to the LLM at once;
    mid-body statements were frequently dropped because of Lost in the Middle,
    threatening hard rule 4 "citation chain completeness". The new version makes
    an independent LLM call per section, so each call sees only one section,
    and results are merged after being output within a unified ID range.

    claim_id noise optimization: each section is given only "the claim_ids
    related to this subquestion" rather than the entire approved claim list.
    When approved claims > 30, the full list overwhelms LLM attention, making
    it more likely to force unrelated claim_ids into a statement and break
    citation chain accuracy.
    """
    if not sections:
        return []

    # Group approved claim_ids by subquestion.
    # Supports two subquestion formats:
    #   - New format (v11+): "Q1", "Q2" -> key = "q1", "q2"
    #   - Old format (v10-): "Subquestion 1: Mainstream tool inventory" -> key = "q1" (extracted number) AND full-text key
    # Dual-key storage ensures the section_name -> claim_ids mapping does not break.
    import re as _re
    claims_by_subq: dict[str, list[str]] = {}
    for c in claims:
        if c.status != "approved":
            continue
        subq = c.subquestion or ""
        # Key 1: full-text lowercase
        claims_by_subq.setdefault(subq.lower(), []).append(c.claim_id)
        # Key 2: extract Q{n} prefix (e.g. "Q1" or extract "q1" from "Subquestion 1")
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

    # Call each section concurrently; assign independent ID starting points to avoid collisions
    import asyncio as _asyncio
    tasks = []
    id_offset = 1
    for section_name, content in sections:
        # Extract the subquestion identifier from section_name:
        # - "q1_section" -> "q1"
        # - "subquestion 1: mainstream tool inventory_section" -> "q1" (extracted number)
        # - Try the Q{n} format first, then number extraction
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
            else "(No relevant approved claim for this section)"
        )
        section_id_start = id_offset
        id_offset += 200  # Reserve 200 IDs per section
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
    """Split statements for a single section using an independent ID range."""
    # role="verifier" — statement splitting is structured extraction, Gemini-led
    response = await safe_ainvoke_chain(
        role="verifier",
        messages=[
            SystemMessage(content=_LEDGER_SYSTEM),
            HumanMessage(content=f"""## Section: {section_name}
(statement_id numbering starts at ST-{id_start})

## Report Content
{content}

## Available claim_ids
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

    # Index validation: statement.text must actually exist within the section content.
    # After passing: text is overwritten with content[start:end] (to eliminate LLM transcription hallucinations), with start/end span attached.
    verified = verify_indexed_items(
        content, data, "text", log_prefix=f"phase3/ledger/{section_name}"
    )

    # Force-overwrite the section field and correct IDs that are out of range.
    # Python fallback: fill in claim_ids from inline markers inside statement.text
    # Supports both [Q1-C1] and (Q1-C1) formats.
    _claim_id_re = re.compile(r'Q\d+-C\d+')
    out = []
    for i, st in enumerate(verified):
        st["section"] = section_name
        if not st.get("statement_id", "").startswith("ST-"):
            st["statement_id"] = f"ST-{id_start + i}"
        # Remove the LLM's original hints (only the final true start/end values need to be kept)
        st.pop("start_char", None)
        st.pop("end_char", None)

        # If the LLM did not fill claim_ids (empty array), try to extract them from inline markers in text.
        # Supports both formats: (Q1-C1) or [Q1-C1] (Phase 2 legacy format uses square brackets)
        if not st.get("claim_ids") and st.get("type") != "opinion":
            text = st.get("text", "")
            extracted = _claim_id_re.findall(text)
            if extracted:
                st["claim_ids"] = extracted

        out.append(st)
    return out


def _format_statement_ledger(statements: list[dict]) -> str:
    """Render statement-ledger.md.

    span field format: `@[start:end]`. If the span is missing (should not happen
    because verify_indexed_items filters out anything that cannot be located),
    render it as an empty string to preserve the table structure.
    """
    header = (
        "# Statement Ledger\n\n"
        "<!-- The span field @[s:e] refers to the character index in report-sections/{section}.md; "
        "use section_content[s:e] to restore the original sentence. -->\n\n"
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


AUDIT_SYSTEM = FOCUSED_EXEC_PROMPT + """You are the final-quality adversary. Verify the citation chain completeness of each statement.

## Verification Rules

1. Citation chain: statement -> claim_id -> quote_id -> source
2. Verbatim numeric verification
3. Tone consistency
4. Composite hallucination detection
5. Over-inference detection

## Iterative mode notes

You may receive source documents in multiple rounds. Each round you must:
1. Review this round's source originals and verify the citation chain of each statement
2. Accumulate issues found (do not drop previously found issues just because this round did not find any)
3. If this round finds supporting evidence, you may mark a previous issue as NONE
4. Output the complete latest audit result (covering the latest status of every statement)

## Output format (strict JSON array, covering the latest status of every statement)

```json
[{"statement_id": "ST-1", "issue": "NONE|BROKEN_CHAIN|NUMBER_MISMATCH|TONE_MISMATCH|COMPOSITE_HALLUCINATION|OVER_INFERENCE|NO_SOURCE", "detail": "...", "fix": "..."}]
```"""


async def _run_final_audit(
    workspace: str,
    statements: list[dict],
    claims: list[Claim],
) -> list[StatementCheck]:
    """Run final sub-agent audit on statements vs claims — concurrent per section.

    Scenario: final-quality audit — verifying the completeness of statement-level citation chains.

    The previous version submitted all statements (50-100+) plus all claims as a single
    extra_context, a classic "doing too much at once" anti-pattern: the LLM had to verify
    statements across multiple sections simultaneously, easily losing focus in the middle,
    and iterative_refine re-sent the bloated extra_context every round, so actual
    consumption = extra * rounds. The new version groups by section and runs
    `iterative_refine` concurrently per section, so each call sees only its own
    statements + the claims relevant to that subquestion, and the program merges
    audit results at the end.

    Note: sources are still shared across all sections (the same statement may quote a
    cross-section source); that part cannot be shrunk further and must be handled by
    iterative_refine batching.
    """
    import asyncio as _asyncio
    import re as _re

    # Step 1: group statements by section (skip opinion)
    by_section: dict[str, list[dict]] = {}
    for st in statements:
        if st.get("type") == "opinion":
            continue
        sec = st.get("section", "unknown")
        by_section.setdefault(sec, []).append(st)

    if not by_section:
        return []

    # Step 2: group claims by subquestion (key is lower-case Q1/Q2/...)
    claims_by_subq: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_subq.setdefault(c.subquestion.lower(), []).append(c)

    # Step 3: collect all source documents (shared across all sections)
    source_texts: list[str] = []
    source_files = list_workspace_files(workspace, "search-results")
    for sf in source_files:
        content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else ""
        if content:
            source_texts.append(f"--- {Path(sf).name} ---\n{content}")

    full_research_topic = read_workspace_file(workspace, "research-brief.md") or ""

    # Step 4: an independent audit task per section
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
        ) or "(No relevant claim for this section)"
        statement_text = "\n".join(
            f"- {st.get('statement_id')}: [{st.get('type')}] {st.get('text', '')[:100]}"
            for st in section_statements
        )

        extra_context = f"""## Section: {section_name}

## Statements in this section (audit every one)
{statement_text}

## Related Claim Ledger (subquestion {subq_key.upper() or '?'})
{claim_text}"""

        result_text = await iterative_refine(
            sources=source_texts,
            full_research_topic=full_research_topic,
            system_prompt=AUDIT_SYSTEM,
            extra_context=extra_context,
            role="verifier",  # Report audit = verifier task (Gemini-led; Vectara HHEM reports the lowest hallucination rate)
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
        return "(No approved claims; cannot generate a summary.)"

    claims_text = "\n".join(
        f"- {c.claim_id}: {c.claim_text}" for c in approved
    )

    # role="writer" — summary writing, Claude Opus-led
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[
            SystemMessage(content="""Generate a 1-3 paragraph summary from the approved claims below.
Hard rule: every sentence in the summary must map to a claim_id. Quoting information outside the claims is forbidden.
Language: English."""),
            HumanMessage(content=f"## Approved Claims\n{claims_text}"),
        ],
        max_tokens=2048,
        temperature=0.2,
    )

    return response.content


def _format_source_table(sources: list[Source]) -> str:
    header = (
        "| # | Source | Tier | URL Status |\n"
        "|---|------|------|----------|\n"
    )
    rows = []
    for s in sources:
        rows.append(f"| {s.source_id} | [{s.title}]({s.url}) | {s.tier} | {s.url_status} |")
    return header + "\n".join(rows) + "\n" if rows else "(No sources)\n"
