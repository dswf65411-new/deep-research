# Phase 3: Report Generation + Final Cross-Check

**Input:** read from the workspace:
- `workspace/report-sections/*.md` (sections with Status: FINAL)
- `workspace/claim-ledger.md` (approved claims)
- `workspace/source-registry.md`
- `workspace/search-results/` (for final cross-check)
- `workspace/gap-log.md` (for unresolved questions)
- `workspace/phase0-plan.md` (for metadata)

**Output:** `workspace/final-report.md` + `workspace/statement-ledger.md` + `workspace/execution-log.md`

**Critical ordering: merge first -> build statement-ledger first -> cross-check first -> only then write summary and charts**

**Required reading for this Phase:** `prompts/ref-citation-embedding.md`, `prompts/ref-focused-execution.md`

---

## Execution Mode: Focused Task Execution

Before producing any report content, emit a `[TASK LIST]` covering:
- T1 merge sections, T2 build statement ledger (one sub-task per subquestion section),
- T3 subagent cross-check (one sub-task per check group),
- T4 self-critique (one sub-task per critique dimension: logic, completeness, bias, citation chain, tone),
- T5 final quality scan, T6 summary writing, T7 charts, T8 compose final report.

Process each wrapped in `[WORKING: T{n}]` / `[DONE: T{n}]`. CRITICAL: do NOT start T6 summary until T1-T5 are all DONE — this mirrors the "build ledger and cross-check before writing summary" rule above.

See `prompts/ref-focused-execution.md` for full rules.

---

## Step 1: Merge Report Sections

Read all `workspace/report-sections/q{n}_section.md` (only those with Status: FINAL) and merge them in subquestion order into the report body.

**At this point do not write the summary or generate charts.** Finish the cross-check first.

---

## Step 2: Build the Statement Ledger

Split the merged report into statement-level rows and write `workspace/statement-ledger.md`:

```markdown
# Statement Ledger

| statement_id | section | text | claim_ids | type | verified |
|-------------|---------|------|-----------|------|----------|
| ST-1 | Q1-advocate | "{original sentence in report}" | Q1-C1 | fact | pending |
| ST-2 | Q1-analysis | "{inferential sentence}" | Q1-C1,Q1-C2 | inference | pending |
| ST-3 | Q2-number | "{sentence containing number}" | Q2-C3 | numeric | pending |
```

**type classification:**
- `fact`: fact directly quoted from the source
- `numeric`: statement containing a number
- `inference`: cross-claim derivation (should already be marked [INFERENCE])
- `opinion`: opinion expression (no cross-check needed)

---

## Step 3: Subagent Final Cross-Check

Spawn a **Sonnet** subagent to do a structured cross-check.

**Key improvement: the subagent cross-checks statement-ledger vs claim-ledger, not a free-form full-text scan.**

**Subagent prompt (model: sonnet):**

```
You are the final quality attacker. Try to find every error in the report.

## Statements to cross-check
Read: {absolute workspace path}/statement-ledger.md
Only cross-check rows with type = fact / numeric / inference (skip opinion).

## Citation ledger
Read: {absolute workspace path}/claim-ledger.md

## Search results
Read every file under: {absolute workspace path}/search-results/
(Use Glob to list all .md files and read them one by one. Skip missing files.)

## Cross-check rules (strict)
For each statement:

1. Citation chain complete?
   - statement has claim_id -> claim_id exists in claim-ledger with status=approved -> claim has quote_id -> quote_id exists in search-results
   - Any broken link = BROKEN_CHAIN

2. Numbers checked verbatim?
   - Number in statement vs number in claim_text vs original QUOTE/NUMBER text
   - Any mismatch = NUMBER_MISMATCH

3. Tone consistent?
   - Original "may" -> statement "will" = TONE_MISMATCH

4. Composite hallucination?
   - Statement cites 2+ claims to derive a conclusion, but no single source has said that conclusion = COMPOSITE_HALLUCINATION

5. Over-inference?
   - Is the derivation in an [INFERENCE] sentence reasonable? Does it go beyond the scope of the claims? = OVER_INFERENCE

If you cannot find precise support, you must report it. Passing something as SUPPORTED on "similar meaning" is forbidden.
Splicing a conclusion across sources and marking it SUPPORTED is forbidden.

## Output format
STATEMENT_ID: {statement_id}
ISSUE: NONE / BROKEN_CHAIN / NUMBER_MISMATCH / TONE_MISMATCH / COMPOSITE_HALLUCINATION / OVER_INFERENCE / NO_SOURCE
DETAIL: {specific issue}
FIX: {fix suggestion}
---

Finally, output a summary:
TOTAL: {N} statements checked
PASS: {N}
FAIL: {N} (list all problematic statement_ids)
```

---

## Step 4: Process Subagent Results

Update the `verified` field in statement-ledger:

| Issue | Action |
|-------|------|
| NONE | verified = pass |
| BROKEN_CHAIN | add the missing claim_id or quote_id, or delete the statement |
| NUMBER_MISMATCH | correct to the original number |
| TONE_MISMATCH | weaken tone |
| COMPOSITE_HALLUCINATION | mark [INFERENCE] or delete |
| OVER_INFERENCE | add qualifiers or move to "Unanswered questions" |
| NO_SOURCE | delete or additional search (**at most 2 additional searches, Fail-Fast**) |

---

## Step 5: Citation Metadata Check (academic sources)

For sources that cite **academic papers**, read the five-way taxonomy in `prompts/ref-citation-embedding.md`:

1. **Title existence:** exact search for full title
2. **Metadata match:** author + title + journal + year
3. **Identifier validation:** DOI/arXiv ID valid and content consistent

Fail -> [FABRICATED] -> remove from the report.

For **non-academic sources**: confirm that `fetched_title` in source-registry matches the title actually cited.

---

## Step 6: Self-Critique

From the perspective of "the most critical reviewer":

1. Conclusion support: does every conclusion have a supporting claim_id? Are there logical leaps?
2. Counter-side adequacy: one-sided?
3. Evidence quality: over-reliance on T5-T6?
4. Completeness: missing facets?
5. Actionability: are conclusions concrete enough?

Severe -> fix (additional search, at most 2, Fail-Fast). Medium -> fix wording. Minor -> fix formatting.

---

## Step 7: Final Quality Scan

| Check item | Standard | Fail -> |
|--------|------|--------|
| Every factual statement has an approved claim_id | 0 broken chains | delete or backfill |
| Every claim_id has a quote_id/number_id | citation chain complete | backfill or delete |
| Advocate/critic balance | not one-sided | additional search |
| Numbers marked ORIGINAL/NORMALIZED/DERIVED | 0 unmarked numbers | backfill marks |
| No self-contradiction | logically consistent | fix |
| **CTran = 1.0** | every pro-con conflict faithfully presented | add back missing conflicts |

---

## Step 8: Only Now Write the Summary and Charts

**Only after all cross-checks pass** do you generate:

**8a. Summary (1-3 paragraphs):**
- May only be recomposed from approved claims (status=approved)
- Re-summarizing freely from report-section prose is forbidden
- Every summary sentence must correspond to a claim_id

**8b. Mermaid chart (auto-select):**

| Topic feature | Chart type |
|---------|---------|
| Pipeline/process | `flowchart` |
| Temporal evolution | `timeline` |
| Classification/structure | `mindmap` |
| Comparison | markdown comparison table |

Do not generate: Quick mode, pure Q&A.

---

## Step 9: Assemble the Final Report

Assemble and write the following into `workspace/final-report.md`:

```markdown
# Research Report: {topic}

**Research date:** {YYYY-MM-DD}
**Research mode:** {mode}
**Research depth:** {depth}
**Overall confidence:** {high/medium/low} — {reason}
**Search stats:** {R} rounds, {N} unique URLs, {M} deep-reads

## Summary
{generated in Step 8a, each sentence with claim_id}

## Visual Overview
{generated in Step 8b}

## Detailed Analysis
{merged report-sections}

## Stakeholder Perspectives
| Perspective | Viewpoint | Source | Bedrock | claim_id |

## Pro-Con Debate Log
| Claim | Advocate | Critic | Bedrock (advocate) | Bedrock (critic) | Adjudicator |

## Citation Source Index
| # | Source | Tier | COI | Date | URL Status | Bedrock |

## Unanswered Questions and Knowledge Gaps
{imported from gap-log.md: missing perspectives + weak evidence + unresolved contradictions + BLOCKER items}

## Research Methodology
```

---

## Step 10: Update execution-log Final Statistics

```
## Final Statistics

Report quality:
  Factual statements: {n} | passed cross-check: {n} | revised: {n} | deleted: {n}
  Citation-chain completeness: {n}/{total} = {pct}%
  CTran: {x}/{y} = {ratio}

URL:
  LIVE: {n} | STALE: {n} | HALLUCINATED: {n}

Uncertainty distribution:
  HIGH:{n} | MEDIUM:{n} | CONFLICTING:{n} | LOW:{n}

Claim Ledger:
  Total: {n} | Approved: {n} | Rejected: {n}

Search totals:
  Search calls: {n}/{budget} | unique URLs: {n} | deep-read: {n} sources | iterations: {R} rounds

Coverage Matrix:
  evidence_found: {n}/{total_required}
  searched_no_evidence: {n}
```

---

## Step 11: Present to User

Output the final report and offer follow-up options:

```
Follow-up options:
1. Deep-dive into a specific subquestion
2. Update specific data
3. Export to PDF (/markdown-to-pdf)
4. Additional validation for CONFLICTING/LOW claims
5. Verify report conclusions via testing
```

---

## Phase 3 Completion Checklist

```
[ ] 1. Statement-ledger built? -> {N} statements
[ ] 2. Subagent final cross-check complete? -> PASS:{n} FAIL:{n}
[ ] 3. All FAIL statements handled? -> OK
[ ] 4. Academic-citation metadata passed? -> {N} verified
[ ] 5. Self-Critique complete? -> severe:{n} medium:{n} minor:{n}
[ ] 6. Final quality scan passed? -> OK
[ ] 7. CTran = 1.0? -> {ratio}
[ ] 8. Summary generated only from approved claims? -> OK
[ ] 9. final-report.md written? -> OK
[ ] 10. execution-log.md updated? -> OK
[ ] 11. All items in gap-log.md reflected in "Unanswered questions" section? -> OK
-> all OK -> research complete
```
