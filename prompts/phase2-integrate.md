# Phase 2: Integration + Contradiction Adjudication

**Input:** read from the workspace:
- `workspace/claim-ledger.md` (**core input: only use claims with status=approved**)
- `workspace/source-registry.md`
- `workspace/search-results/` (for original-text quotes)
- `workspace/coverage.chk` (for completeness check)
- `workspace/gap-log.md` (for unresolved contradictions)

**Output:** `workspace/report-sections/q{n}_section.md`

**Required reading for this Phase:** `prompts/source-criteria.md`, `prompts/ref-citation-embedding.md`

---

## Hard Rule: Generate Only from Approved Claims

**Read claim-ledger.md; use only claims with `status=approved`. Claims that are rejected or pending are forbidden from appearing in the report.**

Before starting integration, build an **Approved Claims list**; during integration, citing any fact or number outside this list is forbidden.

---

## Step 1: Import Results

Read the workspace files. For each subquestion, organize the approved claims:
- Advocate claims + Bedrock scores + quote_ids
- Critic claims + Bedrock scores + quote_ids
- Residual contradictions (both advocate and critic have approved claims with opposite conclusions)

---

## Step 2: Handle Settled Claims

| Status | Action |
|------|------|
| Advocate and critic reach consensus + Bedrock >= 0.7 | Adopt |
| Advocate has approved claim, critic coverage = searched_2x_no_evidence | Adopt, but must be worded as "within the critic search scope covered in this run, no high-quality evidence was found that would overturn this claim" (blanket wording like "critic found no rebuttal" is forbidden) |
| Critic has approved claim, advocate has no valid response | Adopt as risk/limitation |
| Both sides have approved claims with opposite conclusions | -> Step 3 contradiction adjudication |

---

## Step 3: Contradiction Adjudication (three-stage, taking sides based on Bedrock score delta is forbidden)

**Bedrock judges only "whether the text supports the claim", not "which side is actually true in reality".**

For contradicting claims:

**3a. Multi-dimensional comparison (not just Bedrock):**

| Comparison dimension | Advocate | Critic |
|---------|------|------|
| Source Tier | T{n} | T{n} |
| Independence (not same source) | {Y/N} | {Y/N} |
| Methodology Transparency | {Y/N} | {Y/N} |
| Freshness | {date} | {date} |
| Bedrock (reference only, not decisive) | {score} | {score} |

**3b. Additional search for adjudication (if the comparison above cannot resolve it):**
Search third-party meta-analyses, primary data -> run Bedrock on new sources -> update claim-ledger

**3c. Cannot adjudicate:**
After 2 rounds of additional search still no resolution -> keep as [CONFLICTING], present both sides in the report, record in "Unresolved contradictions" of `gap-log.md`. (Fail-Fast: at most 2 rounds, no round 3)

---

## Step 4: Confidence Level + Uncertainty Score

**Every conclusion must be assigned one; no exceptions.**

| Level | Uncertainty | Conditions | Tone rule |
|------|---------|------|---------|
| HIGH | < 0.1 | 2+ independent sources, >=1 T1-T2, Bedrock >= 0.7 | may assert |
| MEDIUM | 0.1-0.4 | 1-2 sources, T1-T4, Bedrock 0.5-0.7 | cautious: "according to available sources" |
| CONFLICTING | 0.4-0.7 | advocate and critic both supported | present both sides |
| LOW | > 0.7 | only T5-T6, or Bedrock < 0.5 | weaken: "some sources claim... but it could not be verified" |

**Hard rules:**
- LOW: assertive tone is forbidden
- Conclusions that contain numbers must be HIGH or MEDIUM, otherwise remove the number or mark [UNVERIFIED]
- **Subquestions supported only by T4-T6 sources** -> can at most be rated CONFLICTING; producing recommendation sentences is forbidden; numeric assertions are forbidden

---

## Step 5: Source Depth Evaluation

| Check item | Action |
|--------|------|
| Originality: Primary / Secondary / Tertiary | Mark |
| Conflict of interest | Mark [COI], disclose in the conclusion |
| Same origin: multiple sources relaying the same study | Count as 1 independent source, **dedupe at root-source level** |
| Freshness: exceeds freshness SLA | Downgrade to "background information" |

---

## Step 6: Write Report Sections

Read `prompts/ref-citation-embedding.md` and build the verified-sources constraint per its rules.

For each subquestion, write `workspace/report-sections/q{n}_section.md`. **Write each one immediately after finishing it.**

```markdown
# Q{n}: {subquestion}
Status: FINAL
Based-On-Claims: Q{n}-C1, Q{n}-C2, ...

## Original Evidence (advocate)
> QUOTE[S{id}-Q{n}]: "{verbatim quote}" — [{source name}]({URL})
> NUMBER[S{id}-N{n}]: {number} — Original: "{original sentence}" — [{source name}]({URL})

## Original Evidence (critic)
> QUOTE[S{id}-Q{n}]: "{verbatim quote}" — [{source name}]({URL})

## Number Reconciliation Table
| Reported number | Type | Original sentence | Source | claim_id |
|---------|------|---------|------|----------|
| {number} | ORIGINAL | "{original sentence}" | [{URL}] | Q{n}-C{m} |
| {converted number} (orig: {original number+unit}) | NORMALIZED | "{original sentence}" | [{URL}] | Q{n}-C{m} |
| {computed value} | DERIVED | formula: {formula} | [{URL}] | Q{n}-C{m} |

## Analysis and Judgment  <- Claude's inference
{analysis based on the original evidence above}
{every inferential sentence must carry a supporting claim_id}
{cross-claim derivations must be marked [INFERENCE] and list all supporting claim_ids}

## Confidence Level
- Level: HIGH/MEDIUM/CONFLICTING/LOW
- Uncertainty: {0.0-1.0}
- Basis: {number of sources, tier, Bedrock score}
```

---

## Phase 2 Completion Checklist (answer item by item)

```
[ ] 1. Each subquestion has a report-section file (Status: FINAL)?
       -> list them

[ ] 2. Every factual statement has a corresponding approved claim_id?
       -> factual statements without a claim_id: {N} (must be 0)

[ ] 3. Every conclusion has a confidence level and uncertainty score?
       -> HIGH:{n} MEDIUM:{n} CONFLICTING:{n} LOW:{n}

[ ] 4. All LOW conclusions have their tone weakened or moved to "Unanswered"?
       -> OK/FAIL

[ ] 5. Every non-verbatim statement marked [INFERENCE] and carries claim_ids?
       -> unmarked inferences: {N} (must be 0)

[ ] 6. Root-source dedup complete? All [COI] disclosed in conclusions?
       -> OK/FAIL

[ ] 7. gap-log.md updated (unresolved contradictions recorded)?
       -> OK/FAIL

-> all OK -> proceed to Phase 3
```
