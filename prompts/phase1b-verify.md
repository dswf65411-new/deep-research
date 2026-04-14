# Phase 1b: Validation + Conditional Debate + Iteration

**Input:** `workspace/search-results/` + `workspace/source-registry.md` + `workspace/coverage.chk`
**Output:** `workspace/claim-ledger.md` + `workspace/grounding-results/` + updated `workspace/coverage.chk` + `workspace/gap-log.md`
**Completion condition:** passes the structured stopping conditions

**This Phase has two sub-stages:**
- **1b-A:** Grounding + quality evaluation -> if all pass, proceed directly to Phase 2
- **1b-B (triggered only when 1b-A fails):** subagent cross-check + debate + additional search -> iterate

---

## Validation Tools (CLI mode)

**Full CLI usage: see `prompts/ref-cli-tools.md`**

```
PY=.venv/bin/python3
GS=grounding_scripts
MINICHECK_PY=.minicheck-venv/bin/python3
```

| Tool | CLI command | Role | When to use |
|------|---------|------|--------|
| Bedrock Grounding | `echo '{...}' \| $PY $GS/bedrock-guardrails.py --cli` | **Primary validator** | Every claim must be run |
| MiniCheck | `echo '{...}' \| $MINICHECK_PY $GS/minicheck.py --cli` | **Fallback** | Only when Bedrock errors |
| NeMo Grounding | `echo '{...}' \| $PY $GS/nemo-guardrails.py --cli` | **Third fallback** | Only when both Bedrock and MiniCheck error |

**Bedrock decision rules (threshold per claim type):**

| Claim type | Bedrock threshold | Extra requirements |
|-----------|-------------|---------|
| Numeric / exact quote | >= 0.8 | Citation API = precise + URL = LIVE/STALE |
| Comparative / ranking | >= 0.75 | at least 2 independent sources |
| Causal / forecast | >= 0.75 | at least 1 primary + 1 secondary source |
| Background qualitative | >= 0.7 | — |

**Bedrock API throttling:** on 429 -> wait 3 seconds and retry, up to 3 times. Continued failure -> switch to MiniCheck.
**Batch strategy:** whenever possible, batch claims for the same subquestion into a single Bash call to Bedrock (multiple claims in one JSON).

**Grounding tool availability check (hard rule, not skippable):**

At the start of Phase 1b, run a simple Bash test first to verify the grounding CLI is working:

```bash
TEST_JSON='{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}'

# 1. Test Bedrock
echo "$TEST_JSON" | $PY $GS/bedrock-guardrails.py --cli 2>/dev/null

# 2. If it fails, test MiniCheck
echo "$TEST_JSON" | $MINICHECK_PY $GS/minicheck.py --cli 2>/dev/null

# 3. If that also fails, test Nemo
echo "$TEST_JSON" | $PY $GS/nemo-guardrails.py --cli 2>/dev/null
```

**Decision rules:**
- At least one of the three must return valid JSON (containing grounding_score or confidence)
- If all three fail -> **stop the research pipeline immediately** and report to the user:
  ```
  [GROUNDING-UNAVAILABLE] All grounding validation tools are unavailable:
  - Bedrock: {error message}
  - MiniCheck: {error message}
  - Nemo: {error message}
  Please fix and re-run /research.
  ```
- **Continuing Phase 1b with no grounding tool available is forbidden**
- **Replacing grounding tools with "manual judgment" or "Claude's own evaluation" is forbidden**

**Bedrock judges only "whether the claim is supported by the provided text", not "which side is actually true in reality". Using Bedrock score deltas to directly adjudicate advocate-vs-critic truth is forbidden.**

---

# 1b-A: Grounding + Quality Evaluation

## Step 1: Grounding Validation

Read all source files under `workspace/search-results/`.

For every QUOTE and NUMBER in each source:
1. **Bedrock**: claim text + source text (when over 2000 tokens, extract a ±250-token window around the quote) -> score
2. **If it contains a number or direct quote** -> also call Citation API

Write results to `workspace/grounding-results/q{n}_grounding.md`.

---

## Step 2: Build the Claim Ledger

Write `workspace/claim-ledger.md`:

```markdown
# Claim Ledger

| claim_id | subquestion | type | claim_text | source_ids | quote_ids | bedrock | citation | status |
|----------|-------------|------|------------|------------|-----------|---------|----------|--------|
| Q1-C1 | Q1 | numeric | "{full claim text}" | S003 | S003-N1 | 0.85 | precise | pending |
| Q1-C2 | Q1 | comparative | "{full claim text}" | S003,S005 | S003-Q1,S005-Q2 | 0.78 | N/A | pending |
```

**Field descriptions:**
- `claim_id`: unique identifier, format Q{n}-C{m}
- `type`: numeric / comparative / causal / forecast / qualitative
- `claim_text`: **canonical text**; all downstream citations and cross-checks use this as the reference
- `quote_ids`: the corresponding verbatim quote IDs (from the search-results files)
- `status`: pending / approved / rejected / needs_revision

**Rule: once `claim_text` is written, downstream phases must not rewrite it. To correct it, create a new claim_id.**

---

## Step 3: 4-Dimension Quality Evaluation

Evaluate each subquestion:

| Dimension | Pass criterion |
|------|----------|
| **Actionability** | Statement is specific, scope is clear, qualifiers are correct. If the source itself is uncertain, preserving the uncertain tone does not count as fail. |
| **Freshness** | Core data meets the freshness SLA set in Phase 0 |
| **Plurality** | >= 2 independent sources (not relaying the same origin) |
| **Completeness** | Both advocate and critic + the main perspectives are covered |

**Evaluation result:**
- 4/4 Pass -> all pending claims for the subquestion become **approved** -> update coverage.chk
- Any Fail dimension -> record the failed dimension -> enter **1b-B**

---

## Step 4: 1b-A Fast-Track Decision

**If every subquestion is 4/4 Pass:**
-> skip 1b-B, proceed directly to Phase 2

**If any subquestion has a Fail dimension:**
-> only subquestions that Fail enter 1b-B (passing subquestions do not need to be redone)

---

# 1b-B: Subagent Cross-Check + Debate + Additional Search (conditionally triggered)

**This section is only executed when 1b-A has not fully passed.**

## Step 5: Subagent Attacker-Style Cross-Check

For each Fail subquestion, spawn a **Sonnet** subagent. Multiple subquestions may run in parallel.

**Subagent prompt (fill in and pass to the Agent tool, model: sonnet):**

```
You are an attacker-style fact-checker. Your task is to try to prove the following claims wrong.

## Claims to check
{extract all pending/needs_revision claims for the subquestion from claim-ledger.md, each with claim_id and claim_text}

## Search-result files (read with Read and Glob)
Directory: {absolute workspace path}/search-results/Q{n}/
Read every .md file in that directory.
If a file does not exist, skip it; do not abort because of missing files.

## Cross-check rules (strict)
For each claim:
1. Find the QUOTE or NUMBER original text in the search results and try to refute the claim
2. Numbers: verbatim check. 15% != about 15% != nearly 15%. Any mismatch = NOT_SUPPORTED
3. Degree words: original says "grew" but claim says "grew substantially" = PARTIAL
4. Tone: original says "may" but claim says "will" = NOT_SUPPORTED
5. Cross-source splicing: if the claim requires two different sources to be supported = PARTIAL and mark COMPOSITE

If no verbatim correspondence or explicitly supporting original text can be found, you must rule NOT_SUPPORTED.
Replacing support with "similar meaning" is forbidden.
Inventing plausible supporting evidence is forbidden.

## Output format (one block per claim, strictly follow this format)
CLAIM_ID: {claim_id}
VERDICT: SUPPORTED / PARTIAL / NOT_SUPPORTED
QUOTE_ID: {supporting quote_id} or NONE
ISSUE: {if not SUPPORTED, describe the issue type and details}
---
```

---

## Step 6: Process Subagent Results

Update the `status` field of claim-ledger.md:

| Subagent verdict | Action | claim status |
|-----------------|------|-------------|
| SUPPORTED | keep | **approved** |
| PARTIAL | revise claim_text to match the original text (create new claim_id) or weaken the tone | **needs_revision** -> approved after revision |
| NOT_SUPPORTED | trigger one additional search. If still unsupported after searching -> delete | **rejected** |

**Fail-Fast:** each claim gets at most 2 additional searches. A third NOT_SUPPORTED -> rejected immediately, no more loops.

---

## Step 7: Pro-Con Debate

1. List advocate approved claims + Bedrock scores
2. List critic approved claims + Bedrock scores
3. Read `prompts/ref-challenge-checklist.md` and **pick applicable items** from the 22 (not all need to pass; skip those marked N/A)
4. A challenge holds -> record in the "Unresolved contradictions" section of `workspace/gap-log.md`, trigger additional search
5. **If perspective sources contain primary facts or substantive counter-evidence** -> promote to advocate/critic claim

---

## Step 8: Update Coverage Checklist

Read `workspace/coverage.chk` and update each item using the Edit tool:

| Update rule | Action |
|---------|------|
| Facet+role has >=1 approved claim | `[ ]` -> `[x]` + `evidence_found (S{ids})` |
| Searched but no approved claim (at least 2 attempts) | `[ ]` -> `[x]` + `searched_2x_no_evidence` |
| Searched but additional search still in progress | keep `[ ]` + `in_progress` |

Also update `workspace/gap-log.md` (weak evidence, missing perspectives).

---

## Step 9: Iteration Decision

**Conditions to trigger a new iteration:** insufficient data / unresolved contradictions / coverage still has required `[ ]` / Subagent NOT_SUPPORTED items not yet handled

**Plan Reflection at the end of each round:**
1. What was newly discovered?
2. Extract new terms from search results -> rewrite next-round queries
3. Update gap-log.md
4. Re-run 4-dim quality; delta = 0 -> saturation signal

**Beast Mode (when search budget hits 80%):**
- Stop opening new subquestions or new facets
- **But do not stop backfilling required coverage items**
- Reserve the remaining 20% for Phase 2-3

**Return to Phase 1a to keep searching -> after searching, return to Phase 1b-A to validate -> loop until stop condition is met**

---

## Structured Stopping Conditions (all must hold to stop)

```
[ ] 1. All required items in coverage.chk are marked [x]?
       -> any unmarked required item = stopping is forbidden

[ ] 2. Each searched_2x_no_evidence item really came from at least 2 different queries?
       -> < 2 -> return to Phase 1a for additional search

[ ] 3. Each subquestion has at least 1 approved claim from advocate and 1 from critic?
       -> either side = 0 and not searched_2x_no_evidence -> additional search

[ ] 4. Every high-risk claim (numeric/comparative/causal/forecast)
       has completed Grounding + URL check?
       -> list the high-risk claims and their validation status

[ ] 5. All "Unresolved contradictions" in gap-log.md are resolved or marked [BLOCKER]?
       -> unresolved contradictions > 0 -> must be resolved or marked BLOCKER before stopping

[ ] 6. All rejected claims handled (confirmed rejected or approved after additional search)?

-> all OK = proceed to Phase 2
-> any fail = return to Phase 1a/Step, continue
```

---

## Update execution-log.md

At the end of each round, append:

```
### Round {R} complete
Search: {N} calls (cumulative {total}/{budget} = {pct}%)
Deep-read: {N} sources
Grounding: {passed} pass / {weak} weak / {filtered} filtered
Claim Ledger: {approved} approved / {rejected} rejected / {pending} pending
4-dim quality: {pass}/4 (Fail: {failed dimensions})
1b-B triggered? {yes/no}
Coverage: {checked}/{total_required} required items done
New URL growth rate: {pct}% | approved claim growth rate: {pct}%
```
