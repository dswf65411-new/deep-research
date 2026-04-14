# Phase 0: Clarification + Research Planning

**Input:** user's research topic
**Output:** `workspace/phase0-plan.md` + `workspace/coverage.chk` + `workspace/gap-log.md`
**Completion condition:** user has confirmed the research plan

---

## Step 1: Clarifying Questions

Confirm with AskUserQuestion. Ask fewer questions when intent is clear, more when ambiguous. For each question, explain "why we need to know this".

1. **Research goal:** make a decision? write a report? learn? solve a problem?
2. **Desired output:** comparison table? recommendation? objective presentation?
3. **Scope boundaries:** what is included/excluded? time range?
4. **Background knowledge:** what is already known? what has been ruled out?
5. **Specific preferences:** tech stack, region, language, sources that must be covered?

---

## Step 2: Topic Classification + Research Mode

**2a. Topic classification:**

| Topic type | Criteria | Phase 0 requirements |
|------|---------|-------------|
| **Single-point fact check** | subquestions < 3, no dependencies | Simplified: only goal, scope, source priority, hallucination anchors. Full DAG not required. |
| **Comparison/decision** | explicit options to compare | Full pipeline: DAG + Adversarial |
| **Trend/policy** | involves timeline or multiple stakeholders | Full pipeline + Multi-Stakeholder/Temporal |

**2b. Research mode selection (default Adversarial, can combine 1-2):**

| Research goal | Mode | Phase 1 strategy |
|---------|------|-------------|
| Make a decision | **Adversarial** | pro-con debate |
| Trend analysis | **Temporal** | history -> current -> forecast |
| Explore new field | **Funnel** | broad scan -> deep dive -> synthesis |
| Policy/impact assessment | **Multi-Stakeholder** | N parties analyzed separately -> synthesis |
| Comparative evaluation | **Adversarial + Matrix** | debate + comparison matrix |

---

## Step 3: Query Enrichment

**3a. Specificity Maximization:**
- Specified dimensions -> list them
- Unspecified but important dimensions -> list them + note how they will be handled

**3b. Source Prioritization:**
official documents > original papers > industry reports > technical blogs > community discussion

**3c. PICO Framing:**
- Population/Problem: research target/problem
- Intervention: the solution/technology being evaluated
- Comparison: comparison baseline/alternatives
- Outcome: desired results/metrics

**3d. Freshness SLA (set per topic, not one-size-fits-all):**

| Claim type | Default window | Adjustable |
|-----------|---------|--------|
| Numeric (price, performance, market share) | 12 months | user may loosen |
| Policy/regulation | 24 months | — |
| Background/theory | 36 months | — |
| Historical primary source | **exempt** | — |

**3e. Anti-Hallucination Anchors:**
- Places prone to hallucination (numbers/causation/trends)
- Items that must be verified against official sources

---

## Step 4: Perspective Discovery

Use 1-2 WebSearch calls to search `{topic} perspectives` or `{topic} stakeholders impact`.

| Perspective | Representative group | Core concerns | Dedicated search angle |
|------|---------|---------|------------|
| (at least 3) | | | |

**Important:** perspective search results default to supplementary viewpoints; but if they contain verifiable primary facts or substantive counter-evidence, promote them to advocate/critic claims participating in the debate.

---

## Step 5: Subquestion DAG

(Single-point fact-check topics may skip this and list 1-2 subquestions directly)

```
[independent] Q1: {description} ──┐
[independent] Q2: {description} ──┼── [depends on Q1+Q2] Q4: {description}
[independent] Q3: {description} ──┘
```

Define **facets** (search facets) for each subquestion:

```
Q1:
  facets: [benchmark, adoption, cost, risks]
  must_cover_roles: [advocate, critic]
```

---

## Step 6: Search Strategy

Design per subquestion, but **do not expand all queries in round 1**:

```
### Q{n}: {subquestion}

Round 1 (minimal set):
- advocate: 1 query family (EN + ZH)
- critic: 1 query family (EN + ZH)
- perspective: 0-1 (if there is a clear viewpoint)
- academic: 0-1 (only for academic/technical topics)

Subsequent rounds' queries must be triggered by:
- facets in coverage.chk not yet marked [x]
- gaps found via the challenge checklist
- Query Rewriting (extracting new terms from search results)
Expanding all queries in round 1 is forbidden.
```

Search breadth coverage (must ensure):

| Source type | Search method | Priority |
|---------|---------|--------|
| Academic papers | "survey"/"paper" + arxiv/scholar | High |
| Official docs | official docs/changelog/pricing | High |
| Industry reports | "benchmark"/"report" | High |
| Developer community | reddit/HN/GitHub | Medium |
| General community | blogs/Medium/Zhihu | Medium |
| News media | TechCrunch/The Verge etc. | Depends on topic |

---

## Step 7: Search Budget Allocation

```
Total budget: {30 / 60 / 150} calls

| Category | Share | Allocated to |
|------|------|--------|
| Core questions | 40% | high-uncertainty subquestions |
| Supporting questions | 20% | background subquestions |
| Perspective coverage | 10% | each perspective |
| Iteration reserve | 20% | Gap Queue + Query Rewriting |
| Phase 2-3 | 10% | contradiction adjudication + report verification |
```

---

## Step 8: Build Coverage Checklist + Gap Log

**This is the core mechanism against "stopping search too early". Use a simple checklist instead of a complex table to reduce formatting errors.**

Write `workspace/coverage.chk`:

```
# Coverage Checklist

## Q1: {subquestion}
- [ ] advocate:benchmark — not_started
- [ ] critic:benchmark — not_started
- [ ] advocate:adoption — not_started
- [ ] critic:risks — not_started
- [ ] perspective:regulator — not_started (optional)

## Q2: {subquestion}
- [ ] advocate:performance — not_started
- [ ] critic:performance — not_started
```

At the same time, initialize `workspace/gap-log.md`:

```markdown
# Gap Log

## Missing perspectives
(positions discovered during Phase 1 search but not yet covered)

## Weak evidence
(claims with only a single source)

## Unresolved contradictions
(both advocate and critic have approved claims with opposite conclusions)
```

**Rules:**
- Items without `(optional)` marker = required, must be marked `[x]` (evidence_found) or recorded as `searched_2x_no_evidence` before Phase 1 ends
- `(optional)` items searched best-effort, but do not block entering Phase 2
- **Deleting checklist items to raise coverage rate is forbidden**
- Update method: use the Edit tool to change `[ ]` to `[x]` and update the status description

---

## Step 9: Write to workspace + present

Write all results to `workspace/phase0-plan.md` in the format:

```markdown
# Research Plan

## Structured Header
- topic: {topic}
- mode: {Adversarial / Temporal / Funnel / Multi-Stakeholder / combination}
- depth: {Quick / Standard / Deep}
- budget: {30 / 60 / 150}
- freshness_sla:
  - numeric: {N} months
  - policy: {N} months
  - background: {N} months
  - historical_exempt: true/false
- subquestions: {N}
- perspectives: {N}
- total_coverage_units: {N} (required: {M})

## Query Enrichment
{PICO + source priority + anti-hallucination anchors}

## Stakeholder Perspectives
{list of perspectives + their concerns and search angles}

## Subquestion DAG
{subquestions + facets + dependencies + execution order}

## Search Strategy
{round 1 minimal query set + rules for triggering subsequent rounds}

## Budget Allocation
{allocation table}

## Hallucination High-Risk Areas
{which claims need special validation: numeric/causal/trend/comparison}

## Inclusion/Exclusion Criteria
- Included: {language, time, region, source types}
- Excluded: {exclusions}

Please confirm this research plan, or propose changes.
```

Also confirm `workspace/coverage.chk` and `workspace/gap-log.md` have been written.

Wait for user confirmation, then enter Phase 1a.

---

## Gate Check (all must pass to exit Phase 0)

```
[ ] 1. phase0-plan.md written? -> OK
[ ] 2. coverage.chk written? -> {N} items (required: {M})
[ ] 3. gap-log.md initialized? -> OK
[ ] 4. Research mode chosen? -> {mode}
[ ] 5. Subquestions decomposed (with facets and dependencies)? -> {N} subquestions
[ ] 6. User confirmed? -> OK
-> all OK -> proceed to Phase 1a
```
