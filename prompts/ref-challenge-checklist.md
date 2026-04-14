# 22-Item Challenge Checklist (mutually applicable to advocate and critic)

Used during Phase 1b Step 5 debate. Advocate and critic use this checklist to attack each other's claims; a sustained challenge triggers an additional search.

## I. Source Credibility (#1-4)

| # | Challenge | Decision | Action |
|---|------|------|------|
| 1 | **Conflict of interest**: does the source party have a commercial motive? | Official/partner of the evaluation target | Mark [COI], search independent third parties |
| 2 | **Low tier**: only T5-T6? | Compare against source-criteria.md | Search higher-tier sources |
| 3 | **Same origin**: multiple sources relaying the same original material? | Trace the citation chain | Count as 1 independent source, additional search |
| 4 | **Inaccessible**: URL won't open | WebFetch + browser both fail | Mark [UNVERIFIABLE] |

## II. Data Quality (#5-9)

| # | Challenge | Decision | Action |
|---|------|------|------|
| 5 | **Stale**: over 12 months? | Check the date | Search for latest data |
| 6 | **Sample too small**: benchmark too small? | Check sample size | Search larger-scale tests |
| 7 | **Opaque methodology** | No description of test method | Mark [METHODOLOGY UNCLEAR] |
| 8 | **Suspicious precision**: no statistical test? | No p-value/CI | Mark [UNVERIFIED PRECISION] |
| 9 | **Cherry-picking** | Contradicts other sources | Search the full dataset |

## III. Logical Reasoning (#10-14)

| # | Challenge | Decision | Action |
|---|------|------|------|
| 10 | **Causation fallacy**: correlation != causation | A+B together != A->B | Search causal experiments |
| 11 | **Survivorship bias** | Only success cases | Search failure cases |
| 12 | **Overgeneralization** | From minority to whole | Search counter-examples |
| 13 | **Straw-man** | Distorts opponent's claim | Return to original source |
| 14 | **Appeal to authority** | No substantive argument | Search substantive evidence |

## IV. Completeness (#15-19)

| # | Challenge | Decision | Action |
|---|------|------|------|
| 15 | **Geographic bias** | Sources concentrated in one region | Search other regions |
| 16 | **Language bias** | English only | Search other languages |
| 17 | **Incomplete timeline** | No history or trend | Search history/trend |
| 18 | **Missing stakeholders** | Impacted parties not covered | Search impact/regulation |
| 19 | **Alternatives not considered** | Important options missing | Search alternatives |

## V. Quantitative Support (#20-22)

| # | Challenge | Decision | Action |
|---|------|------|------|
| 20 | **Vague assessment**: no numbers | No specific numbers | Search benchmarks |
| 21 | **No baseline for comparison** | No baseline | Search competitor data |
| 22 | **Unit inconsistency** | Cannot compare | Add [DERIVED] column for conversion (keep ORIGINAL intact), list formula and source |

## Grounding Augmentation

When challenging, use Bedrock for quantitative comparison: "advocate grounding 0.85 vs critic grounding 0.62 -> advocate is better grounded".
