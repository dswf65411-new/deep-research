# Deep Research

A deep research pipeline built with LangGraph. Multi-phase workflow + grounding validation + structured citation chain, producing auditable research reports.

## Pipeline Architecture

```
User input: topic + ref files (text/image/PDF)
        │
        ▼
┌─ Clarification (main.py, outside the graph) ─────┐
│  Multi-round clarification Q&A (LLM asks → user answers → Judge evaluates) │
└──────────────────────────────────────────────────┘
        │ clarifications + refs
        ▼
Phase 0  (Plan + Research Brief)
  ├─ Research plan + subquestion DAG + coverage checklist
  └─ synthesize_research_topic → fixed context for the entire pipeline
        │
        ▼
  [Human Approval] — ask mode pauses for user confirmation
        │
        ▼
Phase 1a (Search)    → Multi-engine search + deep-read + automatic tier classification + THIN_CONTENT detection
        │                 └ iterative expansion: extract new entities to backfill the next round
        ▼
Phase 1b (Verify)    → Grounding + Relevance + Attack Agent (conditionally triggered)
        │                 └ Low score → trigger_fallback → back to Phase 1a to search more (up to 2 times)
        ▼
Phase 2  (Integrate) → approved claims → paragraph writing + contradiction arbitration
        │                 └ Per-sentence [Q1-C1] inline citation
        ▼
Phase 3  (Report)    → Statement ledger + final audit + Coverage Sanity Check
```

## Quality Assurance Mechanisms

| Mechanism | File | Function |
|------|------|------|
| Bedrock score threshold | `validators.py::validate_claims_for_phase2` | Claims with `bedrock_score < 0.3` are blocked from entering Phase 2 |
| Metadata filter | `validators.py::_is_metadata_claim` | Regex filter for street addresses, SEO boilerplate, legal disclaimers, etc. |
| LLM Relevance Check | `phase1b.py::_run_relevance_checks` | LLM judges whether the claim answers the research subquestion; off-topic → rejected |
| Claim near-duplicate dedup | `harness/claim_dedup.py` | normalize + difflib ratio ≥ 0.92 treated as duplicate |
| Source Tier classification | `harness/source_tier.py` | T1 official / T2 academic / T3 professional media / T4 blog / T5 UGC / T6 unusable |
| Cross-round URL dedup | `state.py::fetched_urls` + `phase1a.py` | `operator.add` reducer accumulates fetched URLs |
| Budget guard | `phase0.py::DEPTH_CONFIG["min_budget_per_sq"]` | Reserves a minimum budget per SQ to prevent later stages from starving |
| Domain bias detection | `phase1a.py::_log_domain_bias` | Single domain > 30% → gap-log warning + 🟠CONFLICTING |
| Fallback Loop | `phase1b.py::trigger_fallback_node` | Low grounding → back to Phase 1a for focused re-search, up to 2 times |
| Coverage cross-check | `phase3.py::_compute_coverage_note` + `_find_uncovered_keywords` | Two-layer check: SQ coverage + brief-mentioned tool coverage |
| Citation chain validation | `validators.py::validate_traceability_chain` | statement → claim_id → quote_id → source_id; broken chain gets flagged |

## Quick Start

```bash
git clone https://github.com/dswf65411-new/deep-research.git
cd deep-research
make init             # or: scripts/setup.sh
```

`make init` will:
1. Check Python >=3.12 is available
2. Create `.venv/` (if missing)
3. Upgrade pip + install project with `[dev]` extras (runtime + pytest + pytest-asyncio)
4. Run a pytest collection sanity check

For API key setup, see the "API Keys" section below.

## API Keys

### LLM (at least one required)

Auto-detection priority: Claude > Gemini > OpenAI. You can also specify via `--model`.

| Provider | Key | Obtain |
|----------|-----|------|
| Claude | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

Currently configured strong/fast models per provider (see `llm.py`):
- Claude: `claude-sonnet-4-6` / `claude-haiku-4-5-20251001`
- Gemini: `gemini-2.5-pro` / `gemini-2.5-flash`
- OpenAI: `gpt-4o` / `gpt-4o-mini`

### Search Engines (both required)

| Service | Key | Obtain |
|---------|-----|------|
| Brave Search | `BRAVE_API_KEY` | https://brave.com/search/api/ |
| Serper (Google) | `SERPER_API_KEY` | https://serper.dev/ |

### Grounding (bundled in the repo)

Grounding scripts are located in `grounding_scripts/`:

- **Bedrock**: requires an AWS account + Bedrock service enabled (`aws configure`)
- **MiniCheck / NeMo**: run automatically using the project venv

LLM grounding is used by default (cross-language support); CLI tools serve as fallback.

## Usage

### CLI

```bash
# Default: deep mode, auto-detect LLM
.venv/bin/python3 main.py "LangGraph vs CrewAI"

# Quick mode + Gemini
.venv/bin/python3 main.py "Taiwan semiconductor industry status" --quick --model gemini

# Standard mode + custom budget
.venv/bin/python3 main.py "AI Agent framework comparison" --standard --budget 80

# Auto approve (skip human approval)
.venv/bin/python3 main.py "Quantum computing progress" --noask

# Attach reference files (text/image/PDF)
.venv/bin/python3 main.py "AI framework comparison" --ref report.pdf arch.png notes.md

# Specify the full model name + adjust context threshold
.venv/bin/python3 main.py "Quantum computing" --model gemini-2.5-pro --context-threshold 0.5

# Plan-only: skip search, just produce phase0 plan
.venv/bin/python3 main.py "topic" --plan-only

# Feed a user project directory so the brief reads local CLAUDE.md / README / source
.venv/bin/python3 main.py "topic" --project-dir /path/to/my-project

# Hard stop limits — graceful partial report when exceeded
.venv/bin/python3 main.py "topic" --max-time-minutes 20 --max-cost-usd 5.00

# Abort a running workspace (drops a .abort marker; next check exits cleanly)
.venv/bin/python3 main.py --abort workspaces/2026-04-15_my-topic
```

### Env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEP_RESEARCH_GROUNDING_CONCURRENCY` | `5` | Max concurrent LLM grounding calls (clamped to `[1, 32]`). Bump to `10` on high-throughput providers; reduce on 429s. |

### Claude Code skill

```
/deep_research LangGraph vs CrewAI
/deep_research Taiwan semiconductor industry status --quick --model gemini
```

### Gemini CLI skill

```
/deep_research topic:"LangGraph vs CrewAI"
/deep_research topic:"Taiwan semiconductor industry status" flags:"--quick --model gemini"
```

## Depth Modes

| Mode | Flag | Budget | Iterations | Sub-questions | Min/SQ |
|------|------|--------|------------|---------------|--------|
| Quick | `--quick` | 30 | 1 | 1-2 | 3 |
| Standard | `--standard` | 60 | 2 | 2-5 | 6 |
| Deep | `--deep` (default) | 150 | 5 | 5-10 | 12 |

`Min/SQ` = the minimum number of searches reserved per subquestion, preventing later SQ budgets from being consumed by earlier ones.

## Workspace Structure

Each research run creates an independent workspace:

```
workspaces/<date>_<topic>/
├── research-brief.md          # Phase 0: full research brief
├── phase0-plan.md             # Phase 0: research plan + DAG
├── coverage.chk               # Phase 0: SQ coverage checklist (## Q1: description)
├── clarifications.md          # Q&A log from the clarify phase
├── source-registry.md         # Phase 1a: source registry (with tier, url_status)
├── search-results/
│   ├── Q1/S001.md            # quotes / numbers / claims metadata for each source
│   ├── Q1/S001_raw.md        # raw fetched full text (< 25K chars)
│   └── ...
├── claim-ledger.md            # Phase 1b: bedrock_score, status for each claim
├── gap-log.md                 # Gap log (UNREACHABLE, MISSING, BIAS WARNING, new entities)
├── report-sections/
│   ├── q1_section.md         # Phase 2: paragraph per SQ (with [Q1-C1] inline cites)
│   └── ...
├── statement-ledger.md        # Phase 3: claim_ids mapping for each statement
├── execution-log.md           # Full execution log
└── final-report.md            # Final report (including coverage completeness report)
```

## Project Structure

```
deep-research/
├── main.py                     # CLI entry point (includes clarify interaction)
├── deep_research/
│   ├── config.py               # Project paths
│   ├── context.py              # Context window management (BM25 + iterative refinement + caching)
│   ├── llm.py                  # LLM factory (Claude / Gemini / OpenAI, strong/fast tier)
│   ├── graph.py                # Main StateGraph topology
│   ├── state.py                # Pydantic models + TypedDict state
│   ├── nodes/
│   │   ├── phase0.py           # Clarify + planning + research brief
│   │   ├── phase1a.py          # Planner + Search + Extractor + Registry
│   │   ├── phase1b.py          # Grounding + Relevance + Quality + Attack Agent
│   │   ├── phase2.py           # Integrator (with biased-source marking)
│   │   └── phase3.py           # Statement ledger + Final audit + Coverage sanity
│   ├── harness/
│   │   ├── gates.py              # 4D quality gate (actionability/freshness/plurality/completeness)
│   │   ├── validators.py         # Tier 1 hard rules + metadata filter + span-based index
│   │   ├── source_tier.py        # T1-T6 automatic classification
│   │   ├── claim_dedup.py        # Normalize + fuzzy ratio near-duplicate dedup
│   │   ├── claim_relevance.py    # Claim-level relevance_to_question scorer (offline, Jaccard + entity boost)
│   │   ├── source_mirror.py      # Cross-source mirror detector (arxiv ID / DOI / title Jaccard)
│   │   ├── stakeholder_collision.py  # Advocate-vs-critic contradiction detector
│   │   ├── self_eval.py          # Pipeline self-scoring + LLM follow-up suggestions
│   │   ├── review_gate.py        # Mid-pipeline /skip /refocus /inject-url command parser
│   │   ├── runtime_limits.py     # --max-time / --max-cost / --abort enforcement
│   │   ├── cost_tracker.py       # LLM call count / token / USD accounting
│   │   ├── secret_scanner.py     # PII / webhook / API key redaction (P0-6)
│   │   └── url_validator.py      # arxiv ID / URL existence check (P0-4)
│   └── tools/
│       ├── search.py             # Brave / Serper direct HTTP
│       ├── grounding.py          # Bedrock / MiniCheck / NeMo CLI wrapper
│       ├── workspace.py          # Workspace I/O tools
│       ├── arxiv_retriever.py    # arxiv API search (official endpoint)
│       └── github_search.py      # GitHub Code Search API
├── grounding_scripts/            # Grounding CLI tools
├── prompts/                      # Phase instruction files
├── tests/                        # 739 pytest across 54 test files
├── scripts/
│   ├── setup.sh                  # Idempotent dev-env bootstrap (venv + pip install -e ".[dev]")
│   ├── pytest_baseline.sh        # Store / diff pytest failure baselines
│   └── archive_workspaces.sh     # TTL-based workspace archival
└── Makefile                      # make init / test / baseline-save / baseline-diff / archive
```

## Testing

```bash
.venv/bin/python -m pytest tests/              # full 739-test suite (<1s)
make test                                       # same as above via Makefile

# Regression-check against a stored baseline
scripts/pytest_baseline.sh save                 # snapshot current failures
# ... make changes ...
scripts/pytest_baseline.sh diff                 # prints NEW red + NEWLY green only
```

Tests cover the key invariants. Highlights (54 files, 739 cases total):

| Area | Representative test files |
|------|---------------------------|
| Grounding | `test_bedrock_flow`, `test_grounding_coverage_assertion`, `test_grounding_concurrency`, `test_march_blind_recheck` |
| Claim quality | `test_claim_dedup`, `test_claim_relevance`, `test_relevance_filter`, `test_metadata_filter`, `test_off_topic_gap` |
| Source tiering | `test_source_tier`, `test_source_curator`, `test_source_mirror`, `test_taiwan_whitelist`, `test_zh_low_info_filter` |
| Phase integrity | `test_phase0_url_validation`, `test_phase2_section_writeback`, `test_phase3_statement_ledger_resilient`, `test_phase3_llm_sanity_check` |
| Search / budget | `test_budget_guard`, `test_url_dedup`, `test_discovery_queries`, `test_iterative_expansion`, `test_serper_scholar`, `test_scholar_auto_upgrade` |
| Runtime / CLI | `test_runtime_limits`, `test_review_gate`, `test_self_eval`, `test_clarify_staged`, `test_coverage_chk_sync`, `test_stakeholder_collision` |
| Security / validation | `test_secret_scanner`, `test_url_validator`, `test_arxiv_retriever`, `test_github_search` |

## Output

`workspaces/<date>_<topic>/final-report.md`, containing:

- Summary (LLM-synthesized)
- Detailed analysis (paragraph per SQ, each sentence with a `[Q1-C1]` inline citation + confidence level)
- Consolidated citation source table
- **Coverage completeness report (two layers)**:
  - Subquestion coverage (planned SQ vs approved claims)
  - Coverage of tools/topics explicitly mentioned in the brief (detects well-known tools that were missed)
- Unanswered questions and knowledge gaps (gap-log)

## Hard Rules (violation = research failure)

1. Sub agents must not call MCP — search and validation are executed by the main Agent; sub agents can only read local files
2. Source-first — every fact must come verbatim from the search source (WebFetch / Serper scrape)
3. Number traceability — ORIGINAL / NORMALIZED / DERIVED three categories are mandatorily tagged
4. Complete citation chain — report sentence → claim_id → quote_id → source_id; broken chain = delete

## License

MIT
