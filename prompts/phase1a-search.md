# Phase 1a: Search + Deep-Read + Verbatim Transcription

**Input:** `workspace/phase0-plan.md` + `workspace/coverage.chk`
**Output:** `workspace/search-results/Q{n}/S{id}.md` + `workspace/source-registry.md` + `workspace/execution-log.md`
**Completion condition:** all search results archived, source-registry built -> proceed to Phase 1b

**This Phase does only 4 things:**
1. Generate queries from coverage gaps
2. Search and record sources
3. Deep-read and transcribe original text verbatim
4. Write all results to workspace files

---

## Search Engines

**Core (used every time, via direct HTTP API calls):**

| Engine | Purpose |
|------|------|
| Brave Search API (EN) | Independent English index |
| Serper API (gl=tw, hl=zh-TW) | Traditional Chinese Google |
| Serper API (EN) | English Google |

**Academic (add for technical topics):**
Serper API + `site:semanticscholar.org` or `site:arxiv.org`

**Extended languages (by topic):** read `prompts/ref-multilingual.md`

**China information:** Serper (gl=cn, hl=zh-CN) + Baidu Search API in parallel

---

## Step 1: Read the Plan + Initialize

1. Read `workspace/phase0-plan.md` (focus on the structured header and the subquestion DAG)
2. Read `workspace/coverage.chk`
3. Create directories: `workspace/search-results/Q1/`, `Q2/` ... etc.
4. Initialize `workspace/source-registry.md`:

```markdown
# Source Registry

| source_id | url | title | fetched_title | tier | url_status | date | engines | roles | subquestion |
|-----------|-----|-------|--------------|------|------------|------|---------|-------|-------------|
```

5. Initialize `workspace/execution-log.md`:

```markdown
# Execution Log
**Research topic:** {topic}
**Start time:** {timestamp}
**Search budget:** 0 / {total}

## Already-searched query list

## Round 1
```

---

## Step 2: Generate Round 1 Queries

**Progressive query generation (do not expand all queries at once):**

In round 1, each subquestion generates only the minimal set:
- advocate: 1 query family (EN + ZH versions)
- critic: 1 query family (EN + ZH versions)
- perspective: 0-1 (when there is a clear viewpoint)
- academic: 0-1 (for academic/technical topics)

Rules:
- 5-10 words per query
- Advocate and critic queries must have clear differentiation
- Semantic dedup against the already-searched query list in execution-log
- Append to execution-log immediately after generation

Queries for subsequent rounds are triggered by coverage gaps and the challenge checklist from Phase 1b.

---

## Step 3: Parallel Search

Follow the DAG order. Independent subquestions are searched in parallel in the same message.

Search combination for each query family:
```
EN query  -> WebSearch + Brave                          (2 calls)
ZH query  -> WebSearch + Serper(gl=tw, hl=zh-TW)        (2 calls)
Academic  -> Serper(site:arxiv.org/semanticscholar.org) (1-2 calls)
Extended  -> per ref-multilingual.md                    (by topic)
```

**After every search, update the search count in execution-log.**

---

## Step 4: URL Priority Ranking

Merge and dedupe all search results, then score by these factors:

| Factor | Bonus |
|------|------|
| Cross-engine hit | +2/engine |
| Cross-role hit (appears in both advocate and critic) | +5 |
| Domain Authority (T1-T2: +3, T3: +2, T4: +1) | per tier |
| Freshness (per freshness SLA) | +3/+2/+1 |

**Deep-read quota (per subquestion per round):**
- advocate: 2 sources
- critic: 2 sources
- perspective: 1 source
- overflow: up to 2 extra (only if the top 5 still leave a critical gap)

**Squeezing critic to 0 via total ranking is forbidden.**

---

## Step 5: URL Liveness Verification

All URLs queued for deep-read -> call urlhealth CLI via Bash (zero exceptions):

```bash
echo '{"urls":["URL1","URL2",...]}' | .venv/bin/python3 grounding_scripts/urlhealth.py --cli
```

| Status | Action |
|------|------|
| LIVE | continue deep-read |
| STALE | citable, attach Wayback URL |
| LIKELY_HALLUCINATED | **remove immediately** |
| UNKNOWN | retry once via Serper Scrape API |

---

## Step 6: Deep-Read + Verbatim Transcription

**Three-tier fetch (try in order):**
1. **WebFetch** (preferred, preserves full structure)
2. **Serper Scrape API** (fallback, plain text)
3. Both fail -> mark the URL `[UNREACHABLE]`, record in `workspace/gap-log.md`, do not write QUOTE/NUMBER based on search snippets alone

**Bedrock API throttling handling:** on 429 or throttling, wait 3 seconds and retry, up to 3 times. Continued failure -> read fallback.md.

**Verbatim transcription rules:**

1. Locate the paragraph relevant to the subquestion
2. Copy **at most 3 key sentences** verbatim (pick the ones with the strongest evidential value)
3. Numbers must be copied together with the original full sentence
4. Fixed format (each quote/number has a unique ID):

```
QUOTE[S{id}-Q{n}]: "{sentence copied verbatim from the original text}" — {url}
NUMBER[S{id}-N{n}]: {number} {unit} — Original: "{original full sentence containing the number}" — {url}
```

**Forbidden:**
- Paraphrasing after reading the original text
- Combining multiple numbers to derive a calculation (unless the original text did the calculation itself)
- Drawing conclusions from the WebFetch summary alone

---

## Step 7: Write to Workspace

**7a. Each source saved as its own file** `workspace/search-results/Q{n}/S{id}.md`:

```markdown
# Source S{id}: {title}

- URL: {url}
- Fetched Title: {actual page title}
- URL Status: {LIVE/STALE}
- Tier: T{n}
- Fetch Date: {YYYY-MM-DD}
- Engines: {brave, serper_tw, ...}
- Role: {advocate/critic/perspective:{name}}
- Subquestion: Q{n}

## Verbatim Quotes
QUOTE[S{id}-Q1]: "{original text}"
QUOTE[S{id}-Q2]: "{original text}"
NUMBER[S{id}-N1]: {number} — Original: "{original sentence}"
```

**7b. Update source-registry.md** (append one row)

**7c. Update execution-log.md**

---

## Gate Check (all must pass to exit Phase 1a)

```
[ ] 1. Every deep-read source has its own file (workspace/search-results/Q{n}/S{id}.md)?
[ ] 2. source-registry.md updated?
[ ] 3. execution-log.md search count correct?
[ ] 4. Every QUOTE/NUMBER has a unique ID?
[ ] 5. Every UNREACHABLE URL recorded in gap-log.md?
[ ] 6. coverage.chk reflects search progress?
-> all OK -> proceed to Phase 1b
```
