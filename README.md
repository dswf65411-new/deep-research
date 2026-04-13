# Deep Research

LangGraph 打造的深度研究 pipeline。多 phase workflow + grounding 驗證 + 結構化溯源鏈，產出可審計的研究報告。

## Pipeline 架構

```
使用者輸入 topic + ref files (text/image/PDF)
        │
        ▼
┌─ Clarification（main.py，graph 外）─────────────┐
│  多輪澄清 Q&A（LLM 提問 → 使用者回答 → Judge 評估）│
└──────────────────────────────────────────────────┘
        │ clarifications + refs
        ▼
Phase 0  (Plan + Research Brief)
  ├─ 研究計畫 + 子問題 DAG + 覆蓋檢查清單
  └─ synthesize_research_topic → 全流程固定 context
        │
        ▼
  [Human Approval] — ask mode 暫停等使用者確認
        │
        ▼
Phase 1a (Search)    → 多引擎搜尋 + 深讀 + 自動 tier 分級 + THIN_CONTENT 偵測
        │                 └ iterative expansion：抽新實體回補下輪
        ▼
Phase 1b (Verify)    → Grounding + Relevance + Attack Agent（條件式觸發）
        │                 └ 低分 → trigger_fallback → 回 Phase 1a 補搜（最多 2 次）
        ▼
Phase 2  (Integrate) → approved claims → 段落寫作 + 矛盾裁決
        │                 └ 逐句 [Q1-C1] inline citation
        ▼
Phase 3  (Report)    → Statement ledger + 最終 audit + Coverage Sanity Check
```

## 品質保證機制

| 機制 | 檔案 | 功能 |
|------|------|------|
| Bedrock 分數門檻 | `validators.py::validate_claims_for_phase2` | `bedrock_score < 0.3` 的 claim 禁止進 Phase 2 |
| Metadata 過濾 | `validators.py::_is_metadata_claim` | 街道地址、SEO 樣板、法律聲明等 regex 過濾 |
| LLM Relevance Check | `phase1b.py::_run_relevance_checks` | LLM 判斷 claim 是否回答研究子問題，off-topic → rejected |
| Claim 近似去重 | `harness/claim_dedup.py` | normalize + difflib ratio ≥ 0.92 視為重複 |
| Source Tier 分級 | `harness/source_tier.py` | T1 官方 / T2 學術 / T3 專業媒體 / T4 部落格 / T5 UGC / T6 不可用 |
| 跨輪 URL 去重 | `state.py::fetched_urls` + `phase1a.py` | `operator.add` reducer 累積已抓 URL |
| Budget 守衛 | `phase0.py::DEPTH_CONFIG["min_budget_per_sq"]` | 每 SQ 保留最低預算，防止後段餓死 |
| Domain 偏差偵測 | `phase1a.py::_log_domain_bias` | 單一 domain > 30% → gap-log 警告 + 🟠CONFLICTING |
| Fallback Loop | `phase1b.py::trigger_fallback_node` | 低 grounding → 回 Phase 1a 聚焦補搜，最多 2 次 |
| 覆蓋率交叉核對 | `phase3.py::_compute_coverage_note` + `_find_uncovered_keywords` | SQ 覆蓋率 + 任務書明確工具覆蓋率雙層檢核 |
| 溯源鏈驗證 | `validators.py::validate_traceability_chain` | statement → claim_id → quote_id → source_id 鏈斷即標 |

## Quick Start

```bash
git clone https://github.com/dswf65411-new/deep-research.git
cd deep-research
make init
```

`make init` will:
1. Install pyenv（if missing）
2. Install Python 3.13.12
3. Create venv + install dependencies（via `uv`）
4. Prompt for API keys
5. Optionally install Claude Code / Gemini CLI skills

## API Keys

### LLM（至少需要一組）

自動偵測優先序：Claude > Gemini > OpenAI。也可用 `--model` 指定。

| Provider | Key | 取得 |
|----------|-----|------|
| Claude | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

目前各家設定的 strong/fast model（見 `llm.py`）：
- Claude: `claude-sonnet-4-6` / `claude-haiku-4-5-20251001`
- Gemini: `gemini-2.5-pro` / `gemini-2.5-flash`
- OpenAI: `gpt-4o` / `gpt-4o-mini`

### 搜尋引擎（兩組皆需）

| Service | Key | 取得 |
|---------|-----|------|
| Brave Search | `BRAVE_API_KEY` | https://brave.com/search/api/ |
| Serper（Google） | `SERPER_API_KEY` | https://serper.dev/ |

### Grounding（內建於 repo）

Grounding 腳本位於 `grounding_scripts/`：

- **Bedrock**：需 AWS 帳號 + Bedrock 服務啟用（`aws configure`）
- **MiniCheck / NeMo**：使用專案 venv 自動執行

預設走 LLM grounding（跨語言支援），CLI tools 為 fallback。

## Usage

### CLI

```bash
# 預設：deep mode，自動偵測 LLM
.venv/bin/python3 main.py "LangGraph vs CrewAI"

# Quick mode + Gemini
.venv/bin/python3 main.py "台灣半導體產業現況" --quick --model gemini

# Standard mode + 自訂預算
.venv/bin/python3 main.py "AI Agent 框架比較" --standard --budget 80

# 自動批准（跳過 human approval）
.venv/bin/python3 main.py "量子計算進展" --noask

# 附加參考文件（text/image/PDF）
.venv/bin/python3 main.py "AI 框架比較" --ref report.pdf arch.png notes.md

# 指定完整模型名 + 調整 context 閾值
.venv/bin/python3 main.py "量子計算" --model gemini-2.5-pro --context-threshold 0.5
```

### Claude Code skill

```
/deep_research LangGraph vs CrewAI
/deep_research 台灣半導體產業現況 --quick --model gemini
```

### Gemini CLI skill

```
/deep_research topic:"LangGraph vs CrewAI"
/deep_research topic:"台灣半導體產業現況" flags:"--quick --model gemini"
```

## Depth Modes

| Mode | Flag | Budget | Iterations | Sub-questions | Min/SQ |
|------|------|--------|------------|---------------|--------|
| Quick | `--quick` | 30 | 1 | 1-2 | 3 |
| Standard | `--standard` | 60 | 2 | 2-5 | 6 |
| Deep | `--deep`（預設） | 150 | 5 | 5-10 | 12 |

`Min/SQ` = 每個子問題保留的最低搜尋次數，防止後段 SQ 預算被前段吃光。

## Workspace 結構

每次研究建立獨立 workspace：

```
workspaces/<date>_<topic>/
├── research-brief.md          # Phase 0：完整任務書
├── phase0-plan.md             # Phase 0：研究計畫 + DAG
├── coverage.chk               # Phase 0：SQ 覆蓋檢查清單（## Q1: 描述）
├── clarifications.md          # Clarify phase 的 Q&A 記錄
├── source-registry.md         # Phase 1a：來源登記（含 tier、url_status）
├── search-results/
│   ├── Q1/S001.md            # 每個來源的 quotes / numbers / claims metadata
│   ├── Q1/S001_raw.md        # 原始 fetch 全文（< 25K chars）
│   └── ...
├── claim-ledger.md            # Phase 1b：每個 claim 的 bedrock_score、status
├── gap-log.md                 # 缺口日誌（UNREACHABLE、MISSING、BIAS WARNING、新實體）
├── report-sections/
│   ├── q1_section.md         # Phase 2：每個 SQ 的段落（含 [Q1-C1] inline cites）
│   └── ...
├── statement-ledger.md        # Phase 3：每句 statement 的 claim_ids 對應
├── execution-log.md           # 完整執行記錄
└── final-report.md            # 最終報告（含覆蓋率完整性報告）
```

## Project Structure

```
deep-research/
├── main.py                     # CLI entry point（含 clarify 互動）
├── deep_research/
│   ├── config.py               # 專案路徑
│   ├── context.py              # Context window 管理（BM25 + iterative refinement + caching）
│   ├── llm.py                  # LLM factory（Claude / Gemini / OpenAI，strong/fast tier）
│   ├── graph.py                # Main StateGraph 拓樸
│   ├── state.py                # Pydantic models + TypedDict state
│   ├── nodes/
│   │   ├── phase0.py           # Clarify + planning + research brief
│   │   ├── phase1a.py          # Planner + Search + Extractor + Registry
│   │   ├── phase1b.py          # Grounding + Relevance + Quality + Attack Agent
│   │   ├── phase2.py           # Integrator（含 biased-source marking）
│   │   └── phase3.py           # Statement ledger + Final audit + Coverage sanity
│   ├── harness/
│   │   ├── gates.py            # 4D quality gate（actionability/freshness/plurality/completeness）
│   │   ├── validators.py       # Tier 1 硬規則 + metadata filter + span-based index
│   │   ├── source_tier.py      # T1-T6 自動分級
│   │   └── claim_dedup.py      # Normalize + fuzzy ratio 近似去重
│   └── tools/
│       ├── search.py           # Brave / Serper 直接 HTTP
│       ├── grounding.py        # Bedrock / MiniCheck / NeMo CLI wrapper
│       └── workspace.py        # Workspace I/O 工具
├── grounding_scripts/          # Grounding CLI tools
├── prompts/                    # Phase instruction 檔案
├── tests/                      # 222 pytest（11 test files）
└── setup.sh                    # 一鍵安裝
```

## 測試

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

測試覆蓋關鍵 invariant：

| Test file | 驗證項目 |
|-----------|----------|
| `test_bedrock_flow.py` | Bedrock score 回寫 + >= 0.3 門檻 |
| `test_budget_guard.py` | 每 SQ 最低預算保留 |
| `test_claim_dedup.py` | Normalize + ratio 去重 |
| `test_coverage_sanity.py` | Keyword + SQ 覆蓋率交叉核對 |
| `test_discovery_queries.py` | Planner 發現面 query 模板 |
| `test_fallback_loop.py` | 低 grounding 觸發補搜 |
| `test_iterative_expansion.py` | 新實體抽取 + 回補下輪 |
| `test_metadata_filter.py` | 地址 / SEO / 法律樣板 regex 過濾 |
| `test_relevance_filter.py` | LLM relevance 判定 + dim_scores 整合 |
| `test_source_tier.py` | T1-T6 domain 分級 |
| `test_taiwan_whitelist.py` | 台灣權威 domain 自動升 T3 |
| `test_thin_content.py` | < 500 chars → THIN_CONTENT 標記 |
| `test_url_dedup.py` | 跨輪 URL 去重 + UNREACHABLE 排除 |

## Output

`workspaces/<date>_<topic>/final-report.md`，包含：

- 摘要（LLM 合成）
- 詳細分析（逐 SQ 段落，每句附 `[Q1-C1]` inline citation + 信心等級）
- 引用來源總表
- **覆蓋率完整性報告（兩層）**：
  - 子問題覆蓋率（計畫 SQ vs approved claims）
  - 任務書明確提及工具／主題的覆蓋率（偵測漏抓的知名工具）
- 未解答問題與知識缺口（gap-log）

## 鐵律（違反即視為研究失敗）

1. Sub agent 禁止呼叫 MCP — 搜尋與驗證由主 Agent 執行，sub agent 只能讀本地檔案
2. 原文先行 — 所有事實必須來自搜尋原文（WebFetch / Serper scrape）逐字引用
3. 數字溯源 — ORIGINAL / NORMALIZED / DERIVED 三類強制標記
4. 溯源鏈完整 — 報告句 → claim_id → quote_id → source_id，鏈斷即刪

## License

MIT
