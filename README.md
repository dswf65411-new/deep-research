# Deep Research

LangGraph-based deep research workflow. Multi-phase pipeline with grounding verification.

```
使用者輸入 topic + ref files (text/image/PDF)
        │
        ▼
┌─ Clarification (main.py, graph 外) ─────────────┐
│  多輪澄清 Q&A（LLM 提問 → 使用者回答 → Judge 評估）│
└──────────────────────────────────────────────────┘
        │ clarifications + refs
        ▼
Phase 0 (Plan + Research Brief)
  ├─ 生成研究計畫
  └─ synthesize_research_topic → full_research_topic（全流程固定 context）
        │
        ▼
  [Human Approval] ─── ask mode 時暫停等使用者確認
        │
        ▼
Phase 1a (Search) → Phase 1b (Verify) → Phase 2 (Integrate) → Phase 3 (Report)
        ↑                    │
        └──── fail ──────────┘

Context Window 管理（context.py）：
  全塞 or Iterative Refinement（BM25 排序 + 分批送入 + prompt prefix caching）
  超過 30% threshold → 分批 │ 超過 100% → 自動切換最大 provider │ 仍超 → error
```

## Quick Start

```bash
git clone https://github.com/dswf65411-new/deep-research.git
cd deep-research
make init
```

`make init` will:
1. Install pyenv (if missing)
2. Install Python 3.13.12
3. Create venv and install dependencies
4. Prompt for API keys (see below)
5. Optionally install Claude Code / Gemini CLI skills

## API Keys

### LLM (at least one required)

Auto-detection priority: Claude > Gemini > OpenAI. Or specify with `--model`.

| Provider | Key | Get it from |
|----------|-----|-------------|
| Claude | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

### Search (both required)

| Service | Key | Get it from |
|---------|-----|-------------|
| Brave Search | `BRAVE_API_KEY` | https://brave.com/search/api/ |
| Serper (Google) | `SERPER_API_KEY` | https://serper.dev/ |

### Grounding (included in repo)

Grounding verification scripts are included in `grounding_scripts/`. They require:

- **Bedrock**: AWS account with Bedrock enabled. Run `aws configure` to set credentials.
- **MiniCheck**: Runs automatically using the project venv.
- **NeMo**: Runs automatically using the project venv.

No additional API keys needed for grounding — just AWS credentials for Bedrock.

## Usage

### CLI (direct)

```bash
# Default: deep mode, auto-detect LLM
.venv/bin/python3 main.py "LangGraph vs CrewAI"

# Quick mode with Gemini
.venv/bin/python3 main.py "台灣半導體產業現況" --quick --model gemini

# Standard mode with budget override
.venv/bin/python3 main.py "AI Agent 框架比較" --standard --budget 80

# Auto-approve research plan (no confirmation prompt)
.venv/bin/python3 main.py "量子計算進展" --noask

# 附加參考文件（支援文字、圖片、PDF）
.venv/bin/python3 main.py "AI Agent 框架比較" --ref report.pdf arch.png notes.md

# 指定完整模型版號 + 調整 context 閾值
.venv/bin/python3 main.py "量子計算進展" --model gemini-2.5-pro --context-threshold 0.5
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

| Mode | Flag | Search Budget | Iterations | Sub-questions |
|------|------|--------------|------------|---------------|
| Quick | `--quick` | 30 | 1 | 1-2 |
| Standard | `--standard` | 60 | 2 | 2-5 |
| Deep | `--deep` (default) | 150 | 5 | 5-10 |

## Project Structure

```
deep-research/
├── main.py                  # CLI entry point
├── deep_research/
│   ├── config.py            # Project paths (all relative)
│   ├── context.py           # Context window management (Iterative Refinement + BM25 + caching)
│   ├── llm.py               # LLM factory (Claude/OpenAI/Gemini)
│   ├── graph.py             # Main StateGraph definition
│   ├── state.py             # Pydantic models + TypedDict states
│   ├── nodes/               # Phase implementations
│   │   ├── phase0.py        # Clarify + planning + research brief synthesis
│   │   ├── phase1a.py       # Search + deep-read
│   │   ├── phase1b.py       # Grounding verification subgraph
│   │   ├── phase2.py        # Conflict resolution + integration
│   │   └── phase3.py        # Report generation + audit
│   ├── harness/
│   │   ├── gates.py         # Deterministic gate checks
│   │   └── validators.py    # Iron rules enforcement
│   └── tools/
│       ├── search.py        # Brave/Serper direct HTTP API
│       ├── grounding.py     # CLI wrappers for verification
│       └── workspace.py     # Workspace file operations
├── grounding_scripts/       # Verification CLI tools
│   ├── bedrock-guardrails.py
│   ├── minicheck.py
│   ├── nemo-guardrails.py
│   └── urlhealth.py
├── prompts/                 # Phase instruction files
├── setup.sh                 # One-command setup
├── requirements.txt         # Pinned dependencies
└── .env.example             # API key template
```

## Output

Research reports are saved to `workspaces/<date>_<topic>/final-report.md`.

## License

MIT
