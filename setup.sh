#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.13.12"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  deep-research setup"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Install pyenv if missing
# ------------------------------------------------------------------
if ! command -v pyenv &>/dev/null; then
    echo "[1/5] Installing pyenv..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install pyenv
        else
            echo "ERROR: Homebrew not found. Install it first: https://brew.sh"
            exit 1
        fi
    else
        curl -fsSL https://pyenv.run | bash
    fi
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
else
    echo "[1/5] pyenv already installed ✓"
    eval "$(pyenv init -)"
fi

# ------------------------------------------------------------------
# 2. Install Python
# ------------------------------------------------------------------
if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "[2/5] Python ${PYTHON_VERSION} already installed ✓"
else
    echo "[2/5] Installing Python ${PYTHON_VERSION}..."
    pyenv install "${PYTHON_VERSION}"
fi

PYENV_PYTHON="$(pyenv prefix ${PYTHON_VERSION})/bin/python3"

# ------------------------------------------------------------------
# 3. Create venv and install dependencies
# ------------------------------------------------------------------
VENV_DIR="${PROJECT_DIR}/.venv"

if [[ -d "${VENV_DIR}" ]]; then
    echo "[3/5] venv already exists ✓"
else
    echo "[3/5] Creating venv..."
    "${PYENV_PYTHON}" -m venv "${VENV_DIR}"
fi

echo "      Installing dependencies..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${PROJECT_DIR}/requirements.txt"
echo "      Dependencies installed ✓"

# ------------------------------------------------------------------
# 4. Setup .env (API keys)
# ------------------------------------------------------------------
echo ""
echo "[4/5] Configuring API keys..."

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    echo "      .env already exists. Skipping key setup."
    echo "      (Delete .env and re-run to reconfigure)"
else
    echo ""
    echo "  ╔═══════════════════════════════════════════════════════╗"
    echo "  ║  需要以下 API keys（Press Enter 跳過非必要項目）      ║"
    echo "  ╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "  ── LLM（至少需要一個，優先順序：Claude > Gemini > OpenAI）──"
    echo "     取得方式："
    echo "     Claude:  https://console.anthropic.com/settings/keys"
    echo "     Gemini:  https://aistudio.google.com/apikey"
    echo "     OpenAI:  https://platform.openai.com/api-keys"
    echo ""

    read -rp "  ANTHROPIC_API_KEY (Claude):  " ANTHROPIC_KEY
    read -rp "  GEMINI_API_KEY (Gemini):     " GEMINI_KEY
    read -rp "  OPENAI_API_KEY (OpenAI):     " OPENAI_KEY

    # Validate: at least one LLM key
    if [[ -z "${ANTHROPIC_KEY}" && -z "${OPENAI_KEY}" && -z "${GEMINI_KEY}" ]]; then
        echo ""
        echo "  ⛔ ERROR: 至少需要一個 LLM API key。"
        echo "     Re-run: make init"
        exit 1
    fi

    echo ""
    echo "  ── 搜尋引擎（必填）──"
    echo "     Brave:   https://brave.com/search/api/"
    echo "     Serper:  https://serper.dev/"
    echo ""

    read -rp "  BRAVE_API_KEY:   " BRAVE_KEY
    read -rp "  SERPER_API_KEY:  " SERPER_KEY

    if [[ -z "${BRAVE_KEY}" ]]; then
        echo ""
        echo "  ⛔ ERROR: BRAVE_API_KEY 為必填。"
        echo "     申請：https://brave.com/search/api/"
        echo "     Re-run: make init"
        exit 1
    fi

    if [[ -z "${SERPER_KEY}" ]]; then
        echo ""
        echo "  ⛔ ERROR: SERPER_API_KEY 為必填。"
        echo "     申請：https://serper.dev/"
        echo "     Re-run: make init"
        exit 1
    fi

    cat > "${PROJECT_DIR}/.env" <<EOF
# LLM providers
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
GEMINI_API_KEY=${GEMINI_KEY}
OPENAI_API_KEY=${OPENAI_KEY}

# Search APIs
BRAVE_API_KEY=${BRAVE_KEY}
SERPER_API_KEY=${SERPER_KEY}
EOF

    # Show which LLM will be used
    if [[ -n "${ANTHROPIC_KEY}" ]]; then
        echo "      ✓ LLM: Claude（優先使用）"
    elif [[ -n "${GEMINI_KEY}" ]]; then
        echo "      ✓ LLM: Gemini（自動偵測）"
    else
        echo "      ✓ LLM: OpenAI（自動偵測）"
    fi
    echo "      .env created ✓"
fi

# ------------------------------------------------------------------
# 5. Install AI CLI skills (optional)
# ------------------------------------------------------------------
echo ""
echo "[5/5] AI CLI skill setup..."

VENV_PYTHON="${VENV_DIR}/bin/python3"
MAIN_PY="${PROJECT_DIR}/main.py"
INSTALLED_SKILLS=""

# --- Claude Code ---
if [[ -d "$HOME/.claude" ]]; then
    read -rp "  偵測到 Claude Code，是否安裝 /deep_research skill？(y/n) " INSTALL_CLAUDE
    if [[ "${INSTALL_CLAUDE}" =~ ^[Yy]$ ]]; then
        SKILL_DIR="$HOME/.claude/commands"
        mkdir -p "${SKILL_DIR}"
        cat > "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
---
description: "LangGraph 深度研究。Trigger on: '/deep_research'."
argument-hint: "<研究主題> [--quick | --standard | --deep] [--budget N] [--model claude|openai|gemini] [--noask]"
---

# /deep_research

使用 deep-research LangGraph pipeline 執行深度研究。

## 執行流程

用 Bash tool 執行，加 `--json` flag 啟用 turn-based protocol。

### Step 1: 啟動研究

```bash
SKILL_EOF
        # Insert dynamic paths
        echo "cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py \$ARGUMENTS --json 2>/dev/null" >> "${SKILL_DIR}/deep_research.md"
        cat >> "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
```

讀取 stdout 的 JSON 輸出。

### Step 2: 互動 Loop

根據 JSON 的 `status` 欄位處理：

**`NEEDS_INPUT` + `type: clarify`：**
- 從 `questions` 陣列取出所有問題
- 用對話問使用者每一個問題，收集回答
- 組成 JSON：`{"0": "第一題回答", "1": "第二題回答"}`
- 繼續執行：
```bash
SKILL_EOF
        echo "cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py --resume <thread_id> --answer '<JSON回答>' --json 2>/dev/null" >> "${SKILL_DIR}/deep_research.md"
        cat >> "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
```

**`NEEDS_INPUT` + `type: approve`：**
- 將 `plan_summary` 呈現給使用者
- 問使用者是否確認
- 確認：`--answer '{"approved": true}'`
- 修改：`--answer '{"approved": false, "revised_plan": "修改內容"}'`

**`DONE`：**
- 讀取 `workspace` 路徑下的 `final-report.md`，呈現給使用者

**`ERROR`：**
- 顯示錯誤訊息

### Step 3: 重複 Step 2 直到 status = DONE 或 ERROR

## 注意事項
- 一定要加 `--json` flag，否則會卡在 stdin 等待
- `--noask` 模式不會有 NEEDS_INPUT，直接跑到 DONE
- 繁體中文回應
SKILL_EOF
        echo "      ✓ Claude Code: /deep_research → ${SKILL_DIR}/deep_research.md"
        INSTALLED_SKILLS="${INSTALLED_SKILLS}\n    /deep_research <topic>                (Claude Code)"
    fi
else
    echo "      Claude Code 未安裝，跳過。"
fi

# --- Gemini CLI ---
if [[ -d "$HOME/.gemini" ]]; then
    read -rp "  偵測到 Gemini CLI，是否安裝 /deep_research skill？(y/n) " INSTALL_GEMINI
    if [[ "${INSTALL_GEMINI}" =~ ^[Yy]$ ]]; then
        GEMINI_DIR="$HOME/.gemini/commands"
        mkdir -p "${GEMINI_DIR}"
        cat > "${GEMINI_DIR}/deep_research.toml" <<GEMINI_EOF
[command]
description = "LangGraph 深度研究 workflow（支援互動澄清）"

[command.args.topic]
description = "研究主題"
required = true

[command.args.flags]
description = "選項：--quick/--standard/--deep --budget N --model claude/openai/gemini --noask"
required = false

[[steps]]
prompt = """
用終端機執行深度研究，使用 --json turn-based protocol。

Step 1: 啟動
  cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py {{topic}} {{flags}} --json 2>/dev/null

Step 2: 讀取 stdout JSON，根據 status 處理：
  - NEEDS_INPUT + type:clarify → 從 questions 陣列取出問題，問使用者，收集回答成 JSON {"0":"回答1","1":"回答2"}，用 --resume <thread_id> --answer '<JSON>' --json 繼續
  - NEEDS_INPUT + type:approve → 呈現 plan_summary，問使用者確認，用 --resume <thread_id> --answer '{"approved":true}' --json 繼續
  - DONE → 讀取 workspace 路徑下的 final-report.md，以繁體中文呈現
  - ERROR → 顯示錯誤

Step 3: 重複 Step 2 直到 DONE 或 ERROR。

注意：--noask 模式不會出現 NEEDS_INPUT。繁體中文回應。
"""
GEMINI_EOF
        echo "      ✓ Gemini CLI: /deep_research → ${GEMINI_DIR}/deep_research.toml"
        INSTALLED_SKILLS="${INSTALLED_SKILLS}\n    /deep_research <topic>                (Gemini CLI)"
    fi
else
    echo "      Gemini CLI 未安裝，跳過。"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Usage:"
echo "    ${VENV_DIR}/bin/python3 main.py <topic>"
echo "    ${VENV_DIR}/bin/python3 main.py <topic> --model openai --quick"
if [[ -n "${INSTALLED_SKILLS}" ]]; then
    echo ""
    echo "  Installed skills:"
    echo -e "${INSTALLED_SKILLS}"
fi
echo "============================================"
