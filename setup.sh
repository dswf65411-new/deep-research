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
    echo "[1/5] pyenv already installed [OK]"
    eval "$(pyenv init -)"
fi

# ------------------------------------------------------------------
# 2. Install Python
# ------------------------------------------------------------------
if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "[2/5] Python ${PYTHON_VERSION} already installed [OK]"
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
    echo "[3/5] venv already exists [OK]"
else
    echo "[3/5] Creating venv..."
    "${PYENV_PYTHON}" -m venv "${VENV_DIR}"
fi

echo "      Installing dependencies..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${PROJECT_DIR}/requirements.txt"
echo "      Dependencies installed [OK]"

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
    echo "  +-------------------------------------------------------+"
    echo "  |  The following API keys are needed (press Enter to    |"
    echo "  |  skip optional items).                                |"
    echo "  +-------------------------------------------------------+"
    echo ""
    echo "  -- LLM (at least one required; priority: Claude > Gemini > OpenAI) --"
    echo "     Where to get keys:"
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
        echo "  [ERROR] At least one LLM API key is required."
        echo "     Re-run: make init"
        exit 1
    fi

    echo ""
    echo "  -- Search engines (required) --"
    echo "     Brave:   https://brave.com/search/api/"
    echo "     Serper:  https://serper.dev/"
    echo ""

    read -rp "  BRAVE_API_KEY:   " BRAVE_KEY
    read -rp "  SERPER_API_KEY:  " SERPER_KEY

    if [[ -z "${BRAVE_KEY}" ]]; then
        echo ""
        echo "  [ERROR] BRAVE_API_KEY is required."
        echo "     Sign up at: https://brave.com/search/api/"
        echo "     Re-run: make init"
        exit 1
    fi

    if [[ -z "${SERPER_KEY}" ]]; then
        echo ""
        echo "  [ERROR] SERPER_API_KEY is required."
        echo "     Sign up at: https://serper.dev/"
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
        echo "      [OK] LLM: Claude (preferred)"
    elif [[ -n "${GEMINI_KEY}" ]]; then
        echo "      [OK] LLM: Gemini (auto-detected)"
    else
        echo "      [OK] LLM: OpenAI (auto-detected)"
    fi
    echo "      .env created [OK]"
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
    read -rp "  Claude Code detected. Install /deep_research skill? (y/n) " INSTALL_CLAUDE
    if [[ "${INSTALL_CLAUDE}" =~ ^[Yy]$ ]]; then
        SKILL_DIR="$HOME/.claude/commands"
        mkdir -p "${SKILL_DIR}"
        cat > "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
---
description: "LangGraph deep research. Trigger on: '/deep_research'."
argument-hint: "<research topic> [--quick | --standard | --deep] [--budget N] [--model claude|openai|gemini] [--noask]"
---

# /deep_research

Run deep research through the deep-research LangGraph pipeline.

## Execution flow

Run with the Bash tool, passing `--json` to enable the turn-based protocol.

### Step 1: Start the research

```bash
SKILL_EOF
        # Insert dynamic paths
        echo "cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py \$ARGUMENTS --json 2>/dev/null" >> "${SKILL_DIR}/deep_research.md"
        cat >> "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
```

Read the JSON output from stdout.

### Step 2: Interactive loop

Handle each turn based on the JSON `status` field:

**`NEEDS_INPUT` + `type: clarify`:**
- Take every item from the `questions` array.
- Ask the user each question through chat, collect each answer.
- Assemble a JSON answer: `{"0": "first answer", "1": "second answer"}`
- Continue:
```bash
SKILL_EOF
        echo "cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py --resume <thread_id> --answer '<JSON answers>' --json 2>/dev/null" >> "${SKILL_DIR}/deep_research.md"
        cat >> "${SKILL_DIR}/deep_research.md" <<'SKILL_EOF'
```

**`NEEDS_INPUT` + `type: approve`:**
- Present `plan_summary` to the user.
- Ask the user to confirm.
- Approve: `--answer '{"approved": true}'`
- Revise: `--answer '{"approved": false, "revised_plan": "revised text"}'`

**`DONE`:**
- Read `final-report.md` from the workspace path and present it to the user.

**`ERROR`:**
- Show the error message.

### Step 3: Repeat Step 2 until status = DONE or ERROR.

## Notes
- Always pass `--json`, otherwise the process will block on stdin.
- `--noask` mode never emits NEEDS_INPUT and runs straight through to DONE.
- Reply in English.
SKILL_EOF
        echo "      [OK] Claude Code: /deep_research -> ${SKILL_DIR}/deep_research.md"
        INSTALLED_SKILLS="${INSTALLED_SKILLS}\n    /deep_research <topic>                (Claude Code)"
    fi
else
    echo "      Claude Code not installed; skipping."
fi

# --- Gemini CLI ---
if [[ -d "$HOME/.gemini" ]]; then
    read -rp "  Gemini CLI detected. Install /deep_research skill? (y/n) " INSTALL_GEMINI
    if [[ "${INSTALL_GEMINI}" =~ ^[Yy]$ ]]; then
        GEMINI_DIR="$HOME/.gemini/commands"
        mkdir -p "${GEMINI_DIR}"
        cat > "${GEMINI_DIR}/deep_research.toml" <<GEMINI_EOF
[command]
description = "LangGraph deep-research workflow (supports interactive clarification)"

[command.args.topic]
description = "Research topic"
required = true

[command.args.flags]
description = "Options: --quick/--standard/--deep --budget N --model claude/openai/gemini --noask"
required = false

[[steps]]
prompt = """
Run the deep-research pipeline in the terminal using the --json turn-based protocol.

Step 1: Start
  cd ${PROJECT_DIR} && ${VENV_PYTHON} main.py {{topic}} {{flags}} --json 2>/dev/null

Step 2: Read the JSON on stdout and act on status:
  - NEEDS_INPUT + type:clarify -> take each item from the questions array, ask the user, collect answers as JSON {"0":"answer1","1":"answer2"}, continue with --resume <thread_id> --answer '<JSON>' --json
  - NEEDS_INPUT + type:approve -> present plan_summary, ask the user to confirm, continue with --resume <thread_id> --answer '{"approved":true}' --json
  - DONE -> read final-report.md from the workspace path and present it in English
  - ERROR -> show the error

Step 3: Repeat Step 2 until DONE or ERROR.

Notes: --noask mode never emits NEEDS_INPUT. Reply in English.
"""
GEMINI_EOF
        echo "      [OK] Gemini CLI: /deep_research -> ${GEMINI_DIR}/deep_research.toml"
        INSTALLED_SKILLS="${INSTALLED_SKILLS}\n    /deep_research <topic>                (Gemini CLI)"
    fi
else
    echo "      Gemini CLI not installed; skipping."
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
