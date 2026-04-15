#!/usr/bin/env bash
# setup.sh — one-shot dev environment bootstrap for deep-research.
#
# Creates .venv (Python 3.12+) and installs both runtime and dev dependencies
# declared in pyproject.toml. Idempotent: safe to re-run.
#
# Usage:
#   scripts/setup.sh                   # use default Python (python3)
#   PYTHON=python3.12 scripts/setup.sh # use a specific interpreter
#
# After this finishes you can:
#   .venv/bin/python -m pytest tests/
#   .venv/bin/deep-research "research topic"

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV="$HERE/.venv"

# ── Step 1: ensure python meets >=3.12 requirement ─────────────────────────
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.12+ or set PYTHON=/path/to/python3.12" >&2
    exit 1
fi

py_version="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
py_major="${py_version%%.*}"
py_minor="${py_version#*.}"
if [[ "$py_major" -lt 3 || ( "$py_major" -eq 3 && "$py_minor" -lt 12 ) ]]; then
    echo "ERROR: Python >=3.12 required, got $py_version (from $PYTHON)" >&2
    exit 1
fi
echo "Using $PYTHON ($py_version)"

# ── Step 2: create venv if missing ─────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "Creating venv at $VENV ..."
    "$PYTHON" -m venv "$VENV"
else
    echo "Venv already exists at $VENV — reusing"
fi

# ── Step 3: upgrade pip + install project with [dev] extras ───────────────
echo "Upgrading pip..."
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null

echo "Installing project + dev dependencies (editable)..."
"$VENV/bin/python" -m pip install -e ".[dev]"

# ── Step 4: sanity check ───────────────────────────────────────────────────
echo
echo "Sanity check — running pytest on an empty selection..."
if "$VENV/bin/python" -m pytest --collect-only tests/ -q >/dev/null 2>&1; then
    test_count="$("$VENV/bin/python" -m pytest --collect-only tests/ -q 2>&1 | grep -E 'test' | tail -1 || true)"
    echo "OK — pytest can discover tests: $test_count"
else
    echo "WARNING: pytest collection failed. Run manually to debug:"
    echo "  $VENV/bin/python -m pytest --collect-only tests/"
fi

echo
echo "Done. Quick commands:"
echo "  .venv/bin/python -m pytest tests/         # full test suite"
echo "  scripts/pytest_baseline.sh save           # store failure baseline"
echo "  scripts/pytest_baseline.sh diff           # check for new regressions"
