"""Project paths and configuration — all paths relative to project root."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Phase instruction prompts (shipped with the project)
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Grounding CLI scripts (shipped with the project)
GROUNDING_SCRIPTS_DIR = PROJECT_ROOT / "grounding_scripts"

# Python for grounding CLI (uses project venv by default)
GROUNDING_PYTHON = os.environ.get(
    "GROUNDING_PYTHON",
    str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
)

# MiniCheck needs Python 3.11 + its own deps — setup.sh creates this venv
MINICHECK_PYTHON = os.environ.get(
    "MINICHECK_PYTHON",
    str(PROJECT_ROOT / ".minicheck-venv" / "bin" / "python3"),
)


def get_prompt(name: str) -> str:
    """Read a prompt file from the prompts directory.

    Args:
        name: filename like "phase0-clarify.md"

    Returns:
        File content as string, or empty string if not found.
    """
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
