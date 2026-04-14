"""Workspace file operations — create directories, read/write workspace files."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from deep_research.harness.secret_scanner import redact_secrets


def create_workspace(topic: str, base_dir: str | None = None) -> str:
    """Create workspace directory structure. Returns the workspace path.

    Default base_dir is <project_root>/workspaces/
    """
    if base_dir is None:
        from deep_research.config import PROJECT_ROOT
        base_dir = str(PROJECT_ROOT / "workspaces")
    base = Path(os.path.expanduser(base_dir))
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Sanitise topic for directory name
    safe_topic = "".join(
        c if c.isalnum() or c in "-_" else "-"
        for c in topic[:40]
    ).strip("-")
    workspace = base / f"{date_str}_{safe_topic}"
    workspace.mkdir(parents=True, exist_ok=True)
    # Owner-only access — workspace may contain redacted-but-still-sensitive
    # research notes, grounding fingerprints, etc.
    try:
        os.chmod(workspace, 0o700)
    except OSError:
        pass  # e.g. Windows / restricted filesystem — best-effort

    # Sub-directories
    for sub in [
        "search-results",
        "grounding-results",
        "report-sections",
    ]:
        (workspace / sub).mkdir(exist_ok=True)

    return str(workspace)


def _safe_content(content: str) -> str:
    """Defense-in-depth redact before any workspace write.

    Input-layer redaction (main.py) catches user-supplied secrets. This
    second pass catches secrets echoed by LLMs, pulled from fetched URLs,
    or introduced by a code path that bypassed the input layer.
    """
    redacted, _ = redact_secrets(content)
    return redacted


def write_workspace_file(workspace_path: str, filename: str, content: str) -> str:
    """Write a file inside the workspace. Returns the full path.

    Secrets are redacted before the file hits disk.
    """
    path = Path(workspace_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_content(content), encoding="utf-8")
    return str(path)


def read_workspace_file(workspace_path: str, filename: str) -> str | None:
    """Read a file from the workspace. Returns None if not found."""
    path = Path(workspace_path) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def append_workspace_file(workspace_path: str, filename: str, content: str) -> str:
    """Append content to a workspace file.

    Secrets are redacted before the file hits disk.
    """
    path = Path(workspace_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_safe_content(content))
    return str(path)


def list_workspace_files(workspace_path: str, subdir: str = "", pattern: str = "*.md") -> list[str]:
    """List files matching a pattern within workspace (or subdir)."""
    base = Path(workspace_path)
    if subdir:
        base = base / subdir
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob(pattern))


def init_source_registry(workspace_path: str) -> str:
    """Initialise the source-registry.md file."""
    content = (
        "# Source Registry\n\n"
        "| source_id | url | title | fetched_title | tier | url_status "
        "| date | engines | roles | subquestion |\n"
        "|-----------|-----|-------|--------------|------|------------|"
        "------|---------|-------|-------------|\n"
    )
    return write_workspace_file(workspace_path, "source-registry.md", content)


def init_execution_log(workspace_path: str, topic: str, budget: int) -> str:
    """Initialise the execution-log.md file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"# Execution Log\n"
        f"**Research Topic:** {topic}\n"
        f"**Start time:** {ts}\n"
        f"**Search budget:** 0 / {budget}\n\n"
        f"## Queries already searched\n\n"
        f"## Round 1\n"
    )
    return write_workspace_file(workspace_path, "execution-log.md", content)


def init_gap_log(workspace_path: str) -> str:
    """Initialise the gap-log.md file."""
    content = (
        "# Gap Log\n\n"
        "## Missing perspectives\n"
        "(positions discovered during Phase 1 search but not yet covered)\n\n"
        "## Weak evidence\n"
        "(claims backed by a single source only)\n\n"
        "## Unresolved contradictions\n"
        "(both advocate and critic sides have approved claims but opposite conclusions)\n"
    )
    return write_workspace_file(workspace_path, "gap-log.md", content)
