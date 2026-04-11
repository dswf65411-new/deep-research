"""Workspace file operations — create directories, read/write workspace files."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


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

    # Sub-directories
    for sub in [
        "search-results",
        "grounding-results",
        "report-sections",
    ]:
        (workspace / sub).mkdir(exist_ok=True)

    return str(workspace)


def write_workspace_file(workspace_path: str, filename: str, content: str) -> str:
    """Write a file inside the workspace. Returns the full path."""
    path = Path(workspace_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def read_workspace_file(workspace_path: str, filename: str) -> str | None:
    """Read a file from the workspace. Returns None if not found."""
    path = Path(workspace_path) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def append_workspace_file(workspace_path: str, filename: str, content: str) -> str:
    """Append content to a workspace file."""
    path = Path(workspace_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
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
        f"# 執行日誌\n"
        f"**研究主題：** {topic}\n"
        f"**開始時間：** {ts}\n"
        f"**搜尋預算：** 0 / {budget}\n\n"
        f"## 已搜 Query 清單\n\n"
        f"## 第 1 輪\n"
    )
    return write_workspace_file(workspace_path, "execution-log.md", content)


def init_gap_log(workspace_path: str) -> str:
    """Initialise the gap-log.md file."""
    content = (
        "# Gap Log\n\n"
        "## 缺失視角\n"
        "（Phase 1 搜尋過程中發現但尚未覆蓋的立場）\n\n"
        "## 薄弱證據\n"
        "（只有單一來源的 claim）\n\n"
        "## 未解矛盾\n"
        "（正反方都有 approved claim 但結論相反）\n"
    )
    return write_workspace_file(workspace_path, "gap-log.md", content)
