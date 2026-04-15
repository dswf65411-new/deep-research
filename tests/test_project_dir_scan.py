"""Tests for `scan_project_dir` (Whisper plan P0-5).

`scan_project_dir` feeds a user's local project directory into the phase0
research brief so that "design X for project Y" research actually knows Y.
It piggybacks on the existing ``refs`` contract (same dict shape as
``read_reference_files``), but behaves as *best-effort* rather than
fail-fast: binary/unreadable files are silently skipped so a single bad
file can't tank the scan.

Covers:
- Skip-dirs (``.git``, ``node_modules``, ``__pycache__``, ``.venv`` …) are
  never descended into.
- Only allow-listed extensions are read; unknown binary extensions are
  dropped.
- Sort priority: ``README.md`` / ``CLAUDE.md`` (priority 0) outrank other
  markdown, which outrank manifests, which outrank source — ties broken
  by depth then filename.
- Per-file cap: files larger than ``_PROJECT_SCAN_PER_FILE_CHARS`` get
  truncated with a trailing marker so the LLM sees it's cut.
- File-count cap: no more than ``_PROJECT_SCAN_MAX_FILES`` entries in the
  return value.
- Total-chars cap: the sum of content lengths stays under
  ``_PROJECT_SCAN_TOTAL_CHARS``.
- Non-UTF-8 / binary files are skipped silently (best-effort).
- Non-directory path → ``ValueError``.
- Returned ref dicts match the ``read_reference_files`` contract so
  ``synthesize_research_topic`` consumes them unchanged: ``type="text"``,
  ``name`` prefixed with ``project-dir/<relpath>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.context import (
    _PROJECT_SCAN_MAX_FILES,
    _PROJECT_SCAN_PER_FILE_CHARS,
    _PROJECT_SCAN_TOTAL_CHARS,
    scan_project_dir,
)


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Baseline contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_non_directory_raises(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            scan_project_dir(str(f))

    def test_empty_dir_returns_empty(self, tmp_path):
        assert scan_project_dir(str(tmp_path)) == []

    def test_ref_shape_matches_read_reference_files(self, tmp_path):
        _write(tmp_path, "README.md", "# hi")
        refs = scan_project_dir(str(tmp_path))
        assert len(refs) == 1
        r = refs[0]
        assert set(r.keys()) == {"type", "name", "content"}
        assert r["type"] == "text"
        assert r["name"].startswith("project-dir/")
        assert r["content"] == "# hi"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_skip_dirs_not_descended(self, tmp_path):
        _write(tmp_path, "README.md", "root")
        _write(tmp_path, ".git/config", "secret")
        _write(tmp_path, "node_modules/foo/index.js", "junk")
        _write(tmp_path, "__pycache__/bar.py", "pyc")
        _write(tmp_path, ".venv/lib/site-packages/pkg/__init__.py", "vendor")
        refs = scan_project_dir(str(tmp_path))
        names = [r["name"] for r in refs]
        assert names == ["project-dir/README.md"]

    def test_unknown_extension_dropped(self, tmp_path):
        _write(tmp_path, "keep.md", "md")
        _write(tmp_path, "drop.bin", "bin")
        _write(tmp_path, "drop.pyc", "bytecode")
        names = [r["name"] for r in scan_project_dir(str(tmp_path))]
        assert names == ["project-dir/keep.md"]

    def test_binary_file_with_allowed_ext_skipped(self, tmp_path):
        # .md extension but content is non-UTF-8 → silent skip (best-effort).
        _write(tmp_path, "good.md", "good")
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"\xff\xfe\x00\x01 binary garbage")
        refs = scan_project_dir(str(tmp_path))
        names = [r["name"] for r in refs]
        assert names == ["project-dir/good.md"]


# ---------------------------------------------------------------------------
# Sort priority
# ---------------------------------------------------------------------------


class TestSortPriority:
    def test_readme_before_manifest_before_source(self, tmp_path):
        _write(tmp_path, "src/main.py", "print(1)")
        _write(tmp_path, "pyproject.toml", "[tool]")
        _write(tmp_path, "README.md", "hi")
        refs = scan_project_dir(str(tmp_path))
        names = [r["name"] for r in refs]
        assert names == [
            "project-dir/README.md",
            "project-dir/pyproject.toml",
            "project-dir/src/main.py",
        ]

    def test_claude_md_ranks_with_readme(self, tmp_path):
        _write(tmp_path, "notes.md", "generic")
        _write(tmp_path, "CLAUDE.md", "claude")
        refs = scan_project_dir(str(tmp_path))
        names = [r["name"] for r in refs]
        # CLAUDE.md is priority 0, notes.md is priority 1 → CLAUDE.md wins.
        assert names[0] == "project-dir/CLAUDE.md"

    def test_shallower_wins_on_tie(self, tmp_path):
        _write(tmp_path, "deep/nested/notes.md", "deep")
        _write(tmp_path, "notes.md", "shallow")
        refs = scan_project_dir(str(tmp_path))
        names = [r["name"] for r in refs]
        assert names[0] == "project-dir/notes.md"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class TestBudgets:
    def test_per_file_truncation(self, tmp_path):
        huge = "x" * (_PROJECT_SCAN_PER_FILE_CHARS + 5000)
        _write(tmp_path, "big.md", huge)
        refs = scan_project_dir(str(tmp_path))
        assert len(refs) == 1
        content = refs[0]["content"]
        assert len(content) <= _PROJECT_SCAN_PER_FILE_CHARS + 50  # marker padding
        assert "[truncated]" in content

    def test_file_count_cap(self, tmp_path):
        # Write more than the cap; verify the cap holds.
        for i in range(_PROJECT_SCAN_MAX_FILES + 10):
            _write(tmp_path, f"docs/f{i:03d}.md", "x")
        refs = scan_project_dir(str(tmp_path))
        assert len(refs) == _PROJECT_SCAN_MAX_FILES

    def test_total_chars_cap(self, tmp_path):
        # Each file is the per-file cap; count chosen so the total budget
        # is blown before the file-count cap.
        per_file = _PROJECT_SCAN_PER_FILE_CHARS
        needed = (_PROJECT_SCAN_TOTAL_CHARS // per_file) + 3
        for i in range(needed):
            _write(tmp_path, f"docs/f{i:03d}.md", "y" * per_file)
        refs = scan_project_dir(str(tmp_path))
        total = sum(len(r["content"]) for r in refs)
        assert total <= _PROJECT_SCAN_TOTAL_CHARS + 100  # allow marker padding
