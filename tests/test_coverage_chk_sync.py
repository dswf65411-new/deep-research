"""Tests for Whisper plan P2-4 — coverage.chk sync after phase2 section writes.

The failed-workspace audit (2026-04-14) showed coverage.chk staying at "10/10
not_started" even after phase2 wrote sections. Phase3's final-report then
printed a coverage table contradicting the actual on-disk state.

The fix: after phase2 finishes, for every section file on disk, flip the
matching ``- [ ] advocate — not_started`` / ``- [ ] critic — ...`` rows in
coverage.chk to ``- [x] advocate — done``.
"""

from __future__ import annotations

from pathlib import Path

from deep_research.nodes.phase2 import (
    _extract_sq_id_from_section_filename,
    _sync_coverage_checklist,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def test_extract_sq_id_from_section_filename():
    assert _extract_sq_id_from_section_filename("q1_section.md") == "Q1"
    assert _extract_sq_id_from_section_filename("q12_section.md") == "Q12"
    # Accept Q uppercase too
    assert _extract_sq_id_from_section_filename("Q1_section.md") == "Q1"
    # Accept hyphen separator as defense
    assert _extract_sq_id_from_section_filename("q3-section.md") == "Q3"


def test_extract_sq_id_rejects_non_matching_names():
    assert _extract_sq_id_from_section_filename("summary.md") is None
    assert _extract_sq_id_from_section_filename("_misc.md") is None
    assert _extract_sq_id_from_section_filename("notq1.md") is None


# ---------------------------------------------------------------------------
# Sync behaviour
# ---------------------------------------------------------------------------


def _write_coverage_chk(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _baseline_chk() -> str:
    return (
        "# Coverage Checklist\n"
        "\n"
        "## Q1: background of langgraph\n"
        "- [ ] advocate — not_started\n"
        "- [ ] critic — not_started\n"
        "\n"
        "## Q2: tool comparison\n"
        "- [ ] advocate — not_started\n"
        "- [ ] critic — not_started\n"
        "\n"
        "## Q3: failure-handling recommendation\n"
        "- [ ] advocate — not_started\n"
        "- [ ] critic — not_started\n"
    )


def test_sync_marks_only_sqs_with_sections(tmp_path: Path):
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, _baseline_chk())

    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "q1_section.md").write_text("body", encoding="utf-8")
    (sections_dir / "q3_section.md").write_text("body", encoding="utf-8")

    marked = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    assert marked == {"Q1", "Q3"}

    new_text = coverage.read_text(encoding="utf-8")
    # Q1 and Q3 rows flipped to done
    assert "[x] advocate — done" in new_text
    assert new_text.count("[x] advocate — done") == 2
    # Q2 stays not_started
    assert "## Q2: tool comparison\n- [ ] advocate — not_started" in new_text


def test_sync_is_idempotent(tmp_path: Path):
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, _baseline_chk())
    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "q1_section.md").write_text("body", encoding="utf-8")

    # Run twice — second run should leave the file alone (no new marks).
    first = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    first_text = coverage.read_text(encoding="utf-8")
    second = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    second_text = coverage.read_text(encoding="utf-8")

    assert first == {"Q1"}
    # Second call sees Q1 already "done" — no ``[ ]`` rows to flip, so marked is empty
    assert second == set()
    assert first_text == second_text


def test_sync_missing_coverage_file_is_noop(tmp_path: Path):
    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "q1_section.md").write_text("x", encoding="utf-8")

    marked = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    assert marked == set()


def test_sync_empty_sections_list_is_noop(tmp_path: Path):
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, _baseline_chk())

    marked = _sync_coverage_checklist(str(tmp_path), [])
    assert marked == set()
    # File untouched
    assert coverage.read_text(encoding="utf-8") == _baseline_chk()


def test_sync_handles_non_matching_filenames(tmp_path: Path):
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, _baseline_chk())
    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "summary.md").write_text("x", encoding="utf-8")

    marked = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    assert marked == set()


def test_sync_preserves_existing_done_rows(tmp_path: Path):
    """If coverage.chk already has some done rows, don't clobber them."""
    body = (
        "# Coverage Checklist\n"
        "\n"
        "## Q1: background\n"
        "- [x] advocate — done\n"
        "- [x] critic — done\n"
        "\n"
        "## Q2: comparison\n"
        "- [ ] advocate — not_started\n"
        "- [ ] critic — not_started\n"
    )
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, body)
    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "q2_section.md").write_text("x", encoding="utf-8")

    marked = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    assert marked == {"Q2"}

    out = coverage.read_text(encoding="utf-8")
    # Q1 untouched
    assert "## Q1: background\n- [x] advocate — done\n- [x] critic — done" in out
    # Q2 now done
    assert "## Q2: comparison\n- [x] advocate — done\n- [x] critic — done" in out


def test_sync_accepts_hyphen_separator(tmp_path: Path):
    """Some plans use '-' instead of em-dash between role and state."""
    body = (
        "# Coverage Checklist\n\n"
        "## Q1: x\n"
        "- [ ] advocate - not_started\n"
        "- [ ] critic - not_started\n"
    )
    coverage = tmp_path / "coverage.chk"
    _write_coverage_chk(coverage, body)
    sections_dir = tmp_path / "report-sections"
    sections_dir.mkdir()
    (sections_dir / "q1_section.md").write_text("x", encoding="utf-8")

    marked = _sync_coverage_checklist(
        str(tmp_path), list(sections_dir.glob("*.md"))
    )
    assert marked == {"Q1"}
    assert "[x] advocate — done" in coverage.read_text(encoding="utf-8")
