"""Tests for Whisper P3-2 — mid-pipeline review gate primitives."""

from __future__ import annotations

from pathlib import Path

from deep_research.harness.review_gate import (
    ReviewCommand,
    format_review_prompt,
    parse_review_command,
    read_source_registry_preview,
)


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------


def test_parse_continue():
    cmd = parse_review_command("/continue")
    assert cmd.action == "continue"
    assert cmd.error is None


def test_parse_skip_with_sq():
    cmd = parse_review_command("/skip Q3")
    assert cmd.action == "skip"
    assert cmd.sq_id == "Q3"


def test_parse_skip_lowercase_sq():
    cmd = parse_review_command("/skip q7")
    assert cmd.action == "skip"
    assert cmd.sq_id == "Q7"


def test_parse_skip_zero_padded():
    cmd = parse_review_command("/skip Q03")
    assert cmd.sq_id == "Q3"


def test_parse_skip_missing_sq_is_noop():
    cmd = parse_review_command("/skip")
    assert cmd.action == "noop"
    assert "requires" in (cmd.error or "")


def test_parse_refocus_requires_direction():
    cmd = parse_review_command("/refocus Q3")
    assert cmd.action == "noop"
    assert cmd.sq_id == "Q3"
    assert "new direction" in (cmd.error or "")


def test_parse_refocus_full():
    cmd = parse_review_command("/refocus Q3 concentrate on supervisor arbitration failure recovery")
    assert cmd.action == "refocus"
    assert cmd.sq_id == "Q3"
    assert "supervisor arbitration" in (cmd.url or "")


def test_parse_inject_url():
    cmd = parse_review_command("/inject-url https://arxiv.org/abs/2310.05193")
    assert cmd.action == "inject_url"
    assert cmd.url == "https://arxiv.org/abs/2310.05193"


def test_parse_inject_url_alias():
    cmd = parse_review_command("/inject_url https://example.com")
    assert cmd.action == "inject_url"


def test_parse_inject_url_requires_http():
    cmd = parse_review_command("/inject-url not-a-url")
    assert cmd.action == "noop"
    assert "http(s)" in (cmd.error or "")


def test_parse_empty_is_noop():
    cmd = parse_review_command("")
    assert cmd.action == "noop"


def test_parse_non_slash_is_noop():
    cmd = parse_review_command("continue")
    assert cmd.action == "noop"
    assert "must start with '/'" in (cmd.error or "")


def test_parse_unknown_command_is_noop():
    cmd = parse_review_command("/nuclear-reset")
    assert cmd.action == "noop"
    assert "unknown command" in (cmd.error or "")


# ---------------------------------------------------------------------------
# Source-registry preview
# ---------------------------------------------------------------------------


def _write_registry(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_preview_missing_file_is_empty(tmp_path: Path):
    preview = read_source_registry_preview(tmp_path)
    assert preview.total_sources == 0
    assert preview.rows == []


def test_preview_parses_rows(tmp_path: Path):
    _write_registry(
        tmp_path / "source-registry.md",
        "# Source Registry\n"
        "| S001 | Q1 | T1 | https://arxiv.org/abs/2310.05193 | ResearchAgent paper |\n"
        "| S002 | Q1 | T4 | https://someblog.com/post1 | Marketing blog |\n"
        "| S003 | Q2 | T2 | https://github.com/foo/bar | GitHub repo |\n",
    )
    preview = read_source_registry_preview(tmp_path, top_n=10)
    assert preview.total_sources == 3
    assert preview.by_tier == {"T1": 1, "T2": 1, "T4": 1}
    assert preview.by_sq == {"Q1": 2, "Q2": 1}
    assert preview.rows[0].source_id == "S001"
    assert preview.rows[0].sq_id == "Q1"
    assert preview.rows[0].tier == "T1"


def test_preview_respects_top_n(tmp_path: Path):
    rows = [
        f"| S{i:03d} | Q1 | T1 | https://example.com/{i} | title {i} |"
        for i in range(1, 26)
    ]
    _write_registry(tmp_path / "source-registry.md", "\n".join(rows))

    preview = read_source_registry_preview(tmp_path, top_n=5)
    assert preview.total_sources == 25  # full count
    assert len(preview.rows) == 5  # but only top 5 detailed


def test_preview_ignores_malformed_lines(tmp_path: Path):
    _write_registry(
        tmp_path / "source-registry.md",
        "| S001 | Q1 | T1 | https://example.com/a | ok |\n"
        "| garbage line |\n"
        "random text not a table\n"
        "| S002 | Q1 | T4 | https://example.com/b | ok2 |\n",
    )
    preview = read_source_registry_preview(tmp_path)
    assert preview.total_sources == 2


def test_format_review_prompt_has_commands_help(tmp_path: Path):
    _write_registry(
        tmp_path / "source-registry.md",
        "| S001 | Q1 | T1 | https://x.com/a | hello |\n",
    )
    preview = read_source_registry_preview(tmp_path)
    rendered = format_review_prompt(preview)
    assert "/continue" in rendered
    assert "/skip" in rendered
    assert "/refocus" in rendered
    assert "/inject-url" in rendered
    assert "S001" in rendered


def test_format_review_prompt_empty():
    rendered = format_review_prompt(read_source_registry_preview("/nonexistent/path"))
    assert "empty" in rendered or "missing" in rendered
    # Still shows how to proceed.
    assert "/continue" in rendered
