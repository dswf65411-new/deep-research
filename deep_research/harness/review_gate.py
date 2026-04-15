"""Whisper P3-2 — mid-pipeline human review gate primitives.

Lets the user intervene after phase1a's first search round: inspect the
source-registry preview, then either continue, skip a sub-question, refocus,
or inject a URL. The goal is to catch obvious disasters (all T4 marketing
blogs, brief entity missed) early instead of burning the whole budget.

This module is intentionally UI-agnostic. It only:
* formats a short preview from ``source-registry.md``
* parses a line of user input into a structured ``ReviewCommand``

The CLI / skill layer decides when to pause, how to collect input, and how to
apply each command to the running graph state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ReviewAction = Literal["continue", "skip", "refocus", "inject_url", "noop"]


@dataclass
class ReviewCommand:
    """Result of ``parse_review_command``.

    ``action='noop'`` means the input was empty or unrecognised; the CLI should
    re-prompt instead of silently defaulting.
    """

    action: ReviewAction
    sq_id: str | None = None
    url: str | None = None
    raw: str = ""
    error: str | None = None


_SQ_RE = re.compile(r"^(q\d+)$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def parse_review_command(line: str) -> ReviewCommand:
    """Parse one line of review-gate input.

    Recognised forms:
    * ``/continue`` → proceed with the existing plan
    * ``/skip Q3`` → drop SQ Q3 from the remaining budget
    * ``/refocus Q3 <new direction text>`` → reseed queries for Q3
    * ``/inject-url https://example.com`` → push a URL into phase1a's fetch
      queue (counted as a seed, not a search)
    """
    raw = (line or "").strip()
    if not raw:
        return ReviewCommand(action="noop", raw=raw, error="empty input")

    if not raw.startswith("/"):
        return ReviewCommand(
            action="noop", raw=raw,
            error=f"commands must start with '/': got {raw!r}",
        )

    parts = raw.split(maxsplit=2)
    cmd = parts[0].lower()
    arg1 = parts[1] if len(parts) > 1 else ""
    rest = parts[2] if len(parts) > 2 else ""

    if cmd == "/continue":
        return ReviewCommand(action="continue", raw=raw)

    if cmd == "/skip":
        sq = _normalise_sq(arg1)
        if not sq:
            return ReviewCommand(
                action="noop", raw=raw,
                error="/skip requires a sub-question id like /skip Q3",
            )
        return ReviewCommand(action="skip", sq_id=sq, raw=raw)

    if cmd == "/refocus":
        sq = _normalise_sq(arg1)
        if not sq:
            return ReviewCommand(
                action="noop", raw=raw,
                error="/refocus requires a sub-question id like /refocus Q3 <new direction>",
            )
        if not rest.strip():
            return ReviewCommand(
                action="noop", raw=raw, sq_id=sq,
                error="/refocus requires a new direction after the SQ id",
            )
        # Stash the new direction in .url field? No — use a dedicated field.
        return ReviewCommand(action="refocus", sq_id=sq, url=rest.strip(), raw=raw)

    if cmd in ("/inject-url", "/inject_url"):
        if not arg1 or not _URL_RE.match(arg1):
            return ReviewCommand(
                action="noop", raw=raw,
                error="/inject-url requires a http(s) URL like /inject-url https://example.com",
            )
        return ReviewCommand(action="inject_url", url=arg1, raw=raw)

    return ReviewCommand(
        action="noop", raw=raw,
        error=f"unknown command: {cmd}",
    )


def _normalise_sq(token: str) -> str | None:
    """``q3`` / ``Q3`` / ``Q03`` → ``Q3``. Anything else → None."""
    token = (token or "").strip()
    m = _SQ_RE.match(token)
    if not m:
        return None
    digits = re.sub(r"^q0*", "", m.group(1), flags=re.IGNORECASE)
    # Safety: if the SQ id was literally "Q0" we'd strip everything; restore.
    if not digits:
        digits = "0"
    return f"Q{digits}"


# ---------------------------------------------------------------------------
# Source-registry preview
# ---------------------------------------------------------------------------


@dataclass
class SourcePreviewRow:
    sq_id: str = ""
    source_id: str = ""
    tier: str = ""
    url: str = ""
    title: str = ""


@dataclass
class SourcePreview:
    rows: list[SourcePreviewRow] = field(default_factory=list)
    total_sources: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    by_sq: dict[str, int] = field(default_factory=dict)


_REGISTRY_ROW_RE = re.compile(
    r"^\|\s*(?P<sid>S\d+)\s*\|\s*(?P<sq>Q\d+)\s*\|\s*(?P<tier>T\d)\s*\|\s*(?P<url>https?://\S+)\s*\|\s*(?P<title>.+?)\s*\|\s*$",
    re.IGNORECASE,
)


def read_source_registry_preview(workspace: str | Path, top_n: int = 20) -> SourcePreview:
    """Read ``source-registry.md`` from ``workspace`` and summarise the first
    ``top_n`` entries. Missing / malformed file → empty ``SourcePreview``.

    The registry format is a pipe-delimited markdown table:

        | S001 | Q1 | T1 | https://... | some title |
    """
    preview = SourcePreview()
    path = Path(workspace) / "source-registry.md"
    if not path.exists():
        return preview

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _REGISTRY_ROW_RE.match(line)
        if not m:
            continue
        row = SourcePreviewRow(
            sq_id=m.group("sq").upper(),
            source_id=m.group("sid").upper(),
            tier=m.group("tier").upper(),
            url=m.group("url"),
            title=m.group("title"),
        )
        preview.total_sources += 1
        preview.by_tier[row.tier] = preview.by_tier.get(row.tier, 0) + 1
        preview.by_sq[row.sq_id] = preview.by_sq.get(row.sq_id, 0) + 1
        if len(preview.rows) < top_n:
            preview.rows.append(row)

    return preview


def format_review_prompt(preview: SourcePreview) -> str:
    """Human-readable preview, suitable for printing to the terminal."""
    if preview.total_sources == 0:
        return (
            "[review-gate] source-registry.md is empty or missing — nothing to "
            "preview. Send /continue to proceed or /skip <SQ> to drop a sub-question."
        )

    lines: list[str] = []
    lines.append(
        f"[review-gate] phase1a round 1 done — {preview.total_sources} sources collected."
    )
    tier_bits = ", ".join(
        f"{t}={n}" for t, n in sorted(preview.by_tier.items())
    )
    sq_bits = ", ".join(
        f"{sq}={n}" for sq, n in sorted(preview.by_sq.items())
    )
    lines.append(f"  tiers: {tier_bits}")
    lines.append(f"  per-SQ: {sq_bits}")
    lines.append("")
    lines.append(f"Top {len(preview.rows)} sources:")
    for r in preview.rows:
        lines.append(
            f"  {r.source_id} [{r.tier}] {r.sq_id} — {r.title[:60]}"
        )
    lines.append("")
    lines.append(
        "Commands: /continue | /skip <SQ> | /refocus <SQ> <new direction> | "
        "/inject-url <url>"
    )
    return "\n".join(lines)
