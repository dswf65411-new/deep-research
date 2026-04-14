"""Secret / credential detection and redaction.

Scans text for common secret patterns (webhooks, API keys, tokens) and
replaces them with `[REDACTED_<TYPE>]` tokens. Runs at two layers:

  1. Input-time scan (main.py): redact user input and reference files
     before they enter the pipeline.
  2. Write-time scan (workspace.write_workspace_file): defense in depth
     against secrets that slipped through layer 1 (e.g. LLM outputs
     echoing a redacted reference).

The scanner is pattern-based — it will miss novel secret formats. Treat
it as a safety net, not an ACL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Secret:
    """One detected secret occurrence."""

    type: str         # e.g. "DISCORD_WEBHOOK", "ANTHROPIC_KEY"
    start: int        # byte offset in source text
    end: int
    raw: str          # original matched string (kept in-memory, never logged)

    @property
    def placeholder(self) -> str:
        return f"[REDACTED_{self.type}]"


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------
# Order matters: more specific patterns first so generic ones don't swallow
# them. Each entry: (type, compiled_regex).
#
# Rules for adding a pattern:
#   - Anchor on a strong prefix (sk-ant-, AKIA, ghp_) to minimise false
#     positives.
#   - Require a minimum length that matches the real credential format.
#   - Test negative cases (ordinary URLs, code identifiers).

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Discord incoming webhook — unique path, case-sensitive
    (
        "DISCORD_WEBHOOK",
        re.compile(r"https://discord(?:app)?\.com/api/webhooks/\d{10,}/[\w-]{40,}"),
    ),
    # Slack incoming webhook
    (
        "SLACK_WEBHOOK",
        re.compile(r"https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[\w]+"),
    ),
    # Anthropic API key — `sk-ant-` prefix
    (
        "ANTHROPIC_KEY",
        re.compile(r"sk-ant-[\w-]{40,}"),
    ),
    # OpenAI project key — `sk-proj-` prefix (match first so generic sk- doesn't win)
    (
        "OPENAI_PROJECT_KEY",
        re.compile(r"sk-proj-[A-Za-z0-9_-]{40,}"),
    ),
    # Generic OpenAI key — `sk-` followed by long base62-ish
    (
        "OPENAI_KEY",
        re.compile(r"sk-[A-Za-z0-9]{40,}"),
    ),
    # Google / Gemini / Firebase API key — `AIza` + 35 chars
    (
        "GOOGLE_KEY",
        re.compile(r"AIza[A-Za-z0-9_-]{35}"),
    ),
    # GitHub fine-grained PAT — `github_pat_` + base62
    (
        "GITHUB_PAT_FINE",
        re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    ),
    # GitHub classic PAT — `ghp_` + 36 chars
    (
        "GITHUB_PAT",
        re.compile(r"ghp_[A-Za-z0-9]{36,}"),
    ),
    # AWS access key ID — exactly `AKIA` + 16 uppercase/digit
    (
        "AWS_ACCESS_KEY_ID",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # AWS secret access key — named assignment (value only, strip on redact)
    (
        "AWS_SECRET",
        re.compile(
            r"aws_secret_access_key\s*=\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?",
            re.IGNORECASE,
        ),
    ),
    # Private key PEM header (indicates block; we only flag the header —
    # full-block redaction is caller's job)
    (
        "PRIVATE_KEY_PEM",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |)PRIVATE KEY-----"
            r"[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |)PRIVATE KEY-----"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_secrets(text: str) -> list[Secret]:
    """Return all detected secrets in ``text``.

    Overlapping matches are resolved by first-pattern-wins (so more specific
    patterns registered earlier beat generic ones). Returned list is sorted
    by start offset.
    """
    if not text:
        return []

    found: list[Secret] = []
    claimed_spans: list[tuple[int, int]] = []

    for secret_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if _overlaps(start, end, claimed_spans):
                continue
            claimed_spans.append((start, end))
            found.append(
                Secret(type=secret_type, start=start, end=end, raw=m.group(0))
            )

    found.sort(key=lambda s: s.start)
    return found


def redact_secrets(text: str) -> tuple[str, list[Secret]]:
    """Redact all detected secrets in ``text``.

    Returns ``(redacted_text, secrets_found)``. If no secrets are found, the
    original text is returned unchanged and ``secrets_found`` is empty.
    """
    secrets = scan_secrets(text)
    if not secrets:
        return text, []

    pieces: list[str] = []
    cursor = 0
    for s in secrets:
        pieces.append(text[cursor : s.start])
        pieces.append(s.placeholder)
        cursor = s.end
    pieces.append(text[cursor:])
    return "".join(pieces), secrets


def contains_secret(text: str) -> bool:
    """Fast check: does ``text`` contain any recognised secret?"""
    return bool(scan_secrets(text))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _overlaps(start: int, end: int, claimed: list[tuple[int, int]]) -> bool:
    for cs, ce in claimed:
        if start < ce and cs < end:
            return True
    return False
