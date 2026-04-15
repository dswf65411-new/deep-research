"""Whisper X-1 — cross-source mirror / duplicate detector.

Motivated by failure workspace issue #106: "5 個 source 可能是同源轉載" —
pipeline treated mirrored copies of the same paper as independent
verification. This module identifies groups of sources that are almost
certainly the same upstream artifact so phase2/phase3 can:

1. Skip treating mirrors as cross-source confirmation.
2. Drop a warning into gap-log listing the groups.

All detection is deterministic and offline — we never re-fetch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse


# arxiv IDs — matches both classic (``1607.01234``) and modern (``2310.05193``)
_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|html|pdf|ps)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_ARXIV_ID_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")
_DOI_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with",
    "to", "from", "by", "at", "as", "is", "are", "be", "it", "this",
    "that", "these", "those", "we", "our", "their", "into",
})


@dataclass
class MirrorGroup:
    """A cluster of sources believed to share an upstream artifact."""

    key: str  # the evidence that merged them (e.g. "arxiv:2310.05193")
    sources: list = field(default_factory=list)  # Source models or dicts


def normalise_title(title: str) -> str:
    """Lowercase, drop punctuation/stopwords, collapse whitespace."""
    if not title:
        return ""
    text = _PUNCT_RE.sub(" ", title.lower())
    tokens = [t for t in _WHITESPACE_RE.split(text) if t and t not in _STOPWORDS]
    return " ".join(tokens)


def title_similarity(a: str, b: str) -> float:
    """Jaccard over normalised token sets; returns 0.0 if either is empty."""
    ta = set(normalise_title(a).split())
    tb = set(normalise_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _extract_arxiv_id(url: str, title: str) -> str | None:
    m = _ARXIV_ID_RE.search(url or "")
    if m:
        return m.group(1)
    # Sometimes the title carries the ID explicitly; only trust that when
    # the URL is also arxiv-flavoured to avoid false positives from blog
    # posts that quote an ID.
    if "arxiv.org" in (url or "").lower():
        m2 = _ARXIV_ID_BARE_RE.search(title or "")
        if m2:
            return m2.group(1)
    return None


def _extract_doi(url: str) -> str | None:
    m = _DOI_RE.search(url or "")
    return m.group(1).lower() if m else None


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key, "")
    return getattr(obj, key, "") or ""


def detect_mirror_groups(
    sources: Iterable,
    *,
    title_threshold: float = 0.85,
) -> list[MirrorGroup]:
    """Cluster sources believed to be mirrors of the same upstream artifact.

    Rules, in priority order:
    1. Same arxiv ID (different URLs: ``/abs``, ``/html``, ``/pdf``).
    2. Same DOI.
    3. Normalised title Jaccard ≥ ``title_threshold`` AND same host OR
       both arxiv-ish. (Plain title match across arbitrary domains is too
       aggressive — a shared phrase like "LangGraph Supervisor" would
       false-positive.)

    Returns groups with ≥ 2 sources. Singleton sources don't show up.
    """
    items = list(sources or [])
    groups: dict[str, MirrorGroup] = {}
    ungrouped: list = []

    # --- Pass 1: arxiv + DOI IDs ---
    for s in items:
        url = _get(s, "url")
        title = _get(s, "title") or _get(s, "fetched_title")
        arxiv = _extract_arxiv_id(url, title)
        if arxiv:
            key = f"arxiv:{arxiv}"
            groups.setdefault(key, MirrorGroup(key=key)).sources.append(s)
            continue
        doi = _extract_doi(url)
        if doi:
            key = f"doi:{doi}"
            groups.setdefault(key, MirrorGroup(key=key)).sources.append(s)
            continue
        ungrouped.append(s)

    # --- Pass 2: title similarity within remaining items ---
    # Quadratic scan — safe up to a few hundred sources, which is the
    # deep-research regime. If we ever exceed that, switch to MinHash.
    used = [False] * len(ungrouped)
    for i, s in enumerate(ungrouped):
        if used[i]:
            continue
        ti = _get(s, "title")
        ui = _get(s, "url")
        if not ti:
            continue
        cluster = [s]
        for j in range(i + 1, len(ungrouped)):
            if used[j]:
                continue
            tj = _get(ungrouped[j], "title")
            uj = _get(ungrouped[j], "url")
            sim = title_similarity(ti, tj)
            if sim < title_threshold:
                continue
            if _same_domain(ui, uj) or (_is_arxiv_ish(ui) and _is_arxiv_ish(uj)):
                cluster.append(ungrouped[j])
                used[j] = True
        if len(cluster) > 1:
            used[i] = True
            key = f"title:{normalise_title(ti)[:60]}"
            groups[key] = MirrorGroup(key=key, sources=cluster)

    # Only return multi-member groups — singletons aren't mirrors.
    return [g for g in groups.values() if len(g.sources) > 1]


def _same_domain(u1: str, u2: str) -> bool:
    try:
        return urlparse(u1).netloc.lower().lstrip("www.") == urlparse(u2).netloc.lower().lstrip("www.")
    except Exception:
        return False


def _is_arxiv_ish(url: str) -> bool:
    return "arxiv.org" in (url or "").lower()


def format_mirror_warnings(groups: list[MirrorGroup]) -> str:
    """Render a gap-log section listing each mirror cluster."""
    if not groups:
        return ""

    lines: list[str] = []
    lines.append("## 同源轉載偵測 (mirrored / duplicate sources)")
    lines.append("")
    lines.append(
        "The following source clusters appear to be mirrors of the same "
        "upstream artifact. Downstream claim grounding should NOT treat them "
        "as independent verification."
    )
    lines.append("")
    for g in groups:
        lines.append(f"### {g.key}")
        for s in g.sources:
            sid = _get(s, "source_id") or "?"
            url = _get(s, "url") or "?"
            title = (_get(s, "title") or "")[:80]
            lines.append(f"- {sid}: {title}  <{url}>")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
