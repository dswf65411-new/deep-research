"""Source tier classification for research quality scoring.

Tier levels (lower = higher quality):
  T1 – Official vendor / first-party documentation
  T2 – Academic / peer-reviewed
  T3 – Professional media (Taiwan priority + major English tech press)
  T4 – General .com / .net / blog (default)
  T5 – UGC / community forums (Reddit, Dcard, PTT …)
  T6 – THIN_CONTENT or UNREACHABLE (set externally, not by this module)

Usage:
    from deep_research.harness.source_tier import classify_tier
    tier = classify_tier(url, title, content)   # returns "T1" … "T6"
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

TierType = Literal["T1", "T2", "T3", "T4", "T5", "T6"]

# ---------------------------------------------------------------------------
# T1 — Official / first-party
# ---------------------------------------------------------------------------
_T1_DOMAINS: frozenset[str] = frozenset({
    "apple.com",
    "microsoft.com",
    "openai.com",
    "anthropic.com",
    "google.com",
    "google.co.jp",
    "amazon.com",
    "aws.amazon.com",
    "azure.microsoft.com",
    "cloud.google.com",
    "developers.google.com",
    "developer.apple.com",
    "developer.mozilla.org",
    "support.apple.com",
    "support.microsoft.com",
    "docs.python.org",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
    "deepmind.google",
    "deepmind.com",
    "meta.ai",
    "ai.meta.com",
    "mistral.ai",
    "groq.com",
    "nvidia.com",
})

# Hostname prefixes that signal official/docs subdomains (e.g. developer.foo.com)
_T1_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "developer.",
    "developers.",
    "support.",
    "docs.",
    "api.",
    "platform.",
    "help.",
    "learn.",
)

# ---------------------------------------------------------------------------
# T2 — Academic / peer-reviewed
# ---------------------------------------------------------------------------
_T2_DOMAINS: frozenset[str] = frozenset({
    "arxiv.org",
    "aclanthology.org",
    "semanticscholar.org",
    "doi.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "dl.acm.org",
    "acm.org",
    "ieeexplore.ieee.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "researchgate.net",
    "scholar.google.com",
    "papers.ssrn.com",
    "ssrn.com",
    "openreview.net",
    "proceedings.mlr.press",
    "nips.cc",
    "neurips.cc",
    "icml.cc",
    "iclr.cc",
})

# TLD patterns for academic: .edu, .edu.tw, .ac.uk, .ac.jp, .ac.tw …
_T2_TLD_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'\.edu(\.[a-z]{2})?$'),   # .edu OR .edu.tw / .edu.cn etc.
    re.compile(r'\.ac\.[a-z]{2}$'),        # .ac.uk / .ac.jp / .ac.tw etc.
)

# ---------------------------------------------------------------------------
# T3 — Professional media
# ---------------------------------------------------------------------------
# Taiwan-specific: already exported as _TAIWAN_DOMAIN_WHITELIST in phase1a;
# duplicated here so source_tier.py has no circular dependency.
_T3_TAIWAN_DOMAINS: frozenset[str] = frozenset({
    "ithome.com.tw",
    "ithelp.ithome.com.tw",
    "techbang.com",
    "kocpc.com.tw",
    "mobile01.com",
    "eprice.com.tw",
    "inside.com.tw",
    "bnext.com.tw",
})

_T3_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "wired.com",
    "theverge.com",
    "techcrunch.com",
    "engadget.com",
    "arstechnica.com",
    "9to5mac.com",
    "macrumors.com",
    "appleinsider.com",
    "venturebeat.com",
    "zdnet.com",
    "cnet.com",
    "tomsguide.com",
    "pcmag.com",
    "wsj.com",
    "nytimes.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "economist.com",
    "thenextweb.com",
    "androidauthority.com",
    "xda-developers.com",
})

# ---------------------------------------------------------------------------
# T5 — UGC / community forums
# ---------------------------------------------------------------------------
_T5_DOMAINS: frozenset[str] = frozenset({
    "reddit.com",
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "ptt.cc",
    "dcard.tw",
    "zhihu.com",
    "v2ex.com",
    "answers.microsoft.com",
    "discussions.apple.com",
    "community.openai.com",
    "forum.openai.com",
    "forums.developer.apple.com",
    "ask.duckduckgo.com",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hostname(url: str) -> str:
    """Extract lowercase hostname from URL."""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _matches(hostname: str, domain_set: frozenset[str]) -> bool:
    """True if hostname equals or is a subdomain of any entry in domain_set."""
    for d in domain_set:
        if hostname == d or hostname.endswith("." + d):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_tier(
    url: str,
    title: str = "",
    content: str = "",
) -> TierType:
    """Classify a source URL into T1–T6.

    Parameters
    ----------
    url:     Full URL (or bare domain).
    title:   Page title (unused in current rules, reserved for future signals).
    content: Fetched page text (unused in current rules; T6 is set externally).

    Returns
    -------
    "T1" … "T5" based on domain matching.
    "T4" is the default when no rule fires.
    "T6" (THIN/UNREACHABLE) is NOT set here — callers should set it explicitly
    when content is too short or fetch failed.
    """
    host = _hostname(url)
    if not host:
        return "T4"

    # T1 — Official
    if _matches(host, _T1_DOMAINS):
        return "T1"
    if any(host.startswith(p) for p in _T1_SUBDOMAIN_PREFIXES):
        return "T1"

    # T2 — Academic
    if _matches(host, _T2_DOMAINS):
        return "T2"
    if any(pat.search(host) for pat in _T2_TLD_PATTERNS):
        return "T2"

    # T3 — Taiwan professional media
    if _matches(host, _T3_TAIWAN_DOMAINS):
        return "T3"

    # T3 — English professional media
    if _matches(host, _T3_MEDIA_DOMAINS):
        return "T3"

    # T5 — UGC / community
    if _matches(host, _T5_DOMAINS):
        return "T5"

    # T4 — Default
    return "T4"


def tier_rank(tier: str) -> int:
    """Return sort key: lower = higher quality (T1=0, T6=5)."""
    return {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5}.get(tier, 3)
