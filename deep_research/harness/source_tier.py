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
    # PapersWithCode — SOTA tracking + paper↔repo linking; treated as
    # academic-adjacent curation rather than a marketing blog.
    "paperswithcode.com",
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
# Low-info zh-TW / zh-CN domains — Whisper plan P1-5
# ---------------------------------------------------------------------------
# Domains that publish high-volume, low-signal "2026 X 趨勢 / 你必須知道"
# style listicles. In the failed-workspace run 84.5% of sources came from
# exactly this class of blog. These are NOT academic, NOT first-party, and
# NOT reviewed engineering writeups — they're content-mill output.
# Classifying them as T5 (same bucket as UGC) means:
#   - They still get fetched when nothing better is available;
#   - They never outrank a genuine T1–T3 source in ``tier_rank`` ordering;
#   - ``_log_domain_bias`` will still flag over-concentration if a run is
#     dominated by one.
_LOW_INFO_ZH_DOMAINS: frozenset[str] = frozenset({
    # Large-portal news aggregators — mostly syndicated marketing copy.
    "tw.news.yahoo.com",
    "news.yahoo.com",
    "tw.stock.yahoo.com",
    # Vendor-run blogs republished as "insight articles".
    "ibm.com",
    # Content-mill republishers (reposts of Weibo / WeChat click-bait).
    "kknews.cc",
    "toutiao.com",
    "sohu.com",
    "163.com",
    "sina.com.cn",
    "cnbeta.com",
    "iask.ca",
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


def _path(url: str) -> str:
    """Extract the URL path (lowercase, without query/fragment)."""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        return (parsed.path or "").lower()
    except Exception:
        return ""


def _matches(hostname: str, domain_set: frozenset[str]) -> bool:
    """True if hostname equals or is a subdomain of any entry in domain_set."""
    for d in domain_set:
        if hostname == d or hostname.endswith("." + d):
            return True
    return False


# GitHub repo root + docs pages count as T2 (canonical README / wiki /
# architecture docs are first-party-quality descriptions of the software).
# Deep source-code URLs stay at T4 — the pipeline is not trying to consume
# raw code as a research claim. ``github.blog`` is editorial marketing and
# is left to fall through to T4.
#
# Match examples:
#   github.com/user/repo                    → T2 (repo root)
#   github.com/user/repo/                   → T2
#   github.com/user/repo/blob/main/README.md → T2 (README)
#   github.com/user/repo/wiki               → T2 (wiki)
#   github.com/user/repo/tree/main/docs     → T2 (docs tree)
#   github.com/user/repo/blob/main/src/x.py → T4 (not matched)
_GITHUB_T2_PATH_RE: re.Pattern = re.compile(
    r"^/[^/]+/[^/]+(?:"
    r"/?$"                                   # repo root (optional trailing slash)
    r"|/blob/[^/]+/readme(?:\.[a-z]+)?$"      # README.*
    r"|/tree/[^/]+/docs(?:/|$)"               # /docs/ tree
    r"|/blob/[^/]+/docs/"                     # file under docs/
    r"|/wiki(?:/|$)"                          # wiki
    r")",
    re.IGNORECASE,
)


def _is_github_t2(host: str, path: str) -> bool:
    """True for github.com URLs that carry curated, T2-grade documentation.

    Keep permissive but not sloppy:
      - github.com only (NOT github.blog / github.io – those have their own rules)
      - ``user/repo`` root (landing page is effectively the README)
      - Explicit README.* blobs
      - /wiki/... and /docs/... trees
    Everything else (raw source files, issues, pulls, actions, …) stays T4.
    """
    if host != "github.com":
        return False
    return bool(_GITHUB_T2_PATH_RE.match(path))


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
    path = _path(url)

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
    # T2 — *.github.io project pages (canonical docs published via GH Pages).
    if host == "github.io" or host.endswith(".github.io"):
        return "T2"
    # T2 — github.com repo root / README / wiki / /docs (not raw source).
    if _is_github_t2(host, path):
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

    # T5 — Low-info zh-TW / zh-CN content mills (Whisper P1-5).
    # Classified alongside UGC so they stay fetchable when nothing better is
    # available, but never outrank T1–T3 in tier_rank ordering.
    if _matches(host, _LOW_INFO_ZH_DOMAINS):
        return "T5"

    # T4 — Default
    return "T4"


def tier_rank(tier: str) -> int:
    """Return sort key: lower = higher quality (T1=0, T6=5)."""
    return {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5}.get(tier, 3)
