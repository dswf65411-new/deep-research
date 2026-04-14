"""Arxiv ID and URL existence validation (P0-4).

Defense against LLM hallucination. In the 2026-04-14 failure workspace,
phase0 produced arxiv IDs ``2604.05550``, ``2508.00031``, ``2504.01222``,
``2601.09929`` — all of which either point at future months or simply
don't exist. They propagated through to the final report as legitimate
citations.

Two validators:

- :func:`validate_arxiv_id` queries the official arxiv Atom API
  (``export.arxiv.org``) to confirm the paper exists. Results are cached
  in ``<workspaces_root>/_cache/arxiv_validation.json`` so rerunning
  research on the same paper doesn't re-hit the API.
- :func:`validate_url` does a 3-second HTTP HEAD. 404/410 → the URL is
  almost certainly hallucinated; network/timeout → ``UNREACHABLE`` so the
  caller can distinguish "known bad" from "couldn't check".

:func:`validate_plan_text` runs both validators concurrently over an
arbitrary markdown blob (phase0's plan, phase1a's seed list, etc.) and
returns a structured verdict. :func:`annotate_invalid` rewrites the
original text in-place, tagging failures with ``[REMOVED: ...]`` so
downstream phases can't accidentally treat a known-bad URL as a seed.

Network behaviour is injectable via the ``client`` parameter so tests
can run fully offline. See ``tests/test_url_validator.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)


UrlStatus = Literal["OK", "LIKELY_HALLUCINATED", "UNREACHABLE"]


# ---------------------------------------------------------------------------
# Regex extractors
# ---------------------------------------------------------------------------

# Arxiv IDs come in the form ``YYMM.NNNNN`` (N=4 or 5 digits) with optional
# version suffix ``vN``. We keep the version off the cache key (the paper
# itself exists regardless of version).
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

# URL extraction -- greedy but stops at common markdown/punctuation wrappers.
# Trailing ASCII + CJK punctuation (U+300B, U+300D, U+300F, U+3001, U+FF0C,
# U+3002, U+FF1B, U+FF1A) is stripped post-hoc so a URL at the end of a
# sentence in mixed-language text does not pick up the closing punctuation.
_URL_RE = re.compile("https?://[^\\s<>\"'\\)\\]\u300d\u300f}]+")

_URL_TRAILING = ".,;:!?)]}>\u3001\uff0c\u3002\uff1b\uff1a"  # runtime-stripped trailing chars


def extract_arxiv_ids(text: str) -> list[str]:
    """Return sorted, deduplicated arxiv IDs found in ``text`` (version stripped)."""
    if not text:
        return []
    return sorted({m.group(1) for m in _ARXIV_ID_RE.finditer(text)})


def extract_urls(text: str) -> list[str]:
    """Return deduplicated URLs in first-appearance order, with trailing punctuation stripped."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0)
        while url and url[-1] in _URL_TRAILING:
            url = url[:-1]
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Optional[Path]) -> dict[str, bool]:
    if cache_path is None:
        return {}
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("url_validator: failed to load cache %s, starting fresh", cache_path)
    return {}


def _save_cache(cache_path: Optional[Path], cache: dict[str, bool]) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("url_validator: failed to persist cache %s", cache_path)


# ---------------------------------------------------------------------------
# Single-item validators
# ---------------------------------------------------------------------------

async def validate_arxiv_id(
    arxiv_id: str,
    cache_path: Optional[Path] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 5.0,
) -> bool:
    """Return True iff arxiv accepts ``arxiv_id``.

    Two-stage check:

    1. The Atom API at ``export.arxiv.org`` is authoritative (empty feed =
       missing paper). This is the preferred signal.
    2. When ``export.arxiv.org`` times out or errors, fall back to a HEAD
       on ``arxiv.org/abs/<id>``. A 200 on the abs page means the paper
       exists; a 404 means it's hallucinated. This avoids false negatives
       when the API mirror is unreachable (observed 2026-04: the export
       host was returning connection timeouts from residential networks
       while the main arxiv.org was reachable).
    """
    cache = _load_cache(cache_path)
    if arxiv_id in cache:
        return cache[arxiv_id]

    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    ok: Optional[bool] = None

    async def _probe(c: httpx.AsyncClient) -> Optional[bool]:
        try:
            resp = await c.get(api_url, timeout=timeout)
            if resp.status_code == 200:
                return "<entry>" in resp.text
        except Exception as exc:
            logger.info("validate_arxiv_id(%s) export API failed, falling back to abs: %s", arxiv_id, exc)
        try:
            resp = await c.head(abs_url, timeout=timeout, follow_redirects=True)
            if resp.status_code == 200:
                return True
            if resp.status_code in (404, 410):
                return False
        except Exception as exc:
            logger.warning("validate_arxiv_id(%s) abs fallback also failed: %s", arxiv_id, exc)
        return None

    if client is None:
        async with httpx.AsyncClient() as c:
            ok = await _probe(c)
    else:
        ok = await _probe(client)

    if ok is None:
        ok = False
    cache[arxiv_id] = ok
    _save_cache(cache_path, cache)
    return ok


async def validate_url(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 3.0,
) -> UrlStatus:
    """Probe ``url`` and classify: OK / LIKELY_HALLUCINATED / UNREACHABLE.

    - 200–399 → OK
    - 404/410 → LIKELY_HALLUCINATED (caller should not treat as seed)
    - DNS failure / invalid URL → LIKELY_HALLUCINATED
    - Network timeout / connection reset → UNREACHABLE (unknown)

    HEAD is preferred; some origins block HEAD (403/405) so the caller
    should still treat non-2xx that isn't 404/410 as UNREACHABLE rather
    than hallucinated.
    """
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                resp = await c.head(url)
        else:
            resp = await client.head(url, timeout=timeout, follow_redirects=True)
        code = resp.status_code
    except httpx.InvalidURL:
        return "LIKELY_HALLUCINATED"
    except (httpx.ConnectError, httpx.RemoteProtocolError):
        # DNS lookup failure — hostname doesn't resolve, almost certainly fake.
        return "LIKELY_HALLUCINATED"
    except (httpx.TimeoutException, httpx.NetworkError):
        return "UNREACHABLE"
    except Exception as exc:
        logger.warning("validate_url(%s) unexpected error: %s", url, exc)
        return "UNREACHABLE"

    if 200 <= code < 400:
        return "OK"
    if code in (404, 410):
        return "LIKELY_HALLUCINATED"
    return "UNREACHABLE"


# ---------------------------------------------------------------------------
# Batch: validate an entire markdown blob
# ---------------------------------------------------------------------------

async def validate_plan_text(
    text: str,
    cache_path: Optional[Path] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Extract all arxiv IDs + URLs from ``text`` and validate concurrently.

    Returns::

        {
          "arxiv": {"2310.05193": True, "2604.05550": False, ...},
          "urls":  {"https://...": "OK" | "LIKELY_HALLUCINATED" | "UNREACHABLE", ...},
        }

    ``cache_path`` is honoured only for arxiv IDs (URL HEAD is cheap and
    caching it risks missing newly-dead links).
    """
    arxiv_ids = extract_arxiv_ids(text)
    urls = extract_urls(text)

    if not arxiv_ids and not urls:
        return {"arxiv": {}, "urls": {}}

    async def _run(owned_client: httpx.AsyncClient):
        arxiv_coros = [validate_arxiv_id(a, cache_path, owned_client) for a in arxiv_ids]
        url_coros = [validate_url(u, owned_client) for u in urls]
        arxiv_res, url_res = await asyncio.gather(
            asyncio.gather(*arxiv_coros, return_exceptions=True),
            asyncio.gather(*url_coros, return_exceptions=True),
        )
        arxiv_map: dict[str, bool] = {}
        for aid, r in zip(arxiv_ids, arxiv_res):
            arxiv_map[aid] = False if isinstance(r, BaseException) else bool(r)
        url_map: dict[str, UrlStatus] = {}
        for u, r in zip(urls, url_res):
            url_map[u] = "UNREACHABLE" if isinstance(r, BaseException) else r
        return {"arxiv": arxiv_map, "urls": url_map}

    if client is not None:
        return await _run(client)
    async with httpx.AsyncClient() as owned:
        return await _run(owned)


def annotate_invalid(text: str, validation: dict) -> str:
    """Rewrite ``text`` replacing invalid arxiv IDs / URLs with a ``[REMOVED: ...]`` marker.

    Downstream regex extractors (phase1a seed harvester) look for raw
    arxiv IDs / URLs — the replacement ensures they can't pick up a
    known-bad value. The original value is preserved inside the marker
    for audit/debugging.
    """
    out = text
    for aid, ok in validation.get("arxiv", {}).items():
        if not ok:
            out = re.sub(
                rf"\b{re.escape(aid)}(?:v\d+)?\b",
                f"[REMOVED: hallucinated arxiv ID {aid}]",
                out,
            )
    for url, status in validation.get("urls", {}).items():
        if status == "LIKELY_HALLUCINATED":
            out = out.replace(url, f"[REMOVED: hallucinated URL {url}]")
    return out


def invalid_items(validation: dict) -> dict[str, list[str]]:
    """Extract the lists of invalid items for logging / execution_log."""
    return {
        "hallucinated_arxiv": [a for a, ok in validation.get("arxiv", {}).items() if not ok],
        "hallucinated_urls": [
            u for u, s in validation.get("urls", {}).items() if s == "LIKELY_HALLUCINATED"
        ],
        "unreachable_urls": [
            u for u, s in validation.get("urls", {}).items() if s == "UNREACHABLE"
        ],
    }
