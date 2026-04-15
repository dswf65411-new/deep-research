"""Direct arxiv.org API retriever.

Why a dedicated retriever alongside `serper_scholar`:
- No API key, no rate budget burn, no third-party dependency for the single
  most important academic source in this pipeline.
- Returns canonical arxiv_id, published date, and categories — data that
  `serper_scholar` does not surface, but which downstream hallucination-check
  (validate_arxiv_id) wants.
- Zero risk of the LLM-indexed "paper does not exist" problem: if arxiv
  returns no `<entry>`, the paper genuinely is not on arxiv.

Called with engine name `"arxiv"` from `_run_single_search` in phase1a.
Output schema mirrors `serper_scholar` so the downstream hit-merge path
(`_select_urls_by_quota`, cross-engine bonus in `_merge_hits`) treats it
identically without special-casing.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_TIMEOUT = 20
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _clean(text: str) -> str:
    """Normalize whitespace in arxiv fields (abstracts are line-wrapped)."""
    return re.sub(r"\s+", " ", text or "").strip()


async def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    """Query the public arxiv API for papers matching `query`.

    Returns list of {"title", "url", "description", "year", "arxiv_id",
    "categories"} — schema-compatible with `serper_scholar` so downstream
    merging code treats arxiv hits identically.

    On network / parse failure returns empty list (matching the policy of
    every other engine in `_run_single_search`).
    """
    params = {
        "search_query": f"all:{query}",
        "max_results": str(max(1, min(max_results, 30))),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_ARXIV_API, params=params)
            if resp.status_code != 200:
                logger.info("arxiv API non-200: %s", resp.status_code)
                return []
            body = resp.text
    except Exception as e:
        logger.info("arxiv API call failed: %s", e)
        return []

    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.info("arxiv API XML parse failed: %s", e)
        return []

    results: list[dict] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = _clean(_text(entry, "atom:title"))
        summary = _clean(_text(entry, "atom:summary"))
        published = _text(entry, "atom:published")  # e.g. "2024-02-07T16:34:12Z"
        entry_id = _text(entry, "atom:id")  # e.g. "http://arxiv.org/abs/2402.04942v1"
        if not title or not entry_id:
            continue

        arxiv_id = _extract_arxiv_id(entry_id)
        canonical_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_id
        year = published[:4] if published and len(published) >= 4 else ""
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", _ATOM_NS)
            if c.attrib.get("term")
        ]

        results.append({
            "title": title,
            "url": canonical_url,
            "description": summary[:1200],  # abstract; clipped to match Serper snippet budget
            "year": year,
            "arxiv_id": arxiv_id,
            "categories": categories,
        })

    return results


def _text(elem: ET.Element, tag: str) -> str:
    node = elem.find(tag, _ATOM_NS)
    return (node.text or "") if node is not None else ""


_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _extract_arxiv_id(entry_id: str) -> str:
    """Pull the bare `YYMM.NNNNN` id out of an arxiv atom entry id URL."""
    m = _ID_RE.search(entry_id)
    return m.group(1) if m else ""
