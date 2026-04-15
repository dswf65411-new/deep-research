"""GitHub Search API retriever.

Why a dedicated retriever alongside Brave / Serper:
- The failed-workspace analysis (2026-04-14) showed Brave+Serper web search
  frequently missed canonical repos (e.g. ``AIDE``, ``MLE-Agent``) because
  they rank by backlink noise rather than GitHub's own signal. Hitting
  ``api.github.com/search/repositories`` directly lets us sort by stars or
  recent activity and get the authoritative repo card.
- Returns structured fields — star count, last push, language, description
  — that are useful when the pipeline needs to tell signal from bit-rot.
- Complements the existing ``arxiv_retriever``: ``arxiv`` handles papers,
  ``github`` handles code / tool discovery.

Called with engine name ``"github"`` from ``_run_single_search`` in
``phase1a.py``. Output schema mirrors the other engines so downstream merge
code (``_merge_hits``, ``_select_urls_by_quota``) treats these hits
identically without special-casing.

Auth:
- Unauthenticated: 10 requests / minute (per IP) — fine for typical runs.
- With a PAT in env var ``GITHUB_TODO_PAT`` or ``GITHUB_TOKEN``: 30 / minute.
  Token usage is optional; when absent we send no Authorization header.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_GITHUB_REPO_SEARCH = "https://api.github.com/search/repositories"
_TIMEOUT = 20
_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"


def _auth_headers() -> dict[str, str]:
    """Attach a PAT if one is available in env; otherwise go anonymous.

    Prefers ``GITHUB_TODO_PAT`` (user's cross-platform TODO PAT, per
    ~/.zshrc) before falling back to the conventional ``GITHUB_TOKEN``.
    """
    token = os.environ.get("GITHUB_TODO_PAT") or os.environ.get("GITHUB_TOKEN")
    headers: dict[str, str] = {
        "Accept": _ACCEPT,
        "X-GitHub-Api-Version": _API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def github_repo_search(
    query: str,
    max_results: int = 10,
    sort: str = "stars",
) -> list[dict]:
    """Search GitHub repositories matching ``query``.

    Returns a list of hits in the same shape as ``arxiv_search``:
    ``{"title", "url", "description", "year", "stars", "language"}``. On
    network / parse failure returns empty list (matches the policy of
    every other engine in ``_run_single_search``).

    Parameters
    ----------
    query : str
        Free-text query; GitHub's search syntax is passed through. Example:
        ``"AIDE machine learning agent"`` or ``"MLE-Agent in:readme"``.
    max_results : int
        Number of hits to return (clamped 1..30 to stay under GitHub's
        per-request cap without fetching more than the planner needs).
    sort : str
        ``"stars"`` (default, most useful for "find the canonical repo"),
        ``"updated"`` (last-push; useful for "what's alive today"), or
        ``"best-match"`` (GitHub's own relevance ranking).
    """
    params: dict[str, Any] = {
        "q": query,
        "per_page": max(1, min(max_results, 30)),
    }
    if sort in ("stars", "updated"):
        params["sort"] = sort
        params["order"] = "desc"
    # "best-match" → omit sort entirely; GitHub defaults to relevance.

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                _GITHUB_REPO_SEARCH, params=params, headers=_auth_headers(),
            )
            if resp.status_code != 200:
                logger.info(
                    "github search non-200: %s (%s)",
                    resp.status_code, resp.text[:200],
                )
                return []
            data = resp.json()
    except Exception as exc:
        logger.info("github search call failed: %s", exc)
        return []

    items = data.get("items") or []
    results: list[dict] = []
    for item in items[:max_results]:
        full_name = item.get("full_name") or ""
        html_url = item.get("html_url") or ""
        if not full_name or not html_url:
            continue
        description = (item.get("description") or "").strip()
        stars = item.get("stargazers_count") or 0
        language = item.get("language") or ""
        pushed_at = item.get("pushed_at") or ""
        year = pushed_at[:4] if pushed_at and len(pushed_at) >= 4 else ""
        results.append({
            "title": f"{full_name} — GitHub ({stars}★)",
            "url": html_url,
            "description": description[:1200],
            "year": year,
            "stars": stars,
            "language": language,
        })
    return results
