"""Direct HTTP API calls to Brave Search, Serper (Google), and web fetch."""

from __future__ import annotations

import json
import os
from urllib.parse import quote_plus

import httpx

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

_CLIENT_TIMEOUT = 30


async def brave_search(
    query: str,
    count: int = 10,
    country: str = "",
    search_lang: str = "",
) -> list[dict]:
    """Call Brave Web Search API directly.

    Returns list of {"title", "url", "description"}.
    country: ISO 3166-1 alpha-2 country code, e.g. "TW" for Taiwan.
    search_lang: BCP 47 language code, e.g. "zh-hant" for Traditional Chinese.
    """
    if not BRAVE_API_KEY:
        return []

    url = "https://api.search.brave.com/res/v1/web/search"
    params: dict[str, str | int] = {"q": query, "count": count}
    if country:
        params["country"] = country
    if search_lang:
        params["search_lang"] = search_lang
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            return []
        data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        })
    return results


async def serper_search(
    query: str,
    gl: str = "us",
    hl: str = "en",
    num: int = 10,
) -> list[dict]:
    """Call Serper Google Search API directly.

    Returns list of {"title", "url", "description"}.
    """
    if not SERPER_API_KEY:
        return []

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "gl": gl, "hl": hl, "num": num}

    async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            return []
        data = resp.json()

    results = []
    for item in data.get("organic", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "description": item.get("snippet", ""),
        })
    return results


async def serper_scrape(url: str) -> str:
    """Call Serper Scrape API to fetch page content.

    Returns plain text content of the page.
    """
    if not SERPER_API_KEY:
        return ""

    api_url = "https://scrape.serper.dev"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"url": url}

    async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
        resp = await client.post(api_url, headers=headers, json=payload)
        if resp.status_code != 200:
            return ""
        data = resp.json()

    return data.get("text", "")


async def web_fetch(url: str) -> str:
    """Fetch a URL directly and return text content."""
    try:
        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text[:50000]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# LangChain Tool wrappers (for create_react_agent)
# ---------------------------------------------------------------------------

from langchain_core.tools import tool


@tool
async def brave_search_tool(query: str) -> str:
    """Search the web using Brave Search. Returns titles, URLs, and snippets."""
    results = await brave_search(query)
    if not results:
        return "No results found."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
async def serper_search_en_tool(query: str) -> str:
    """Search Google (English) using Serper. Returns titles, URLs, and snippets."""
    results = await serper_search(query, gl="us", hl="en")
    if not results:
        return "No results found."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
async def serper_search_tw_tool(query: str) -> str:
    """Search Google (繁體中文) using Serper. Returns titles, URLs, and snippets."""
    results = await serper_search(query, gl="tw", hl="zh-TW")
    if not results:
        return "No results found."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
async def serper_scrape_tool(url: str) -> str:
    """Scrape a URL and return its text content using Serper."""
    text = await serper_scrape(url)
    return text[:20000] if text else "Failed to scrape URL."


@tool
async def web_fetch_tool(url: str) -> str:
    """Fetch a URL directly and return its HTML/text content."""
    text = await web_fetch(url)
    return text[:20000] if text else "Failed to fetch URL."


def get_search_tools() -> list:
    """Return all search tools as LangChain tool objects."""
    tools = []
    if BRAVE_API_KEY:
        tools.append(brave_search_tool)
    if SERPER_API_KEY:
        tools.extend([serper_search_en_tool, serper_search_tw_tool, serper_scrape_tool])
    tools.append(web_fetch_tool)
    return tools
