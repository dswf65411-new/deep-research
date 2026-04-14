"""Verify the Scholar engine hits /scholar (not the web /search) and preserves pdfUrl.

The old phase1a had a hacky `site:arxiv.org OR ...` prefix on regular web search.
That returns marketing blogs with arxiv in the URL before the abstract itself, and
strips publicationInfo / citation counts. Borrowing Tongyi DeepResearch's
tool_scholar.py design, we now call Serper's dedicated /scholar endpoint.
"""

import asyncio

import httpx

from deep_research.tools import search as search_module
from deep_research.tools.search import serper_scholar


class _FakeResp:
    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


def _make_fake_client(data: dict, seen: dict | None = None):
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, headers=None, json=None):
            if seen is not None:
                seen["url"] = url
                seen["payload"] = json
                seen["api_key"] = headers.get("X-API-KEY") if headers else None
            return _FakeResp(data)

    return _FakeClient


def test_serper_scholar_calls_scholar_endpoint(monkeypatch):
    """/scholar, not /search — the whole point of the migration."""
    seen: dict = {}
    monkeypatch.setattr(search_module, "SERPER_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", _make_fake_client({"organic": []}, seen))

    asyncio.run(serper_scholar("AIDE automl agent", num=5))

    assert seen["url"] == "https://google.serper.dev/scholar"
    assert seen["payload"] == {"q": "AIDE automl agent", "num": 5}
    assert seen["api_key"] == "test-key"


def test_serper_scholar_prefers_pdf_url(monkeypatch):
    """pdfUrl (direct paper PDF) is strictly more useful than the landing link
    for downstream fetch — put it first and fall back to link if absent."""
    data = {
        "organic": [
            {
                "title": "AIDE Agent",
                "link": "https://openreview.net/forum?id=abc",
                "pdfUrl": "https://arxiv.org/pdf/2502.13957.pdf",
                "snippet": "We propose AIDE...",
                "year": "2025",
                "citedBy": 17,
            },
            {
                "title": "Plain paper",
                "link": "https://example.com/paper",
                "snippet": "Some abstract...",
            },
        ]
    }
    monkeypatch.setattr(search_module, "SERPER_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", _make_fake_client(data))

    results = asyncio.run(serper_scholar("test"))
    assert results[0]["url"] == "https://arxiv.org/pdf/2502.13957.pdf"
    assert results[0]["citedBy"] == 17
    assert results[0]["year"] == "2025"
    assert results[1]["url"] == "https://example.com/paper"


def test_serper_scholar_no_key_returns_empty(monkeypatch):
    """Missing key: don't crash, don't hit the wire, return []."""
    monkeypatch.setattr(search_module, "SERPER_API_KEY", "")
    assert asyncio.run(serper_scholar("anything")) == []
