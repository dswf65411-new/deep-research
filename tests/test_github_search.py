"""Tests for Whisper plan P1-4 — GitHub Search API retriever.

Parallel to ``test_arxiv_retriever.py``. Verifies:
- ``github_repo_search`` returns the normalised schema that the downstream
  ``_select_urls_by_quota`` / ``_merge_hits`` pipeline expects.
- Auth header picks up ``GITHUB_TODO_PAT`` first, then ``GITHUB_TOKEN``,
  then goes anonymous.
- Sort parameter maps correctly (stars/updated/best-match).
- Empty / error responses don't raise.
- ``phase1a._run_single_search`` dispatches ``"github"`` to the retriever.
- ``_seed_paper_queries`` puts the ``"github"`` engine into the github
  seed query list so the new API actually gets called.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

import deep_research.tools.github_search as gh


def _fake_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=payload,
        request=httpx.Request("GET", gh._GITHUB_REPO_SEARCH),
    )


class _Recorder:
    """Captures the (url, params, headers) of each GET the client makes."""

    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status
        self.last_url: str | None = None
        self.last_params: dict | None = None
        self.last_headers: dict | None = None

    async def get(self, url, *, params=None, headers=None):
        self.last_url = url
        self.last_params = dict(params or {})
        self.last_headers = dict(headers or {})
        return _fake_response(self.payload, self.status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _install_recorder(monkeypatch, recorder):
    monkeypatch.setattr(gh.httpx, "AsyncClient", lambda **_k: recorder)


def test_returns_normalised_hits(monkeypatch):
    payload = {
        "items": [
            {
                "full_name": "langchain-ai/langgraph",
                "html_url": "https://github.com/langchain-ai/langgraph",
                "description": "Build stateful agents with LLMs",
                "stargazers_count": 15234,
                "language": "Python",
                "pushed_at": "2026-01-10T03:22:11Z",
            },
        ],
    }
    _install_recorder(monkeypatch, _Recorder(payload))

    hits = asyncio.run(gh.github_repo_search("langgraph", max_results=5))
    assert len(hits) == 1
    h = hits[0]
    assert h["title"].startswith("langchain-ai/langgraph")
    assert "15234" in h["title"]
    assert h["url"] == "https://github.com/langchain-ai/langgraph"
    assert h["stars"] == 15234
    assert h["language"] == "Python"
    assert h["year"] == "2026"
    # Keys required by downstream _merge_hits / selection — schema parity
    # with arxiv_search hits.
    assert {"title", "url", "description", "year"} <= h.keys()


def test_clamps_max_results(monkeypatch):
    r = _Recorder({"items": []})
    _install_recorder(monkeypatch, r)
    asyncio.run(gh.github_repo_search("x", max_results=999))
    assert r.last_params["per_page"] == 30

    r2 = _Recorder({"items": []})
    _install_recorder(monkeypatch, r2)
    asyncio.run(gh.github_repo_search("x", max_results=0))
    assert r2.last_params["per_page"] == 1


def test_sort_mapping(monkeypatch):
    for sort in ("stars", "updated"):
        r = _Recorder({"items": []})
        _install_recorder(monkeypatch, r)
        asyncio.run(gh.github_repo_search("x", sort=sort))
        assert r.last_params["sort"] == sort
        assert r.last_params["order"] == "desc"

    r_best = _Recorder({"items": []})
    _install_recorder(monkeypatch, r_best)
    asyncio.run(gh.github_repo_search("x", sort="best-match"))
    # best-match → omit sort entirely so GitHub uses relevance default
    assert "sort" not in r_best.last_params


def test_auth_header_prefers_todo_pat(monkeypatch):
    monkeypatch.setenv("GITHUB_TODO_PAT", "ghp_primary")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secondary")
    r = _Recorder({"items": []})
    _install_recorder(monkeypatch, r)
    asyncio.run(gh.github_repo_search("x"))
    assert r.last_headers.get("Authorization") == "Bearer ghp_primary"


def test_auth_header_falls_back_to_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TODO_PAT", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_backup")
    r = _Recorder({"items": []})
    _install_recorder(monkeypatch, r)
    asyncio.run(gh.github_repo_search("x"))
    assert r.last_headers.get("Authorization") == "Bearer ghp_backup"


def test_auth_header_omitted_when_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TODO_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = _Recorder({"items": []})
    _install_recorder(monkeypatch, r)
    asyncio.run(gh.github_repo_search("x"))
    assert "Authorization" not in r.last_headers


def test_non_200_returns_empty(monkeypatch):
    _install_recorder(monkeypatch, _Recorder({"message": "rate limited"}, status=403))
    hits = asyncio.run(gh.github_repo_search("x"))
    assert hits == []


def test_network_error_returns_empty(monkeypatch):
    class _Boom:
        async def __aenter__(self):
            raise httpx.ConnectError("offline")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(gh.httpx, "AsyncClient", lambda **_k: _Boom())
    hits = asyncio.run(gh.github_repo_search("x"))
    assert hits == []


def test_missing_fields_filtered_out(monkeypatch):
    payload = {
        "items": [
            {"description": "no url", "stargazers_count": 1},
            {"full_name": "ok/repo", "html_url": "https://github.com/ok/repo"},
        ],
    }
    _install_recorder(monkeypatch, _Recorder(payload))
    hits = asyncio.run(gh.github_repo_search("x"))
    assert len(hits) == 1
    assert hits[0]["url"] == "https://github.com/ok/repo"


# ---------------------------------------------------------------------------
# phase1a wiring
# ---------------------------------------------------------------------------


def test_phase1a_dispatches_github_engine(monkeypatch):
    """_run_single_search must route ``engine="github"`` to the new retriever."""
    import deep_research.nodes.phase1a as p1a

    called: dict = {}

    async def fake(query, max_results=10, sort="stars"):
        called["query"] = query
        called["sort"] = sort
        return [{"title": "t", "url": "u", "description": "d", "year": ""}]

    monkeypatch.setattr(p1a, "github_repo_search", fake)
    hits = asyncio.run(p1a._run_single_search("AIDE agent", "github"))
    assert called["query"] == "AIDE agent"
    assert called["sort"] == "stars"
    assert hits and hits[0]["url"] == "u"


def test_seed_paper_queries_include_github_engine():
    """github seed queries must go through the new ``github`` engine so the
    API actually gets exercised."""
    import deep_research.nodes.phase1a as p1a

    queries = p1a._seed_paper_queries(
        entities=["AIDE", "MLE-Agent"],
        sq_ids=["Q1", "Q2"],
        budget_cap=10,
    )
    github_qs = [q for q in queries if q["query"].endswith("github")]
    assert github_qs, "expected at least one github seed query"
    for q in github_qs:
        assert "github" in q["engines"], q
