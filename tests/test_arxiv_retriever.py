"""Tests for the arxiv retriever (PR #3b).

Covers:
- `arxiv_search` parses a real arxiv Atom response (canonical id, year, url).
- Falsy-corner cases: non-200, empty <feed>, XML parse error → empty list.
- Returned schema matches `serper_scholar` so `_run_single_search` can merge
  arxiv hits with scholar/brave/serper hits without special casing.
- phase1a dispatches engine name `"arxiv"` to the retriever.
- The seed-paper-query helper now asks `arxiv` before `serper_scholar` so we
  exhaust the free source before paying Serper for the same paper.
- `_parse_verdict`-style tolerance is NOT needed here — arxiv returns
  deterministic XML — so we only test the parser's happy + failure paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx

from deep_research.tools.arxiv_retriever import arxiv_search


# Minimal but realistic arxiv Atom 1.0 response.
_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arxiv sample</title>
  <entry>
    <id>http://arxiv.org/abs/2402.04942v1</id>
    <published>2024-02-07T16:34:12Z</published>
    <title>MLE-Agent: Integrating LLM Agents into ML Engineering Loops</title>
    <summary>We introduce MLE-Agent, a language-model-driven ML engineering
      agent that plans, codes and debugs end-to-end experiments.
    </summary>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.04943</id>
    <published>2024-02-08T00:00:00Z</published>
    <title>A second paper</title>
    <summary>Second abstract.</summary>
    <category term="cs.CL"/>
  </entry>
</feed>
"""

_EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arxiv sample</title>
</feed>
"""


class _FakeResp:
    def __init__(self, status: int, text: str):
        self.status_code = status
        self.text = text


class _FakeClient:
    def __init__(self, resp: _FakeResp | Exception):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, *_args, **_kwargs):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _patch_client(resp):
    """Monkey-patch httpx.AsyncClient for the arxiv module."""
    def _factory(*_args, **_kwargs):
        return _FakeClient(resp)
    return patch("deep_research.tools.arxiv_retriever.httpx.AsyncClient", _factory)


# ---------------------------------------------------------------------------
# arxiv_search
# ---------------------------------------------------------------------------


def test_arxiv_search_parses_realistic_atom_feed():
    with _patch_client(_FakeResp(200, _SAMPLE_XML)):
        results = asyncio.run(arxiv_search("MLE-Agent"))
    assert len(results) == 2
    r = results[0]
    assert r["title"].startswith("MLE-Agent")
    assert r["url"] == "https://arxiv.org/abs/2402.04942"
    assert r["year"] == "2024"
    assert r["arxiv_id"] == "2402.04942"
    assert "cs.LG" in r["categories"]
    assert "plans, codes and debugs" in r["description"]


def test_arxiv_search_schema_matches_serper_scholar_keys():
    """Contract test: same keys a downstream `_merge_hits` expects."""
    with _patch_client(_FakeResp(200, _SAMPLE_XML)):
        results = asyncio.run(arxiv_search("Q"))
    required = {"title", "url", "description", "year"}
    for r in results:
        assert required.issubset(r.keys())


def test_arxiv_search_non_200_returns_empty():
    with _patch_client(_FakeResp(503, "upstream error")):
        results = asyncio.run(arxiv_search("anything"))
    assert results == []


def test_arxiv_search_network_error_returns_empty():
    with _patch_client(httpx.ConnectError("connection refused")):
        results = asyncio.run(arxiv_search("anything"))
    assert results == []


def test_arxiv_search_empty_feed_returns_empty():
    with _patch_client(_FakeResp(200, _EMPTY_XML)):
        results = asyncio.run(arxiv_search("nothing matches"))
    assert results == []


def test_arxiv_search_malformed_xml_returns_empty():
    with _patch_client(_FakeResp(200, "<not-xml>")):
        results = asyncio.run(arxiv_search("q"))
    assert results == []


def test_arxiv_search_clamps_max_results():
    """The API can't be asked for 0 or > 30 results; enforced client-side."""
    captured = {}

    class _CaptureClient(_FakeClient):
        async def get(self, url, params=None, **_):
            captured["params"] = params
            return _FakeResp(200, _EMPTY_XML)

    with patch(
        "deep_research.tools.arxiv_retriever.httpx.AsyncClient",
        lambda *a, **k: _CaptureClient(None),
    ):
        asyncio.run(arxiv_search("q", max_results=99))
    assert captured["params"]["max_results"] == "30"


# ---------------------------------------------------------------------------
# phase1a dispatch
# ---------------------------------------------------------------------------


def test_phase1a_dispatches_arxiv_engine(monkeypatch):
    from deep_research.nodes import phase1a

    called = {}

    async def fake_arxiv(query, max_results=10):
        called["query"] = query
        called["max"] = max_results
        return [{"title": "T", "url": "https://arxiv.org/abs/2402.04942",
                 "description": "abs", "year": "2024", "arxiv_id": "2402.04942"}]

    monkeypatch.setattr(phase1a, "arxiv_search", fake_arxiv)

    hits = asyncio.run(phase1a._run_single_search("MLE-Agent", "arxiv"))
    assert called == {"query": "MLE-Agent", "max": 10}
    assert hits[0]["arxiv_id"] == "2402.04942"


def test_phase1a_arxiv_engine_handles_exception(monkeypatch):
    """`_run_single_search` must swallow engine-level exceptions (policy)."""
    from deep_research.nodes import phase1a

    async def raising(*_a, **_k):
        raise RuntimeError("arxiv down")

    monkeypatch.setattr(phase1a, "arxiv_search", raising)
    hits = asyncio.run(phase1a._run_single_search("q", "arxiv"))
    assert hits == []


# ---------------------------------------------------------------------------
# seed-paper-query integration
# ---------------------------------------------------------------------------


def test_seed_paper_queries_prefers_arxiv_first():
    """`arxiv` should come before `serper_scholar` in the engine list for
    arxiv-kind seed queries, so the free source is exhausted first."""
    from deep_research.nodes.phase1a import _seed_paper_queries

    qs = _seed_paper_queries(["MLE-Agent"], sq_ids=["Q1"], budget_cap=10)
    arxiv_q = next(q for q in qs if q["query"] == "MLE-Agent arxiv")
    assert arxiv_q["engines"][0] == "arxiv"
    assert "serper_scholar" in arxiv_q["engines"]
    # github kind should not include arxiv
    github_q = next(q for q in qs if q["query"] == "MLE-Agent github")
    assert "arxiv" not in github_q["engines"]
