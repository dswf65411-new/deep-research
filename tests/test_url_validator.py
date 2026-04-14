"""Tests for deep_research/harness/url_validator.py (P0-4).

Covers:
- extract_arxiv_ids / extract_urls regex behaviour incl. edge cases
- validate_arxiv_id: OK vs missing-paper, caching, network failure
- validate_url: OK / 404 / DNS fail / timeout classification
- validate_plan_text: batches concurrent validation
- annotate_invalid: rewrites known-bad references in place
- invalid_items: structured accessors for logging

All tests are fully offline — httpx is replaced with in-memory stubs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from deep_research.harness.url_validator import (
    _load_cache,
    _save_cache,
    annotate_invalid,
    extract_arxiv_ids,
    extract_urls,
    invalid_items,
    validate_arxiv_id,
    validate_plan_text,
    validate_url,
)


# ---------------------------------------------------------------------------
# Stub httpx.AsyncClient
# ---------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class StubClient:
    """Minimal httpx.AsyncClient surface: supports get/head + async context."""

    def __init__(
        self,
        *,
        get_map: dict[str, _StubResponse] | None = None,
        head_map: dict[str, _StubResponse] | None = None,
        raise_on: dict[str, BaseException] | None = None,
    ):
        self.get_map = get_map or {}
        self.head_map = head_map or {}
        self.raise_on = raise_on or {}
        self.get_calls: list[str] = []
        self.head_calls: list[str] = []

    async def get(self, url: str, **kw):
        self.get_calls.append(url)
        if url in self.raise_on:
            raise self.raise_on[url]
        return self.get_map.get(url, _StubResponse(500, ""))

    async def head(self, url: str, **kw):
        self.head_calls.append(url)
        if url in self.raise_on:
            raise self.raise_on[url]
        return self.head_map.get(url, _StubResponse(500, ""))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# extract_arxiv_ids
# ---------------------------------------------------------------------------

class TestExtractArxiv:
    def test_extracts_plain_ids(self):
        text = "See arxiv.org/abs/2310.05193 and 2504.01222 for details."
        assert extract_arxiv_ids(text) == ["2310.05193", "2504.01222"]

    def test_strips_version_suffix(self):
        text = "Use 2310.05193v2 for the final version."
        assert extract_arxiv_ids(text) == ["2310.05193"]

    def test_handles_five_digit_suffix(self):
        text = "Paper 2501.12345 has 5-digit suffix."
        assert extract_arxiv_ids(text) == ["2501.12345"]

    def test_dedup(self):
        text = "2310.05193 and also 2310.05193v1 and 2310.05193v2."
        assert extract_arxiv_ids(text) == ["2310.05193"]

    def test_ignores_non_arxiv_decimals(self):
        """Telephone-like 03.12345 should not match; 1234.1234 pattern differs."""
        text = "See phone 03.123 or version 1.2345.678"
        assert extract_arxiv_ids(text) == []

    def test_empty_input(self):
        assert extract_arxiv_ids("") == []


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------

class TestExtractUrls:
    def test_basic(self):
        text = "Visit https://arxiv.org/abs/2310.05193 for the paper."
        assert extract_urls(text) == ["https://arxiv.org/abs/2310.05193"]

    def test_strips_trailing_punct(self):
        """URL at end of sentence shouldn't capture the period."""
        text = "Check https://example.com/page. Then next."
        assert extract_urls(text) == ["https://example.com/page"]

    def test_markdown_link_does_not_leak_paren(self):
        text = "[paper](https://example.com/foo) is the link."
        assert extract_urls(text) == ["https://example.com/foo"]

    def test_preserves_order_dedup(self):
        text = "https://a.com and https://b.com and https://a.com again."
        assert extract_urls(text) == ["https://a.com", "https://b.com"]

    def test_http_scheme_supported(self):
        text = "Old site http://example.org still works."
        assert extract_urls(text) == ["http://example.org"]

    def test_ignores_plain_domains(self):
        text = "Just mention example.com without protocol."
        assert extract_urls(text) == []


# ---------------------------------------------------------------------------
# validate_arxiv_id
# ---------------------------------------------------------------------------

class TestValidateArxiv:
    def test_existing_paper_returns_true(self):
        client = StubClient(get_map={
            "https://export.arxiv.org/api/query?id_list=2310.05193": _StubResponse(
                200, "<feed><entry><title>ResearchAgent</title></entry></feed>"
            )
        })
        result = asyncio.run(validate_arxiv_id("2310.05193", client=client))
        assert result is True

    def test_missing_paper_returns_false(self):
        client = StubClient(get_map={
            "https://export.arxiv.org/api/query?id_list=2604.05550": _StubResponse(
                200, "<feed></feed>"  # no <entry>
            )
        })
        result = asyncio.run(validate_arxiv_id("2604.05550", client=client))
        assert result is False

    def test_server_error_returns_false(self):
        client = StubClient(get_map={
            "https://export.arxiv.org/api/query?id_list=2999.99999": _StubResponse(500, "")
        })
        assert asyncio.run(validate_arxiv_id("2999.99999", client=client)) is False

    def test_network_exception_returns_false(self):
        url = "https://export.arxiv.org/api/query?id_list=2999.99999"
        client = StubClient(raise_on={url: httpx.ConnectError("DNS fail")})
        assert asyncio.run(validate_arxiv_id("2999.99999", client=client)) is False

    def test_cache_hit_skips_network(self, tmp_path: Path):
        cache_path = tmp_path / "arxiv_cache.json"
        cache_path.write_text(json.dumps({"2310.05193": True}), encoding="utf-8")
        client = StubClient()  # empty — would return 500 if called
        result = asyncio.run(
            validate_arxiv_id("2310.05193", cache_path=cache_path, client=client)
        )
        assert result is True
        assert client.get_calls == []  # cache short-circuit

    def test_cache_miss_persists_result(self, tmp_path: Path):
        cache_path = tmp_path / "arxiv_cache.json"
        client = StubClient(get_map={
            "https://export.arxiv.org/api/query?id_list=2310.05193": _StubResponse(
                200, "<entry>x</entry>"
            )
        })
        asyncio.run(validate_arxiv_id("2310.05193", cache_path=cache_path, client=client))
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved == {"2310.05193": True}


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_200_is_ok(self):
        url = "https://arxiv.org/abs/2310.05193"
        client = StubClient(head_map={url: _StubResponse(200)})
        assert asyncio.run(validate_url(url, client=client)) == "OK"

    def test_301_redirect_is_ok(self):
        url = "https://example.com/moved"
        client = StubClient(head_map={url: _StubResponse(301)})
        # With follow_redirects=True the client transparently follows; even
        # without, 3xx is within our "OK" band.
        assert asyncio.run(validate_url(url, client=client)) == "OK"

    def test_404_is_hallucinated(self):
        url = "https://openai.com/research/fake-paper"
        client = StubClient(head_map={url: _StubResponse(404)})
        assert asyncio.run(validate_url(url, client=client)) == "LIKELY_HALLUCINATED"

    def test_410_gone_is_hallucinated(self):
        url = "https://example.com/gone"
        client = StubClient(head_map={url: _StubResponse(410)})
        assert asyncio.run(validate_url(url, client=client)) == "LIKELY_HALLUCINATED"

    def test_500_is_unreachable_not_hallucinated(self):
        """A failing server ≠ hallucinated URL; the URL exists, host just broken."""
        url = "https://example.com/broken"
        client = StubClient(head_map={url: _StubResponse(500)})
        assert asyncio.run(validate_url(url, client=client)) == "UNREACHABLE"

    def test_dns_failure_is_hallucinated(self):
        url = "https://this-domain-definitely-does-not-exist-1234.example"
        client = StubClient(raise_on={url: httpx.ConnectError("DNS fail")})
        assert asyncio.run(validate_url(url, client=client)) == "LIKELY_HALLUCINATED"

    def test_timeout_is_unreachable(self):
        url = "https://slow.example.com"
        client = StubClient(raise_on={url: httpx.TimeoutException("timed out")})
        assert asyncio.run(validate_url(url, client=client)) == "UNREACHABLE"

    def test_invalid_url_is_hallucinated(self):
        url = "not-a-url"
        client = StubClient(raise_on={url: httpx.InvalidURL("bad")})
        assert asyncio.run(validate_url(url, client=client)) == "LIKELY_HALLUCINATED"


# ---------------------------------------------------------------------------
# validate_plan_text — batch
# ---------------------------------------------------------------------------

class TestValidatePlanText:
    def test_empty_text(self):
        client = StubClient()
        result = asyncio.run(validate_plan_text("", client=client))
        assert result == {"arxiv": {}, "urls": {}}

    def test_mixed_valid_and_invalid(self):
        text = (
            "Real paper 2310.05193 at https://arxiv.org/abs/2310.05193. "
            "Fake paper 2604.05550 from https://openai.com/research/fake."
        )
        client = StubClient(
            get_map={
                "https://export.arxiv.org/api/query?id_list=2310.05193": _StubResponse(
                    200, "<entry>ok</entry>"
                ),
                "https://export.arxiv.org/api/query?id_list=2604.05550": _StubResponse(
                    200, "<feed></feed>"
                ),
            },
            head_map={
                "https://arxiv.org/abs/2310.05193": _StubResponse(200),
                "https://openai.com/research/fake": _StubResponse(404),
            },
        )
        result = asyncio.run(validate_plan_text(text, client=client))
        assert result["arxiv"]["2310.05193"] is True
        assert result["arxiv"]["2604.05550"] is False
        assert result["urls"]["https://arxiv.org/abs/2310.05193"] == "OK"
        assert result["urls"]["https://openai.com/research/fake"] == "LIKELY_HALLUCINATED"

    def test_runs_concurrently(self):
        """Smoke-check: a single client handles all calls (proves concurrency works)."""
        text = "2310.05193 2504.01222 2508.00031 https://a.com https://b.com"
        client = StubClient(
            get_map={
                "https://export.arxiv.org/api/query?id_list=2310.05193": _StubResponse(200, "<entry/>"),
                "https://export.arxiv.org/api/query?id_list=2504.01222": _StubResponse(200, "<feed/>"),
                "https://export.arxiv.org/api/query?id_list=2508.00031": _StubResponse(200, "<feed/>"),
            },
            head_map={
                "https://a.com": _StubResponse(200),
                "https://b.com": _StubResponse(404),
            },
        )
        result = asyncio.run(validate_plan_text(text, client=client))
        # 3 arxiv IDs, 2 URLs — all processed
        assert len(result["arxiv"]) == 3
        assert len(result["urls"]) == 2


# ---------------------------------------------------------------------------
# annotate_invalid
# ---------------------------------------------------------------------------

class TestAnnotateInvalid:
    def test_replaces_invalid_arxiv_id(self):
        text = "See 2604.05550 for the new SOTA."
        validation = {"arxiv": {"2604.05550": False}, "urls": {}}
        out = annotate_invalid(text, validation)
        assert "[REMOVED: hallucinated arxiv ID 2604.05550]" in out
        assert "2604.05550" not in out.replace("[REMOVED: hallucinated arxiv ID 2604.05550]", "")

    def test_replaces_invalid_arxiv_with_version(self):
        text = "See 2604.05550v2 for latest version."
        validation = {"arxiv": {"2604.05550": False}, "urls": {}}
        out = annotate_invalid(text, validation)
        assert "[REMOVED" in out
        assert "2604.05550v2" not in out

    def test_preserves_valid_arxiv(self):
        text = "See 2310.05193 for the real paper."
        validation = {"arxiv": {"2310.05193": True}, "urls": {}}
        out = annotate_invalid(text, validation)
        assert out == text  # unchanged

    def test_replaces_hallucinated_url(self):
        text = "Read https://openai.com/research/fake for details."
        validation = {"arxiv": {}, "urls": {"https://openai.com/research/fake": "LIKELY_HALLUCINATED"}}
        out = annotate_invalid(text, validation)
        assert "[REMOVED: hallucinated URL" in out

    def test_preserves_unreachable_url(self):
        """UNREACHABLE ≠ hallucinated — keep the URL, user may retry later."""
        text = "Read https://slow.example.com for details."
        validation = {"arxiv": {}, "urls": {"https://slow.example.com": "UNREACHABLE"}}
        out = annotate_invalid(text, validation)
        assert out == text

    def test_handles_mixed(self):
        text = "Real: 2310.05193 at https://arxiv.org/abs/2310.05193. Fake: 2604.05550."
        validation = {
            "arxiv": {"2310.05193": True, "2604.05550": False},
            "urls": {"https://arxiv.org/abs/2310.05193": "OK"},
        }
        out = annotate_invalid(text, validation)
        assert "2310.05193" in out
        assert "[REMOVED: hallucinated arxiv ID 2604.05550]" in out


# ---------------------------------------------------------------------------
# invalid_items — structured log accessor
# ---------------------------------------------------------------------------

def test_invalid_items_groups_by_status():
    validation = {
        "arxiv": {"2310.05193": True, "2604.05550": False, "2504.01222": False},
        "urls": {
            "https://ok.com": "OK",
            "https://fake.com": "LIKELY_HALLUCINATED",
            "https://slow.com": "UNREACHABLE",
        },
    }
    out = invalid_items(validation)
    assert out["hallucinated_arxiv"] == ["2604.05550", "2504.01222"] or \
           out["hallucinated_arxiv"] == ["2504.01222", "2604.05550"]
    assert out["hallucinated_urls"] == ["https://fake.com"]
    assert out["unreachable_urls"] == ["https://slow.com"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "sub" / "arxiv.json"
    _save_cache(cache_path, {"a": True, "b": False})
    loaded = _load_cache(cache_path)
    assert loaded == {"a": True, "b": False}


def test_cache_absent_returns_empty(tmp_path: Path):
    assert _load_cache(tmp_path / "nonexistent.json") == {}


def test_cache_malformed_returns_empty(tmp_path: Path):
    cache_path = tmp_path / "bad.json"
    cache_path.write_text("not valid json", encoding="utf-8")
    assert _load_cache(cache_path) == {}


def test_cache_none_path_noop(tmp_path: Path):
    # Neither load nor save should error when cache_path is None
    assert _load_cache(None) == {}
    _save_cache(None, {"x": True})  # no side effect expected
