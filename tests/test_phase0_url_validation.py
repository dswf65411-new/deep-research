"""Integration tests for phase0's arxiv/URL validation hook (P0-4).

Mirrors the 2026-04-14 failure pattern: a plan that includes a mix of
real arxiv references (2310.05193), obvious hallucinations (future-dated
2604.05550), and placeholder URLs (openai.com/research/... returning 404).
The validation step must rewrite the plan so downstream phases can't pick
up the bad references as seeds.

Network is stubbed - these tests run fully offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import deep_research.nodes.phase0 as phase0


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class StubClient:
    def __init__(self, *, get_map=None, head_map=None, raise_on=None):
        self.get_map = get_map or {}
        self.head_map = head_map or {}
        self.raise_on = raise_on or {}

    async def get(self, url: str, **kw):
        if url in self.raise_on:
            raise self.raise_on[url]
        return self.get_map.get(url, _Resp(500))

    async def head(self, url: str, **kw):
        if url in self.raise_on:
            raise self.raise_on[url]
        return self.head_map.get(url, _Resp(500))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOOD_ID = "2310.05193"
_BAD_ID = "2604.05550"
_GOOD_URL = "https://arxiv.org/abs/2310.05193"
_BAD_URL = "https://openai.com/research/nonexistent-paper"

_PLAN_TEXT = f"""# Research Plan

## Query Enrichment
- Real paper: {_GOOD_ID} (ResearchAgent, {_GOOD_URL})
- Placeholder: {_BAD_ID} (unconfirmed)
- Fake link: {_BAD_URL}

## Hard Rules
- Query 2 papers
"""


def _install_stub_client(monkeypatch) -> None:
    """Patch httpx.AsyncClient to return deterministic stub responses."""
    stub = StubClient(
        get_map={
            f"https://export.arxiv.org/api/query?id_list={_GOOD_ID}": _Resp(
                200, "<feed><entry>ok</entry></feed>"
            ),
            f"https://export.arxiv.org/api/query?id_list={_BAD_ID}": _Resp(
                200, "<feed></feed>"
            ),
        },
        head_map={
            _GOOD_URL: _Resp(200),
            _BAD_URL: _Resp(404),
        },
    )

    def _factory(*args, **kwargs):
        return stub

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_validate_annotate_marks_hallucinated_arxiv(tmp_path: Path, monkeypatch):
    _install_stub_client(monkeypatch)
    annotated, log = asyncio.run(
        phase0._validate_and_annotate_plan(_PLAN_TEXT, str(tmp_path))
    )

    # Real paper + URL preserved verbatim
    assert _GOOD_ID in annotated
    assert _GOOD_URL in annotated
    # Hallucinated arxiv gets annotated
    assert f"[REMOVED: hallucinated arxiv ID {_BAD_ID}]" in annotated
    assert _BAD_ID not in annotated.replace(
        f"[REMOVED: hallucinated arxiv ID {_BAD_ID}]", ""
    )
    # Hallucinated URL gets annotated
    assert f"[REMOVED: hallucinated URL {_BAD_URL}]" in annotated


def test_validation_manifest_written(tmp_path: Path, monkeypatch):
    _install_stub_client(monkeypatch)
    asyncio.run(phase0._validate_and_annotate_plan(_PLAN_TEXT, str(tmp_path)))

    manifest = tmp_path / "plan-url-validation.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["arxiv"][_GOOD_ID] is True
    assert data["arxiv"][_BAD_ID] is False
    assert data["urls"][_GOOD_URL] == "OK"
    assert data["urls"][_BAD_URL] == "LIKELY_HALLUCINATED"
    assert _BAD_ID in data["summary"]["hallucinated_arxiv"]
    assert _BAD_URL in data["summary"]["hallucinated_urls"]


def test_execution_log_reports_count(tmp_path: Path, monkeypatch):
    _install_stub_client(monkeypatch)
    _, log = asyncio.run(
        phase0._validate_and_annotate_plan(_PLAN_TEXT, str(tmp_path))
    )

    # At least one log line should say content was removed
    assert any("removed" in line for line in log)
    # The specific arxiv / URL should appear in the log for post-hoc review
    assert any(_BAD_ID in line for line in log)
    assert any(_BAD_URL in line for line in log)


def test_all_valid_plan_is_unchanged(tmp_path: Path, monkeypatch):
    """No hallucinations -> plan text passes through untouched."""
    stub = StubClient(
        get_map={
            f"https://export.arxiv.org/api/query?id_list={_GOOD_ID}": _Resp(
                200, "<entry>ok</entry>"
            ),
        },
        head_map={_GOOD_URL: _Resp(200)},
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: stub)

    plan = f"Only real refs: {_GOOD_ID} at {_GOOD_URL}"
    annotated, log = asyncio.run(
        phase0._validate_and_annotate_plan(plan, str(tmp_path))
    )
    assert annotated == plan
    assert "REMOVED" not in annotated
    # Successful log exists and reports the pass count
    assert any("validation passed" in line for line in log) or any("validation" in line for line in log)


def test_cache_file_written_between_runs(tmp_path: Path, monkeypatch):
    """Rerunning on the same workspace should hit the cache and skip network."""
    stub = StubClient(
        get_map={
            f"https://export.arxiv.org/api/query?id_list={_GOOD_ID}": _Resp(
                200, "<entry>ok</entry>"
            ),
        },
        head_map={_GOOD_URL: _Resp(200)},
    )

    call_count = {"get": 0}
    real_get = stub.get

    async def _counting_get(url, **kw):
        call_count["get"] += 1
        return await real_get(url, **kw)

    stub.get = _counting_get
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: stub)

    plan = f"See {_GOOD_ID} at {_GOOD_URL}"

    asyncio.run(phase0._validate_and_annotate_plan(plan, str(tmp_path)))
    first_calls = call_count["get"]
    assert first_calls >= 1

    asyncio.run(phase0._validate_and_annotate_plan(plan, str(tmp_path)))
    # Second run: arxiv goes through cache; get_calls should not increase (URL HEAD is not cached, goes via head)
    assert call_count["get"] == first_calls


def test_graceful_on_validator_exception(tmp_path: Path, monkeypatch):
    """If validator crashes, plan is returned unchanged - research still proceeds."""

    async def _boom(*args, **kwargs):
        raise RuntimeError("validator blew up")

    monkeypatch.setattr(phase0, "validate_plan_text", _boom)

    plan = f"Plan text with {_BAD_ID} reference"
    annotated, log = asyncio.run(
        phase0._validate_and_annotate_plan(plan, str(tmp_path))
    )
    assert annotated == plan  # fail-open: don't lose the plan
    assert any("validation skipped" in line for line in log)
