"""Tests for Phase 2 Critic-revise loop (PR #2).

Pattern ported from gpt-researcher multi_agents/agents/editor.py::_draft_article
where a reviewer agent critiques the draft and a revisor rewrites the flagged
sections, capped at max_revisions=2. Here the critic is `phase2_review`, the
rewrite is `phase2_integrate` re-entered with the prior verdict as state, and
the cap is enforced by `route_after_review`.

Covers:
- `_parse_review_verdict` handles raw JSON, ```json fenced, trailing prose,
  invalid JSON, and missing braces without raising.
- `_write_review_log` appends each round's verdict so the log accumulates
  across revisions.
- `phase2_review` writes the verdict to state and to review-log.md; returns
  default accept when the critic LLM fails.
- `phase2_review` short-circuits with accept=true when report-sections/ is
  empty (don't block phase3 if phase2 produced nothing).
- `phase2_integrate` injects prior verdict issues as a "Revision requested"
  hint into the writer system prompt when revision_count > 0.
- `route_after_review` routes: accept → phase3; reject+count<max → bump;
  reject+count==max → phase3 (safety cap).
- `bump_revision_count` increments counter by 1.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from deep_research.graph import bump_revision_count, route_after_review
from deep_research.nodes.phase2 import (
    MAX_REVISIONS,
    _parse_review_verdict,
    _write_review_log,
    phase2_integrate,
    phase2_review,
)
from deep_research.state import Claim
from deep_research.tools.workspace import create_workspace, init_gap_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved(claim_id: str, subq: str, text: str, source_ids: list[str] | None = None) -> Claim:
    c = Claim(
        claim_id=claim_id,
        claim_text=text,
        subquestion=subq,
        source_ids=source_ids or ["S001"],
        quote_ids=["QT001"],
    )
    c.status = "approved"
    c.bedrock_score = 0.9
    return c


def _make_workspace(tmp_path) -> str:
    ws = create_workspace("phase2-review-test", base_dir=str(tmp_path))
    init_gap_log(ws)
    return ws


def _seed_sections(workspace: str, sections: dict[str, str]) -> None:
    """Write canned section files so phase2_review has something to read."""
    d = Path(workspace) / "report-sections"
    d.mkdir(parents=True, exist_ok=True)
    for sq, body in sections.items():
        (d / f"{sq.lower()}_section.md").write_text(body, encoding="utf-8")


class _Resp:
    def __init__(self, content: str):
        self.content = content


# ---------------------------------------------------------------------------
# _parse_review_verdict — unit
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_raw_json_accept(self):
        v = _parse_review_verdict('{"accept": true, "issues": [], "per_sq_issues": {}}')
        assert v == {"accept": True, "issues": [], "per_sq_issues": {}}

    def test_json_with_rejects_and_per_sq(self):
        v = _parse_review_verdict(
            '{"accept": false, "issues": ["X"], "per_sq_issues": {"Q1": ["fix this"]}}'
        )
        assert v["accept"] is False
        assert v["issues"] == ["X"]
        assert v["per_sq_issues"] == {"Q1": ["fix this"]}

    def test_markdown_fenced(self):
        v = _parse_review_verdict(
            "```json\n"
            '{"accept": false, "issues": ["empty"], "per_sq_issues": {"Q3": ["truncated"]}}\n'
            "```"
        )
        assert v["accept"] is False
        assert "Q3" in v["per_sq_issues"]

    def test_trailing_prose_ignored(self):
        v = _parse_review_verdict(
            'Here is my verdict:\n{"accept": true, "issues": [], "per_sq_issues": {}}\nThanks!'
        )
        assert v["accept"] is True

    def test_missing_json_defaults_accept(self):
        v = _parse_review_verdict("no JSON here, just chat")
        assert v["accept"] is True
        assert any("no JSON" in i for i in v["issues"])

    def test_invalid_json_defaults_accept(self):
        # Regex will match the outer braces, but json.loads should fail on
        # the unquoted token inside.
        v = _parse_review_verdict('{"accept": true, "issues": [unquoted_token]}')
        assert v["accept"] is True
        assert any("parse failed" in i for i in v["issues"])

    def test_missing_fields_normalised(self):
        v = _parse_review_verdict('{"accept": true}')
        assert v["issues"] == []
        assert v["per_sq_issues"] == {}


# ---------------------------------------------------------------------------
# _write_review_log — persistence
# ---------------------------------------------------------------------------

class TestWriteReviewLog:
    def test_log_captures_round_header(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_review_log(ws, 0, {"accept": True, "issues": [], "per_sq_issues": {}})
        text = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
        assert "## Revision 0" in text
        assert "accept: True" in text

    def test_log_accumulates_across_rounds(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_review_log(ws, 0, {"accept": False, "issues": ["round0"], "per_sq_issues": {"Q1": ["a"]}})
        _write_review_log(ws, 1, {"accept": False, "issues": ["round1"], "per_sq_issues": {"Q2": ["b"]}})
        _write_review_log(ws, 2, {"accept": True, "issues": [], "per_sq_issues": {}})
        text = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
        # All three rounds visible
        assert "## Revision 0" in text
        assert "## Revision 1" in text
        assert "## Revision 2" in text
        # Per-SQ issues land under correct round
        assert "round0" in text and "round1" in text

    def test_log_note_recorded(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_review_log(
            ws, 0,
            {"accept": True, "issues": [], "per_sq_issues": {}},
            note="no sections to review",
        )
        text = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
        assert "note: no sections to review" in text


# ---------------------------------------------------------------------------
# phase2_review — node
# ---------------------------------------------------------------------------

def _run_review(state: dict) -> dict:
    return asyncio.run(phase2_review(state))


def test_review_no_sections_passes_through(tmp_path, monkeypatch):
    """Empty report-sections/ must not trap the pipeline; default-accept
    so phase3 can run (blocker for empty case already logged by phase2)."""
    ws = _make_workspace(tmp_path)

    async def should_not_call(**kwargs):
        raise AssertionError("LLM must not be called when there are no sections")
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", should_not_call)

    out = _run_review({"workspace_path": ws, "revision_count": 0, "claims": []})
    assert out["review_verdict"]["accept"] is True
    log = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
    assert "no sections to review" in log


def test_review_accepts_good_draft(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    _seed_sections(ws, {"Q1": "Analysis paragraph. [Q1-C1]\n\nConfidence: HIGH"})

    async def critic_accept(**kwargs):
        return _Resp('{"accept": true, "issues": [], "per_sq_issues": {}}')
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", critic_accept)

    out = _run_review({"workspace_path": ws, "revision_count": 0, "claims": []})
    assert out["review_verdict"]["accept"] is True

    log = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
    assert "## Revision 0" in log
    assert "accept: True" in log


def test_review_rejects_bad_draft(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    _seed_sections(ws, {"Q1": "## Q1 — section integration failed, using fallback format\n- x [Q1-C1]"})

    verdict_json = (
        '{"accept": false, "issues": ["fallback section detected"], '
        '"per_sq_issues": {"Q1": ["rewrite without fallback header"]}}'
    )
    async def critic_reject(**kwargs):
        return _Resp(verdict_json)
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", critic_reject)

    out = _run_review({"workspace_path": ws, "revision_count": 0, "claims": []})
    assert out["review_verdict"]["accept"] is False
    assert out["review_verdict"]["per_sq_issues"]["Q1"] == ["rewrite without fallback header"]

    log = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
    assert "rewrite without fallback header" in log


def test_review_llm_failure_defaults_accept(tmp_path, monkeypatch):
    """Critic LLM crash must never block the pipeline; default-accept with
    the error recorded in review-log for post-mortem."""
    ws = _make_workspace(tmp_path)
    _seed_sections(ws, {"Q1": "Analysis. [Q1-C1]"})

    async def boom(**kwargs):
        raise RuntimeError("simulated critic failure")
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", boom)

    out = _run_review({"workspace_path": ws, "revision_count": 0, "claims": []})
    assert out["review_verdict"]["accept"] is True
    assert any("critic LLM call failed" in i for i in out["review_verdict"]["issues"])


def test_review_records_revision_count_in_log(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path)
    _seed_sections(ws, {"Q1": "body [Q1-C1]"})

    async def critic_accept(**kwargs):
        return _Resp('{"accept": true, "issues": [], "per_sq_issues": {}}')
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", critic_accept)

    _run_review({"workspace_path": ws, "revision_count": 0, "claims": []})
    _run_review({"workspace_path": ws, "revision_count": 1, "claims": []})

    log = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
    assert "## Revision 0" in log
    assert "## Revision 1" in log


# ---------------------------------------------------------------------------
# route_after_review + bump_revision_count — graph edge
# ---------------------------------------------------------------------------

class TestRouting:
    def test_accept_goes_to_phase3(self):
        state = {
            "review_verdict": {"accept": True, "issues": [], "per_sq_issues": {}},
            "revision_count": 0,
        }
        assert route_after_review(state) == "phase3"

    def test_reject_under_cap_bumps(self):
        state = {
            "review_verdict": {"accept": False, "issues": ["x"], "per_sq_issues": {"Q1": ["y"]}},
            "revision_count": 0,
        }
        assert route_after_review(state) == "bump_revision"
        state["revision_count"] = 1
        assert route_after_review(state) == "bump_revision"

    def test_reject_at_cap_triggers_heavy_mode(self):
        """At max revisions with still-failing critic, route to Heavy Mode
        (not phase3) unless heavy mode has already run once."""
        state = {
            "review_verdict": {"accept": False, "issues": ["still bad"], "per_sq_issues": {"Q1": ["still bad"]}},
            "revision_count": MAX_REVISIONS,
        }
        assert route_after_review(state) == "heavy_mode"

    def test_reject_at_cap_post_heavy_goes_to_phase3(self):
        """After heavy mode fires once, re-entering the cap branch must go
        straight to phase3 — heavy mode is a one-shot last pass."""
        state = {
            "review_verdict": {"accept": False, "issues": ["still bad"], "per_sq_issues": {"Q1": ["still bad"]}},
            "revision_count": MAX_REVISIONS,
            "heavy_mode_triggered": True,
        }
        assert route_after_review(state) == "phase3"

    def test_missing_verdict_treated_as_reject(self):
        # Defensive: no verdict on first call would route to bump; in practice
        # the graph never calls route_after_review before phase2_review runs.
        state = {"revision_count": 0}
        assert route_after_review(state) == "bump_revision"


class TestBumpRevisionCount:
    def test_increments_from_zero(self):
        out = asyncio.run(bump_revision_count({"revision_count": 0}))
        assert out["revision_count"] == 1

    def test_increments_preserves_other_state(self):
        out = asyncio.run(bump_revision_count({"revision_count": 1}))
        assert out["revision_count"] == 2

    def test_missing_counter_defaults_to_zero(self):
        out = asyncio.run(bump_revision_count({}))
        assert out["revision_count"] == 1


# ---------------------------------------------------------------------------
# phase2_integrate revision-hint injection
# ---------------------------------------------------------------------------

def test_phase2_injects_revision_hint_on_second_pass(tmp_path, monkeypatch):
    """When revision_count > 0 and review_verdict has per_sq_issues, the
    writer system prompt must carry a 'Revision requested' block naming
    those issues so the LLM knows what to fix."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "first claim")]

    captured_prompts: list[str] = []

    async def capture_refine(**kwargs):
        captured_prompts.append(kwargs.get("system_prompt", ""))
        return "Revised paragraph. [Q1-C1]\n\nConfidence: HIGH"

    class _Ok:
        content = "Revised paragraph. [Q1-C1]"

    async def capture_invoke(**kwargs):
        # Extract the SystemMessage content from the messages list
        for m in kwargs.get("messages", []):
            if getattr(m, "type", None) == "system" or m.__class__.__name__ == "SystemMessage":
                captured_prompts.append(m.content)
                break
        return _Ok()

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", capture_refine)
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", capture_invoke)

    state = {
        "workspace_path": ws,
        "claims": claims,
        "sources": [],
        "blockers": [],
        "revision_count": 1,
        "review_verdict": {
            "accept": False,
            "issues": ["top-level issue A"],
            "per_sq_issues": {"Q1": ["specific Q1 fix"]},
        },
    }
    asyncio.run(phase2_integrate(state))

    assert captured_prompts, "writer must have been called"
    joined = "\n".join(captured_prompts)
    assert "Revision requested" in joined
    assert "specific Q1 fix" in joined
    assert "top-level issue A" in joined


def test_phase2_no_revision_hint_on_first_pass(tmp_path, monkeypatch):
    """First pass (revision_count=0) must not inject any 'Revision requested'
    block — there is no prior verdict to revise from."""
    ws = _make_workspace(tmp_path)
    claims = [_approved("Q1-C1", "Q1", "first claim")]

    captured: list[str] = []

    async def capture_refine(**kwargs):
        captured.append(kwargs.get("system_prompt", ""))
        return "Fresh paragraph. [Q1-C1]"

    class _Ok:
        content = "Fresh paragraph. [Q1-C1]"

    async def capture_invoke(**kwargs):
        for m in kwargs.get("messages", []):
            if m.__class__.__name__ == "SystemMessage":
                captured.append(m.content)
                break
        return _Ok()

    monkeypatch.setattr("deep_research.nodes.phase2.iterative_refine", capture_refine)
    monkeypatch.setattr("deep_research.nodes.phase2.safe_ainvoke_chain", capture_invoke)

    asyncio.run(phase2_integrate({
        "workspace_path": ws,
        "claims": claims,
        "sources": [],
        "blockers": [],
        # revision_count defaults to 0, no review_verdict
    }))

    assert captured
    joined = "\n".join(captured)
    assert "Revision requested" not in joined
