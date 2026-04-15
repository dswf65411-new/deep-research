"""Tests for Tongyi-style Heavy Mode (PR #4).

Triggered by `route_after_review` when the Critic-revise loop exhausts its
MAX_REVISIONS budget without an accept verdict. The node rewrites each
flagged section as n=3 rollouts (temperatures 0.3 / 0.7 / 1.0) and a
selector LLM picks the best candidate per section — no merging, per Tongyi
ParallelMuse INTEGRATE_PROMPT:58-66.

Covers:
- `_infer_sq_from_filename`, `_find_section_file` normalise filenames.
- `_select_best` returns the parsed winner_index on clean JSON, falls back
  to 0 on LLM failure / missing JSON / out-of-range index.
- `_generate_rollouts` always returns exactly HEAVY_MODE_N candidates
  (even if some rewrite calls fail, where the duplicate original is
  substituted so the selector still has n entries).
- `heavy_mode_rollout` is a no-op with `heavy_mode_triggered=True` when
  `report-sections/` is empty.
- `heavy_mode_rollout` uses `per_sq_issues` to pick which sections to
  rewrite and writes the selector's winner back to disk.
- When `per_sq_issues` is empty (critic rejected without per-SQ detail)
  every section is rewritten.
- `heavy_mode_rollout` appends an audit block to `review-log.md`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from deep_research.nodes.heavy_mode import (
    HEAVY_MODE_N,
    _find_section_file,
    _generate_rollouts,
    _infer_sq_from_filename,
    _select_best,
    heavy_mode_rollout,
)
from deep_research.state import Claim
from deep_research.tools.workspace import create_workspace, init_gap_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws(tmp_path) -> str:
    ws = create_workspace("heavy-mode-test", base_dir=str(tmp_path))
    init_gap_log(ws)
    return ws


def _seed_section(workspace: str, sq: str, body: str) -> Path:
    """Write a canned section file. Returns the path for assertions."""
    d = Path(workspace) / "report-sections"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sq.lower()}_section.md"
    path.write_text(body, encoding="utf-8")
    return path


def _approved(claim_id: str, sq: str, text: str) -> Claim:
    c = Claim(
        claim_id=claim_id,
        claim_text=text,
        subquestion=sq,
        source_ids=["S001"],
        quote_ids=["QT001"],
    )
    c.status = "approved"
    c.bedrock_score = 0.9
    return c


class _Resp:
    def __init__(self, content: str):
        self.content = content


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


class TestFilenameHelpers:
    def test_infer_sq_from_standard_name(self):
        assert _infer_sq_from_filename("q1_section.md") == "Q1"
        assert _infer_sq_from_filename("Q10_section.md") == "Q10"

    def test_infer_sq_unparseable_returns_empty(self):
        assert _infer_sq_from_filename("summary.md") == ""
        assert _infer_sq_from_filename("q_section.md") == ""

    def test_find_section_file_case_insensitive(self, tmp_path):
        f1 = tmp_path / "q1_section.md"
        f1.write_text("a", encoding="utf-8")
        f2 = tmp_path / "q10_section.md"
        f2.write_text("b", encoding="utf-8")
        assert _find_section_file([f1, f2], "Q1") == f1
        assert _find_section_file([f1, f2], "Q10") == f2
        assert _find_section_file([f1, f2], "Q2") is None


# ---------------------------------------------------------------------------
# _select_best — selector LLM integration
# ---------------------------------------------------------------------------


class TestSelectBest:
    def test_clean_json_returns_parsed_index(self, monkeypatch):
        async def fake_chain(**kwargs):
            return _Resp('{"winner_index": 2, "rationale": "fixes Q1 coverage"}')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        idx = asyncio.run(
            _select_best(sq="Q1", issues=["too short"], candidates=["a", "b", "c"])
        )
        assert idx == 2

    def test_fenced_json_still_parsed(self, monkeypatch):
        async def fake_chain(**kwargs):
            return _Resp('```json\n{"winner_index": 1, "rationale": "ok"}\n```')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )
        idx = asyncio.run(
            _select_best(sq="Q1", issues=[], candidates=["a", "b", "c"])
        )
        assert idx == 1

    def test_llm_failure_returns_zero(self, monkeypatch):
        async def boom(**kwargs):
            raise RuntimeError("selector outage")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", boom
        )
        idx = asyncio.run(
            _select_best(sq="Q1", issues=["x"], candidates=["a", "b", "c"])
        )
        assert idx == 0

    def test_malformed_json_returns_zero(self, monkeypatch):
        async def fake_chain(**kwargs):
            return _Resp("not json at all")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )
        idx = asyncio.run(
            _select_best(sq="Q1", issues=[], candidates=["a", "b", "c"])
        )
        assert idx == 0

    def test_out_of_range_index_returns_zero(self, monkeypatch):
        async def fake_chain(**kwargs):
            return _Resp('{"winner_index": 99, "rationale": "bug"}')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )
        idx = asyncio.run(
            _select_best(sq="Q1", issues=[], candidates=["a", "b", "c"])
        )
        assert idx == 0


# ---------------------------------------------------------------------------
# _generate_rollouts — fan-out behaviour
# ---------------------------------------------------------------------------


class TestGenerateRollouts:
    def test_returns_n_candidates_on_success(self, monkeypatch):
        call_count = 0

        async def fake_chain(**kwargs):
            nonlocal call_count
            call_count += 1
            return _Resp(f"rewritten-{call_count}")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        candidates = asyncio.run(
            _generate_rollouts(
                section_text="## Q1\noriginal draft",
                sq="Q1",
                issues=["too short"],
                approved=[_approved("Q1-C1", "Q1", "fact")],
            )
        )
        assert len(candidates) == HEAVY_MODE_N
        # Rollout 0 is always the original draft
        assert candidates[0] == "## Q1\noriginal draft"
        # Rollouts 1+ go through the LLM (2 calls for HEAVY_MODE_N=3)
        assert call_count == HEAVY_MODE_N - 1

    def test_rewrite_failure_substitutes_original(self, monkeypatch):
        async def fail_chain(**kwargs):
            raise RuntimeError("writer outage")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fail_chain
        )

        original = "## Q1\nthis is the original"
        candidates = asyncio.run(
            _generate_rollouts(
                section_text=original,
                sq="Q1",
                issues=[],
                approved=[],
            )
        )
        # Even when rewrite LLM fails, every position is filled so the
        # selector sees exactly HEAVY_MODE_N candidates.
        assert len(candidates) == HEAVY_MODE_N
        assert all(c == original for c in candidates)


# ---------------------------------------------------------------------------
# heavy_mode_rollout — node
# ---------------------------------------------------------------------------


class TestHeavyModeNode:
    def test_empty_sections_is_noop(self, tmp_path, monkeypatch):
        ws = _ws(tmp_path)
        # No report-sections/ on disk → node should short-circuit.
        state = {
            "workspace_path": ws,
            "review_verdict": {"accept": False, "per_sq_issues": {"Q1": ["x"]}},
            "claims": [],
        }

        async def never_called(**kwargs):
            raise AssertionError("LLM must not be called when no sections exist")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", never_called
        )

        out = asyncio.run(heavy_mode_rollout(state))
        assert out["heavy_mode_triggered"] is True

    def test_rewrites_flagged_sections_and_picks_winner(self, tmp_path, monkeypatch):
        ws = _ws(tmp_path)
        section_path = _seed_section(ws, "Q1", "## Q1\noriginal draft [Q1-C1]")
        other_section = _seed_section(ws, "Q2", "## Q2\nuntouched [Q2-C1]")
        other_original = other_section.read_text(encoding="utf-8")

        rewrite_counter = 0

        async def fake_chain(**kwargs):
            """First two writer calls produce rewrites; selector picks index 2."""
            nonlocal rewrite_counter
            # verifier role ≡ selector; writer role ≡ rewrite
            for msg in kwargs.get("messages", []):
                content = getattr(msg, "content", "")
                if "numbered candidate rewrites" in content or "Pick exactly ONE" in content:
                    # Selector path — look at the system message to disambiguate
                    pass
            # Simpler: dispatch by `role` kwarg
            role = kwargs.get("role")
            if role == "writer":
                rewrite_counter += 1
                return _Resp(f"## Q1\nrewrite #{rewrite_counter} [Q1-C1]")
            if role == "verifier":
                return _Resp('{"winner_index": 2, "rationale": "best coverage"}')
            return _Resp("")

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        state = {
            "workspace_path": ws,
            "review_verdict": {
                "accept": False,
                "per_sq_issues": {"Q1": ["cites only 1 of 3 approved claims"]},
            },
            "claims": [
                _approved("Q1-C1", "Q1", "fact 1"),
                _approved("Q1-C2", "Q1", "fact 2"),
                _approved("Q1-C3", "Q1", "fact 3"),
            ],
        }

        out = asyncio.run(heavy_mode_rollout(state))
        assert out["heavy_mode_triggered"] is True

        # Winner (rollout #2) written back to disk.
        winning = section_path.read_text(encoding="utf-8")
        assert "rewrite #2" in winning

        # Untouched Q2 section is preserved byte-for-byte (only flagged
        # sections are rewritten).
        assert other_section.read_text(encoding="utf-8") == other_original

    def test_empty_per_sq_issues_rewrites_all(self, tmp_path, monkeypatch):
        ws = _ws(tmp_path)
        q1 = _seed_section(ws, "Q1", "## Q1\norig1")
        q2 = _seed_section(ws, "Q2", "## Q2\norig2")

        rewrite_count = 0

        async def fake_chain(**kwargs):
            nonlocal rewrite_count
            if kwargs.get("role") == "writer":
                rewrite_count += 1
                return _Resp(f"rewritten content for call {rewrite_count}")
            return _Resp('{"winner_index": 0, "rationale": "draft wins"}')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        state = {
            "workspace_path": ws,
            "review_verdict": {"accept": False, "per_sq_issues": {}},
            "claims": [],
        }
        out = asyncio.run(heavy_mode_rollout(state))
        assert out["heavy_mode_triggered"] is True
        # 2 sections × (HEAVY_MODE_N - 1) rewrite calls = 4 writer invocations
        assert rewrite_count == 2 * (HEAVY_MODE_N - 1)

    def test_selector_picks_zero_keeps_original(self, tmp_path, monkeypatch):
        """Winner index 0 = keep the current draft. Sanity check the
        end-to-end wiring: winning candidate text lands on disk unchanged."""
        ws = _ws(tmp_path)
        path = _seed_section(ws, "Q1", "## Q1\nkeep me")

        async def fake_chain(**kwargs):
            if kwargs.get("role") == "writer":
                return _Resp("## Q1\ncandidate rewrite")
            return _Resp('{"winner_index": 0, "rationale": "draft is fine"}')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        state = {
            "workspace_path": ws,
            "review_verdict": {"accept": False, "per_sq_issues": {"Q1": ["x"]}},
            "claims": [],
        }
        asyncio.run(heavy_mode_rollout(state))
        assert path.read_text(encoding="utf-8") == "## Q1\nkeep me"

    def test_appends_audit_log_to_review_log(self, tmp_path, monkeypatch):
        ws = _ws(tmp_path)
        _seed_section(ws, "Q1", "## Q1\ndraft")

        async def fake_chain(**kwargs):
            if kwargs.get("role") == "writer":
                return _Resp("## Q1\nrewritten [Q1-C1]")
            return _Resp('{"winner_index": 1, "rationale": "better"}')

        monkeypatch.setattr(
            "deep_research.nodes.heavy_mode.safe_ainvoke_chain", fake_chain
        )

        state = {
            "workspace_path": ws,
            "review_verdict": {"accept": False, "per_sq_issues": {"Q1": ["x"]}},
            "claims": [],
        }
        asyncio.run(heavy_mode_rollout(state))

        log = (Path(ws) / "review-log.md").read_text(encoding="utf-8")
        assert "## Heavy Mode rollout" in log
        assert f"rollouts per section: {HEAVY_MODE_N}" in log
        assert "Q1:rollout#1" in log


# ---------------------------------------------------------------------------
# graph-level — route_after_review integration
# ---------------------------------------------------------------------------


def test_graph_routing_triggers_heavy_mode():
    """Integration: when the critic stays unhappy at MAX_REVISIONS and
    heavy_mode has not run, route_after_review dispatches to heavy_mode
    (this is what chains the node into the graph after phase2_review)."""
    from deep_research.graph import route_after_review
    from deep_research.nodes.phase2 import MAX_REVISIONS

    state = {
        "review_verdict": {"accept": False, "per_sq_issues": {"Q1": ["x"]}},
        "revision_count": MAX_REVISIONS,
    }
    assert route_after_review(state) == "heavy_mode"
