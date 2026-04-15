"""Tests for the benchmark regression harness (PR #3a).

Covers:
- `load_dataset` parses the bundled YAML and catches schema errors early
  (missing fields, duplicate ids, invalid difficulty, empty must_include).
- `BenchmarkQuestion.must_include` anchors the judge — empty list is rejected.
- `grade_report` tolerates markdown-fenced / prose-wrapped / malformed JSON
  and returns an all-MISSING verdict instead of raising.
- `JudgeVerdict.passed` interprets grades correctly (all PRESENT → pass,
  any MISSING must_include → fail, any PRESENT must_not_include → fail).
- `grade_workspace` returns all-MISSING when final-report.md is absent.
- `Scorecard.scores_by_category / by_difficulty` aggregate correctly.
- `grade_many` respects concurrency limits and records missing reports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_research.benchmarks.dataset import (
    BENCHMARK_DATASET_PATH,
    BenchmarkQuestion,
    load_dataset,
)
from deep_research.benchmarks.judge import (
    ClaimGrade,
    JudgeVerdict,
    _parse_verdict,
    _verdict_all_missing,
    grade_report,
)
from deep_research.benchmarks.runner import (
    Scorecard,
    grade_many,
    grade_workspace,
)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class TestDatasetLoader:
    def test_bundled_dataset_loads(self):
        ds = load_dataset()
        assert len(ds) >= 10
        assert all(isinstance(q, BenchmarkQuestion) for q in ds)
        # Every id unique, stable format BQ###
        ids = [q.id for q in ds]
        assert len(ids) == len(set(ids))
        assert all(q.id.startswith("BQ") for q in ds)

    def test_bundled_dataset_covers_multiple_categories(self):
        ds = load_dataset()
        cats = {q.category for q in ds}
        # At minimum cover the categories the pipeline has struggled with
        assert {"academic", "comparison", "how-to"}.issubset(cats)

    def test_missing_required_field_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "- id: X1\n  question: q?\n  # must_include missing\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must_include"):
            load_dataset(bad)

    def test_empty_must_include_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "- id: X1\n  question: q?\n  must_include: []\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must_include.*empty"):
            load_dataset(bad)

    def test_duplicate_id_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "- id: DUP\n  question: a?\n  must_include: [x]\n"
            "- id: DUP\n  question: b?\n  must_include: [y]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_dataset(bad)

    def test_invalid_difficulty_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "- id: X1\n  question: q?\n  must_include: [x]\n  difficulty: impossible\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="difficulty"):
            load_dataset(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.yaml")

    def test_root_must_be_list(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("id: X1\nquestion: q?\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_dataset(bad)


# ---------------------------------------------------------------------------
# JudgeVerdict behaviour
# ---------------------------------------------------------------------------

class TestJudgeVerdict:
    def _q(self) -> BenchmarkQuestion:
        return BenchmarkQuestion(
            id="BQ-T",
            question="t?",
            must_include=["fact A", "fact B"],
            must_not_include=["bad claim"],
            category="general",
            difficulty="medium",
        )

    def test_all_present_passes(self):
        v = JudgeVerdict(
            question_id="BQ-T",
            must_include_grades=[
                ClaimGrade("fact A", "PRESENT", ""),
                ClaimGrade("fact B", "PRESENT", ""),
            ],
            must_not_include_violations=[
                ClaimGrade("bad claim", "MISSING", ""),
            ],
            overall_score=1.0,
        )
        assert v.passed is True

    def test_partial_still_passes(self):
        """PARTIAL is a soft-pass: the topic is addressed, score is docked."""
        v = JudgeVerdict(
            question_id="BQ-T",
            must_include_grades=[
                ClaimGrade("fact A", "PRESENT", ""),
                ClaimGrade("fact B", "PARTIAL", ""),
            ],
            overall_score=0.75,
        )
        assert v.passed is True

    def test_any_missing_fails(self):
        v = JudgeVerdict(
            question_id="BQ-T",
            must_include_grades=[
                ClaimGrade("fact A", "PRESENT", ""),
                ClaimGrade("fact B", "MISSING", ""),
            ],
            overall_score=0.5,
        )
        assert v.passed is False

    def test_must_not_include_violation_fails(self):
        v = JudgeVerdict(
            question_id="BQ-T",
            must_include_grades=[
                ClaimGrade("fact A", "PRESENT", ""),
                ClaimGrade("fact B", "PRESENT", ""),
            ],
            must_not_include_violations=[
                ClaimGrade("bad claim", "PRESENT", "found in report"),
            ],
            overall_score=0.5,
        )
        assert v.passed is False


# ---------------------------------------------------------------------------
# _parse_verdict — tolerant JSON parsing
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def _q(self) -> BenchmarkQuestion:
        return BenchmarkQuestion(
            id="BQ-X",
            question="q?",
            must_include=["A"],
            must_not_include=["bad"],
            category="g",
            difficulty="easy",
        )

    def test_clean_json(self):
        raw = (
            '{"question_id":"BQ-X","must_include_grades":'
            '[{"claim":"A","status":"PRESENT","justification":"found"}],'
            '"must_not_include_violations":[],"overall_score":1.0,"judge_notes":""}'
        )
        v = _parse_verdict(raw, self._q())
        assert v.question_id == "BQ-X"
        assert len(v.must_include_grades) == 1
        assert v.must_include_grades[0].status == "PRESENT"
        assert v.overall_score == 1.0

    def test_fenced_json(self):
        raw = (
            "```json\n"
            '{"question_id":"BQ-X","must_include_grades":'
            '[{"claim":"A","status":"PARTIAL","justification":"close"}],'
            '"overall_score":0.5}\n'
            "```"
        )
        v = _parse_verdict(raw, self._q())
        assert v.must_include_grades[0].status == "PARTIAL"

    def test_malformed_defaults_all_missing(self):
        v = _parse_verdict("not json at all", self._q())
        # Defaults: every must_include → MISSING, score 0.0
        assert all(g.status == "MISSING" for g in v.must_include_grades)
        assert v.overall_score == 0.0
        assert "no JSON" in v.judge_notes

    def test_score_clamped(self):
        raw = '{"question_id":"BQ-X","must_include_grades":[],"overall_score":2.5}'
        v = _parse_verdict(raw, self._q())
        assert v.overall_score == 1.0

    def test_unknown_status_coerced_to_missing(self):
        raw = (
            '{"question_id":"BQ-X","must_include_grades":'
            '[{"claim":"A","status":"WEIRD","justification":"?"}],"overall_score":0.5}'
        )
        v = _parse_verdict(raw, self._q())
        assert v.must_include_grades[0].status == "MISSING"


# ---------------------------------------------------------------------------
# grade_report — async LLM call
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, content: str):
        self.content = content


def test_grade_report_calls_llm_and_parses_verdict(monkeypatch):
    """End-to-end check: judge LLM returns JSON → grade_report → verdict."""
    q = BenchmarkQuestion(
        id="BQ-T",
        question="what is X?",
        must_include=["X is Y"],
        must_not_include=[],
        category="g",
        difficulty="easy",
    )
    json_reply = (
        '{"question_id":"BQ-T","must_include_grades":'
        '[{"claim":"X is Y","status":"PRESENT","justification":"cited"}],'
        '"must_not_include_violations":[],"overall_score":1.0,"judge_notes":"ok"}'
    )

    async def fake_chain(**kwargs):
        return _Resp(json_reply)

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", fake_chain)

    verdict = asyncio.run(grade_report(q, "Report text where X is Y is stated."))
    assert verdict.passed
    assert verdict.overall_score == 1.0


def test_grade_report_handles_llm_failure(monkeypatch):
    q = BenchmarkQuestion(
        id="BQ-T",
        question="q?",
        must_include=["A"],
        must_not_include=[],
        category="g",
        difficulty="easy",
    )

    async def boom(**kwargs):
        raise RuntimeError("judge outage")

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", boom)

    verdict = asyncio.run(grade_report(q, "some report"))
    assert verdict.overall_score == 0.0
    assert "judge LLM call failed" in verdict.judge_notes
    assert all(g.status == "MISSING" for g in verdict.must_include_grades)


def test_verdict_all_missing_populates_both_lists():
    q = BenchmarkQuestion(
        id="BQ-T",
        question="q?",
        must_include=["A", "B"],
        must_not_include=["bad"],
        category="g",
        difficulty="easy",
    )
    v = _verdict_all_missing(q, "test reason")
    assert len(v.must_include_grades) == 2
    assert len(v.must_not_include_violations) == 1
    assert all(g.status == "MISSING" for g in v.must_include_grades)
    assert v.judge_notes == "test reason"


# ---------------------------------------------------------------------------
# grade_workspace / grade_many / Scorecard
# ---------------------------------------------------------------------------

def test_grade_workspace_missing_report_returns_all_missing(tmp_path, monkeypatch):
    q = BenchmarkQuestion(
        id="BQ-X",
        question="q?",
        must_include=["A"],
        must_not_include=[],
        category="g",
        difficulty="easy",
    )

    async def should_not_call(**kwargs):
        raise AssertionError("judge must not be called when report is missing")

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", should_not_call)

    verdict = asyncio.run(grade_workspace(tmp_path, q))
    assert verdict.overall_score == 0.0
    assert "not found" in verdict.judge_notes


def test_grade_workspace_reads_final_report(tmp_path, monkeypatch):
    q = BenchmarkQuestion(
        id="BQ-Y",
        question="q?",
        must_include=["A"],
        must_not_include=[],
        category="g",
        difficulty="easy",
    )
    (tmp_path / "final-report.md").write_text("content about A", encoding="utf-8")

    captured_report: dict[str, str] = {}

    async def fake_chain(**kwargs):
        for m in kwargs.get("messages", []):
            if m.__class__.__name__ == "HumanMessage":
                captured_report["human"] = m.content
        return _Resp(
            '{"question_id":"BQ-Y","must_include_grades":'
            '[{"claim":"A","status":"PRESENT","justification":"yes"}],'
            '"overall_score":1.0}'
        )

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", fake_chain)

    verdict = asyncio.run(grade_workspace(tmp_path, q))
    assert verdict.passed
    assert "content about A" in captured_report["human"]


def test_scorecard_pass_rate_and_mean():
    verdicts = [
        JudgeVerdict("BQ1", [ClaimGrade("A", "PRESENT", "")], [], 1.0),
        JudgeVerdict("BQ2", [ClaimGrade("B", "MISSING", "")], [], 0.0),
        JudgeVerdict("BQ3", [ClaimGrade("C", "PARTIAL", "")], [], 0.5),
    ]
    card = Scorecard(verdicts=verdicts)
    assert card.pass_rate == pytest.approx(2 / 3)  # 2 of 3 passed (all-PRESENT or PARTIAL)
    assert card.mean_score == pytest.approx(0.5)


def test_scorecard_by_category_aggregates():
    questions = [
        BenchmarkQuestion("BQ1", "?", ["A"], [], "academic", "easy"),
        BenchmarkQuestion("BQ2", "?", ["B"], [], "academic", "easy"),
        BenchmarkQuestion("BQ3", "?", ["C"], [], "how-to", "easy"),
    ]
    verdicts = [
        JudgeVerdict("BQ1", [ClaimGrade("A", "PRESENT", "")], [], 1.0),
        JudgeVerdict("BQ2", [ClaimGrade("B", "MISSING", "")], [], 0.0),
        JudgeVerdict("BQ3", [ClaimGrade("C", "PRESENT", "")], [], 1.0),
    ]
    card = Scorecard(verdicts=verdicts)
    by_cat = card.scores_by_category(questions)
    assert by_cat == {"academic": 0.5, "how-to": 1.0}


def test_scorecard_by_difficulty_aggregates():
    questions = [
        BenchmarkQuestion("BQ1", "?", ["A"], [], "academic", "easy"),
        BenchmarkQuestion("BQ2", "?", ["B"], [], "academic", "hard"),
    ]
    verdicts = [
        JudgeVerdict("BQ1", [ClaimGrade("A", "PRESENT", "")], [], 1.0),
        JudgeVerdict("BQ2", [ClaimGrade("B", "PARTIAL", "")], [], 0.5),
    ]
    card = Scorecard(verdicts=verdicts)
    by_diff = card.scores_by_difficulty(questions)
    assert by_diff == {"easy": 1.0, "hard": 0.5}


def test_scorecard_empty_has_zero_rates():
    card = Scorecard()
    assert card.pass_rate == 0.0
    assert card.mean_score == 0.0


def test_grade_many_records_missing_workspaces(tmp_path, monkeypatch):
    """A question with no mapped workspace lands in missing_reports, not in verdicts."""
    q1 = BenchmarkQuestion("BQ1", "?", ["A"], [], "g", "easy")
    q2 = BenchmarkQuestion("BQ2", "?", ["B"], [], "g", "easy")

    ws1 = tmp_path / "ws1"
    ws1.mkdir()
    (ws1 / "final-report.md").write_text("A is there", encoding="utf-8")

    async def fake_chain(**kwargs):
        return _Resp(
            '{"question_id":"BQ1","must_include_grades":'
            '[{"claim":"A","status":"PRESENT","justification":"yes"}],"overall_score":1.0}'
        )

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", fake_chain)

    card = asyncio.run(grade_many({"BQ1": ws1}, questions=[q1, q2]))
    assert len(card.verdicts) == 1
    assert card.missing_reports == ["BQ2"]


def test_grade_many_respects_concurrency_semaphore(tmp_path, monkeypatch):
    """Batch mode must cap concurrent judge calls."""
    questions = []
    ws_map: dict[str, Path] = {}
    for i in range(5):
        q = BenchmarkQuestion(f"BQ{i}", "?", ["A"], [], "g", "easy")
        questions.append(q)
        ws = tmp_path / f"ws{i}"
        ws.mkdir()
        (ws / "final-report.md").write_text("A", encoding="utf-8")
        ws_map[q.id] = ws

    in_flight = 0
    peak = 0

    async def counting_chain(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _Resp(
            '{"question_id":"x","must_include_grades":'
            '[{"claim":"A","status":"PRESENT","justification":"y"}],"overall_score":1.0}'
        )

    monkeypatch.setattr("deep_research.benchmarks.judge.safe_ainvoke_chain", counting_chain)

    asyncio.run(grade_many(ws_map, questions=questions, max_concurrency=2))
    assert peak <= 2
