"""Benchmark runner: grade pre-existing workspaces against the dataset.

Offline-only for this PR: pointed at a directory of workspaces (or a single
one), matches each `final-report.md` to a `BenchmarkQuestion` by id, and
produces a `Scorecard` aggregating per-question verdicts + per-category
averages.

Online mode (invoking the full graph) lives on the roadmap but is deferred:
it needs CI bandwidth decisions (cost per run, rate-limit budget) that belong
in a separate PR.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from deep_research.benchmarks.dataset import BenchmarkQuestion, load_dataset
from deep_research.benchmarks.judge import JudgeVerdict, grade_report

logger = logging.getLogger(__name__)


@dataclass
class Scorecard:
    """Aggregate result of running the benchmark over a set of workspaces.

    The per-category and per-difficulty breakdowns feed regression-detection:
    a drop in the "academic" category after a phase1a prompt change is a
    sharper signal than an overall average wobble.
    """
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    missing_reports: list[str] = field(default_factory=list)  # question ids with no workspace found

    @property
    def pass_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.passed) / len(self.verdicts)

    @property
    def mean_score(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.overall_score for v in self.verdicts) / len(self.verdicts)

    def scores_by_category(self, questions: list[BenchmarkQuestion]) -> dict[str, float]:
        """Per-category mean score. Categories with no verdicts are omitted."""
        cat_by_id = {q.id: q.category for q in questions}
        buckets: dict[str, list[float]] = {}
        for v in self.verdicts:
            cat = cat_by_id.get(v.question_id, "unknown")
            buckets.setdefault(cat, []).append(v.overall_score)
        return {cat: sum(xs) / len(xs) for cat, xs in buckets.items() if xs}

    def scores_by_difficulty(self, questions: list[BenchmarkQuestion]) -> dict[str, float]:
        diff_by_id = {q.id: q.difficulty for q in questions}
        buckets: dict[str, list[float]] = {}
        for v in self.verdicts:
            diff = diff_by_id.get(v.question_id, "unknown")
            buckets.setdefault(diff, []).append(v.overall_score)
        return {diff: sum(xs) / len(xs) for diff, xs in buckets.items() if xs}


async def grade_workspace(
    workspace: Path | str,
    question: BenchmarkQuestion,
) -> JudgeVerdict:
    """Grade one workspace against one benchmark question.

    Expects `workspace/final-report.md` to exist. On missing or empty report,
    returns an all-MISSING verdict so the scorecard visibly reflects the
    missing output rather than silently scoring 0.
    """
    report_path = Path(workspace) / "final-report.md"
    if not report_path.exists():
        from deep_research.benchmarks.judge import _verdict_all_missing
        return _verdict_all_missing(question, f"final-report.md not found at {report_path}")
    report_text = report_path.read_text(encoding="utf-8")
    return await grade_report(question, report_text)


async def grade_many(
    workspace_map: dict[str, Path | str],
    questions: list[BenchmarkQuestion] | None = None,
    max_concurrency: int = 3,
) -> Scorecard:
    """Grade a batch of workspaces, one per benchmark question.

    Args:
        workspace_map: `{question_id: workspace_path}`. Any benchmark question
            whose id is not in the map is recorded in `missing_reports`.
        questions: Override the dataset (useful for partial regression runs).
            Defaults to `load_dataset()`.
        max_concurrency: Cap on concurrent judge calls — judges are LLM calls
            and the leaderboard rate-limits at ~5 rps per API key.
    """
    qs = questions if questions is not None else load_dataset()

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _run(q: BenchmarkQuestion) -> JudgeVerdict | None:
        ws = workspace_map.get(q.id)
        if ws is None:
            return None
        async with sem:
            return await grade_workspace(ws, q)

    results = await asyncio.gather(*[_run(q) for q in qs])
    card = Scorecard()
    for q, r in zip(qs, results, strict=True):
        if r is None:
            card.missing_reports.append(q.id)
        else:
            card.verdicts.append(r)
    return card
