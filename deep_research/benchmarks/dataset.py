"""Benchmark dataset schema and loader.

Each question is a **closed-book, grounded QA** task: the question has one or
more key claims whose presence in the final report is what the pipeline is
being scored on. Judges do NOT penalise surrounding prose or differences in
phrasing — only whether every `must_include` key claim is substantiated.

We keep the initial dataset small (10-15 questions spanning research types
the pipeline has historically struggled with) so running the benchmark is
cheap enough to gate PRs. Extend it as new failure modes are observed.

Data lives in `dataset.yaml`; Python loader validates and returns typed
`BenchmarkQuestion` objects so downstream judge/runner code can stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

BENCHMARK_DATASET_PATH = Path(__file__).parent / "dataset.yaml"

# Difficulty levels inform future weighting in aggregate scores; for now the
# runner treats them as metadata only.
_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One row of the benchmark dataset.

    Attributes:
        id: Stable identifier (e.g. "BQ001"). Used to match graded results
            across runs so score diffs are aligned.
        question: The research topic exactly as it would be fed to
            `build_deep_research()`.
        must_include: Key factual claims the final report must contain. Each
            entry is a short phrase (< 20 words) describing the claim; the
            judge checks semantic presence, not verbatim match.
        must_not_include: Claims or patterns that must NOT appear (e.g.
            hallucinated URLs, wrong numbers). Optional; defaults to empty.
        category: Tag for slicing scorecards (e.g. "academic", "comparison",
            "how-to"). Does not affect grading.
        difficulty: One of "easy", "medium", "hard". Metadata only.
    """
    id: str
    question: str
    must_include: list[str]
    must_not_include: list[str]
    category: str
    difficulty: Literal["easy", "medium", "hard"]


def load_dataset(path: Path | str | None = None) -> list[BenchmarkQuestion]:
    """Load and validate the benchmark dataset.

    Raises:
        FileNotFoundError: when the YAML file is missing.
        ValueError: when a row fails schema validation (missing required
            field, unknown difficulty, duplicate id).
    """
    source = Path(path) if path else BENCHMARK_DATASET_PATH
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"benchmark dataset root must be a list, got {type(raw).__name__}")

    seen_ids: set[str] = set()
    out: list[BenchmarkQuestion] = []
    for i, row in enumerate(raw):
        q = _validate_row(row, i)
        if q.id in seen_ids:
            raise ValueError(f"duplicate benchmark id: {q.id}")
        seen_ids.add(q.id)
        out.append(q)
    return out


def _validate_row(row: Any, idx: int) -> BenchmarkQuestion:
    if not isinstance(row, dict):
        raise ValueError(f"benchmark row {idx}: expected dict, got {type(row).__name__}")

    def _required(key: str) -> Any:
        if key not in row:
            raise ValueError(f"benchmark row {idx}: missing required field '{key}'")
        return row[key]

    qid = str(_required("id"))
    question = str(_required("question")).strip()
    if not question:
        raise ValueError(f"benchmark row {idx}: 'question' cannot be empty")

    must_include = row.get("must_include") or []
    if not isinstance(must_include, list) or not all(isinstance(x, str) for x in must_include):
        raise ValueError(f"benchmark row {idx}: 'must_include' must be list[str]")
    if not must_include:
        raise ValueError(f"benchmark row {idx}: 'must_include' cannot be empty (grading needs anchors)")

    must_not_include = row.get("must_not_include") or []
    if not isinstance(must_not_include, list) or not all(isinstance(x, str) for x in must_not_include):
        raise ValueError(f"benchmark row {idx}: 'must_not_include' must be list[str]")

    category = str(row.get("category") or "general")
    difficulty = row.get("difficulty") or "medium"
    if difficulty not in _DIFFICULTIES:
        raise ValueError(
            f"benchmark row {idx}: difficulty must be one of {_DIFFICULTIES}, got {difficulty!r}"
        )

    return BenchmarkQuestion(
        id=qid,
        question=question,
        must_include=list(must_include),
        must_not_include=list(must_not_include),
        category=category,
        difficulty=difficulty,
    )
