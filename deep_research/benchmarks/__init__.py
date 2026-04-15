"""Benchmark-as-regression-test harness for the deep-research pipeline.

Inspired by local-deep-research's `benchmarks/graders.py:163-180` and
MiroThinker's `benchmarks/evaluators/eval_utils.py`: keep a curated list of
research questions whose expected key claims are known, then grade each
pipeline run against that list with a deterministic LLM judge so prompt
changes can be evaluated quantitatively instead of eyeballed.

Two run modes:

1. **Offline** (`runner.grade_workspace`): points the judge at an already-
   written `final-report.md` plus its `claim-ledger.md`. Zero pipeline cost;
   use this to grade every past workspace as regression evidence.

2. **Online** (not implemented here): would invoke the full
   `build_deep_research()` graph end-to-end for each question. Deferred to a
   follow-up because it costs ~$0.50-$1 per question and needs CI bandwidth.

The judge itself is pinned to a single reasoning-capable model (Claude Sonnet
4.6 by default) at `temperature=0` with a strict JSON schema so scores are
reproducible within the same model version.
"""

from deep_research.benchmarks.dataset import (
    BENCHMARK_DATASET_PATH,
    BenchmarkQuestion,
    load_dataset,
)
from deep_research.benchmarks.judge import (
    JudgeVerdict,
    grade_report,
)
from deep_research.benchmarks.runner import (
    Scorecard,
    grade_workspace,
)

__all__ = [
    "BENCHMARK_DATASET_PATH",
    "BenchmarkQuestion",
    "load_dataset",
    "JudgeVerdict",
    "grade_report",
    "Scorecard",
    "grade_workspace",
]
