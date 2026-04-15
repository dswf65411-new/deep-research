"""LLM judge that scores a single final report against a benchmark question.

Design notes:
- Judge model is pinned (Claude Sonnet 4.6) so score drift caused by the judge
  itself is eliminated. `temperature=0` + explicit JSON schema make outputs
  reproducible within a model version.
- The judge sees only the final report + must_include/must_not_include lists;
  it does NOT see the claim-ledger. This keeps the grade aligned with what an
  end-user would actually read.
- Verdict is a structured dataclass so downstream aggregation (per-category
  scores, regression diffs) is type-safe without JSON re-parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.benchmarks.dataset import BenchmarkQuestion
from deep_research.llm import safe_ainvoke_chain

logger = logging.getLogger(__name__)


VerdictStatus = Literal["PRESENT", "PARTIAL", "MISSING"]


@dataclass
class ClaimGrade:
    """Judge's verdict on a single must_include / must_not_include claim."""
    claim: str
    status: VerdictStatus
    justification: str


@dataclass
class JudgeVerdict:
    """Complete verdict for a single benchmark question."""
    question_id: str
    must_include_grades: list[ClaimGrade] = field(default_factory=list)
    must_not_include_violations: list[ClaimGrade] = field(default_factory=list)
    overall_score: float = 0.0  # 0.0-1.0; computed from grades + violations
    judge_notes: str = ""

    @property
    def passed(self) -> bool:
        """A question passes when all must_include are PRESENT or PARTIAL and
        no must_not_include claims are PRESENT."""
        any_missing = any(g.status == "MISSING" for g in self.must_include_grades)
        any_violations = any(g.status == "PRESENT" for g in self.must_not_include_violations)
        return not any_missing and not any_violations


_JUDGE_SYSTEM = """You are a strict, deterministic LLM judge grading a research-report draft against a checklist.

Your single job: for each claim on the checklist, decide whether the report contains that claim.

## Rules
1. Grade by MEANING, not exact wording. "Whisper achieves 8.3% WER on Mandarin" matches "around 8% Chinese word error rate".
2. A PARTIAL grade means the report mentions the topic but misses a key detail the checklist demands (e.g. the WER number is named without the baseline comparison).
3. A MISSING grade means the report does not address the checklist item at all.
4. must_not_include items are graded in the OPPOSITE direction: PRESENT = violation, MISSING = clean.
5. Every grade MUST include a ≤30-word justification citing a short phrase or sentence from the report as evidence. If the evidence is not present, say so.

## Output — STRICT JSON, no markdown fences, no prose outside the JSON:

{
  "question_id": "<the id given>",
  "must_include_grades": [
    {"claim": "<checklist entry>", "status": "PRESENT" | "PARTIAL" | "MISSING", "justification": "<≤30 words>"}
  ],
  "must_not_include_violations": [
    {"claim": "<checklist entry>", "status": "PRESENT" | "PARTIAL" | "MISSING", "justification": "<≤30 words>"}
  ],
  "overall_score": <float in [0.0, 1.0]>,
  "judge_notes": "<optional overall comment, ≤50 words>"
}

Scoring formula (compute this yourself, don't leave blank):
- Start at 1.0
- Subtract 1.0 / N_must_include for every MISSING must_include, 0.5 / N_must_include for every PARTIAL
- Subtract 0.5 for every PRESENT must_not_include violation
- Clamp to [0.0, 1.0]
"""


async def grade_report(
    question: BenchmarkQuestion,
    report_text: str,
) -> JudgeVerdict:
    """Grade one benchmark question against one final report.

    Args:
        question: The benchmark row being graded.
        report_text: Full text of the pipeline's `final-report.md`. The judge
            handles the truncation decision if this is larger than its
            context window (we rely on Claude Sonnet's 1M-token window).

    Returns:
        A fully-populated JudgeVerdict. On LLM failure or malformed JSON,
        returns a MISSING-all verdict with the failure reason in judge_notes
        so the scorecard can surface it instead of silently dropping the row.
    """
    must_include_block = "\n".join(f"- {c}" for c in question.must_include)
    must_not_include_block = (
        "\n".join(f"- {c}" for c in question.must_not_include)
        or "(none)"
    )

    human_content = f"""## Question ID
{question.id}

## Research question
{question.question}

## must_include checklist ({len(question.must_include)} items)
{must_include_block}

## must_not_include checklist
{must_not_include_block}

## Final report to grade
{report_text or '(report is empty)'}
"""

    try:
        response = await safe_ainvoke_chain(
            role="writer",  # Claude leads the "writer" chain; good fit for critical judging
            messages=[
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=human_content),
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        content = getattr(response, "content", "") or ""
        return _parse_verdict(content, question)
    except Exception as e:
        logger.warning("judge: LLM call failed for %s (%s)", question.id, e)
        return _verdict_all_missing(question, f"judge LLM call failed: {e!r}")


def _parse_verdict(text: str, question: BenchmarkQuestion) -> JudgeVerdict:
    """Tolerant parse: strip markdown fences, find first balanced JSON object.

    Defaults to an all-MISSING verdict when parsing fails so the scorecard
    visibly reflects the judge failure rather than silently recording a pass.
    """
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.MULTILINE).strip()
    m = re.search(r"\{[\s\S]*\}", stripped)
    if not m:
        return _verdict_all_missing(question, "judge returned no JSON")
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return _verdict_all_missing(question, "judge JSON parse failed")

    def _grades(key: str) -> list[ClaimGrade]:
        items = raw.get(key) or []
        out: list[ClaimGrade] = []
        if isinstance(items, list):
            for x in items:
                if isinstance(x, dict):
                    status = x.get("status", "MISSING")
                    if status not in ("PRESENT", "PARTIAL", "MISSING"):
                        status = "MISSING"
                    out.append(ClaimGrade(
                        claim=str(x.get("claim", "")),
                        status=status,
                        justification=str(x.get("justification", "")),
                    ))
        return out

    try:
        score = float(raw.get("overall_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    return JudgeVerdict(
        question_id=str(raw.get("question_id", question.id)),
        must_include_grades=_grades("must_include_grades"),
        must_not_include_violations=_grades("must_not_include_violations"),
        overall_score=score,
        judge_notes=str(raw.get("judge_notes", "")),
    )


def _verdict_all_missing(question: BenchmarkQuestion, reason: str) -> JudgeVerdict:
    """Construct a verdict that records a judge failure clearly."""
    return JudgeVerdict(
        question_id=question.id,
        must_include_grades=[
            ClaimGrade(claim=c, status="MISSING", justification=reason)
            for c in question.must_include
        ],
        must_not_include_violations=[
            ClaimGrade(claim=c, status="MISSING", justification=reason)
            for c in question.must_not_include
        ],
        overall_score=0.0,
        judge_notes=reason,
    )
