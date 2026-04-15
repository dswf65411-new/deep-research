"""Whisper P3-4 — pipeline self-evaluation + follow-up suggestions.

Rolls up per-subquestion outcomes (covered / thin / blocked) and optionally
asks an LLM to draft up to five follow-up directions. Designed so phase3 can
append a short ``## 本次研究自評 + 建議後續`` block to ``final-report.md``.

Scoring is deterministic and LLM-free. Only the follow-up generation talks
to an LLM, and the caller owns the invocation (we keep ``async`` surface
area to a bare minimum so tests can inject a fake).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal

logger = logging.getLogger(__name__)


Status = Literal["covered", "thin", "blocked"]


@dataclass
class SqOutcome:
    sq_id: str
    status: Status
    approved_count: int = 0
    rejected_count: int = 0
    top_sources: list[str] = field(default_factory=list)
    notes: str = ""


# A sub-question with fewer than this many approved claims is considered
# "thin" — enough to mention in the self-eval, not enough to celebrate.
_THIN_THRESHOLD = 3
_BLOCKED_KEYWORDS = re.compile(r"\[(BLOCKER|CRITICAL)", re.IGNORECASE)


def score_subquestions(
    claims: Iterable,
    blockers: Iterable[str] | None = None,
    coverage_sqs: Iterable[str] | None = None,
) -> list[SqOutcome]:
    """Deterministically score each sub-question.

    Parameters
    ----------
    claims:
        Iterable of ``Claim`` models or plain dicts. We read
        ``subquestion``, ``status``, and ``source_ids``.
    blockers:
        Blocker strings from phase2 / phase3. Any blocker line mentioning
        a sub-question id (``Q3``…) flips that sub-question to ``blocked``.
    coverage_sqs:
        Optional list of the full set of planned sub-questions; any SQ in
        here that has no approved claim is reported as ``thin`` with
        approved_count=0 (rather than being silently dropped).
    """
    approved: dict[str, list[dict]] = {}
    rejected_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}

    for c in claims or []:
        sq = (_get(c, "subquestion") or "").strip()
        status = (_get(c, "status") or "").strip().lower()
        if not sq:
            continue
        if status == "approved":
            approved.setdefault(sq, []).append(c)
            for sid in (_get(c, "source_ids") or []):
                source_counts.setdefault(sq, Counter())[sid] += 1
        elif status == "rejected":
            rejected_counts[sq] += 1

    blocked_sqs: set[str] = set()
    for line in (blockers or []):
        for sq in re.findall(r"Q\d+", line or ""):
            if _BLOCKED_KEYWORDS.search(line or ""):
                blocked_sqs.add(sq)

    all_sqs: set[str] = set()
    all_sqs.update(approved.keys())
    all_sqs.update(rejected_counts.keys())
    all_sqs.update(blocked_sqs)
    all_sqs.update(sq for sq in (coverage_sqs or []) if sq)

    outcomes: list[SqOutcome] = []
    for sq in sorted(all_sqs, key=_sq_sort_key):
        n_app = len(approved.get(sq, []))
        n_rej = rejected_counts.get(sq, 0)
        top = [sid for sid, _ in source_counts.get(sq, Counter()).most_common(3)]

        if sq in blocked_sqs:
            status: Status = "blocked"
            notes = "Blocker flagged in gap-log / phase logs"
        elif n_app == 0:
            status = "thin"
            notes = "No approved claims produced"
        elif n_app < _THIN_THRESHOLD:
            status = "thin"
            notes = f"Only {n_app} approved claim(s), below the {_THIN_THRESHOLD}-claim threshold"
        else:
            status = "covered"
            notes = f"{n_app} approved claims across {len(top)} distinct top sources"

        outcomes.append(SqOutcome(
            sq_id=sq, status=status,
            approved_count=n_app, rejected_count=n_rej,
            top_sources=top, notes=notes,
        ))

    return outcomes


def format_self_eval_section(
    outcomes: list[SqOutcome],
    follow_ups: list[str] | None = None,
) -> str:
    """Render the ``本次研究自評 + 建議後續`` markdown block."""
    lines: list[str] = []
    lines.append("## 本次研究自評 + 建議後續")
    lines.append("")
    if not outcomes:
        lines.append("(no sub-questions scored — phase 1 produced no claims.)")
    else:
        covered = [o for o in outcomes if o.status == "covered"]
        thin = [o for o in outcomes if o.status == "thin"]
        blocked = [o for o in outcomes if o.status == "blocked"]
        lines.append(
            f"Coverage tally: covered={len(covered)}, thin={len(thin)}, "
            f"blocked={len(blocked)}."
        )
        lines.append("")
        lines.append("| SQ | status | approved | rejected | notes |")
        lines.append("|---|---|---|---|---|")
        for o in outcomes:
            lines.append(
                f"| {o.sq_id} | {o.status} | {o.approved_count} | "
                f"{o.rejected_count} | {o.notes} |"
            )

    if follow_ups:
        lines.append("")
        lines.append("### 建議後續研究方向 (follow-ups)")
        for i, fu in enumerate(follow_ups[:5], start=1):
            lines.append(f"{i}. {fu}")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Follow-up generation (LLM)
# ---------------------------------------------------------------------------


_FOLLOW_UP_SYSTEM = (
    "You are the self-evaluation module of a deep-research pipeline. "
    "Given the research brief and a structured summary of how each "
    "sub-question fared (covered / thin / blocked), suggest up to FIVE "
    "concrete follow-up research directions. Each item must: "
    "(1) reference a specific SQ or a gap that the current run left open; "
    "(2) be phrased as an actionable next step (not a question, not a slogan); "
    "(3) be short — one sentence, under 40 words. "
    "Reply with a JSON object: "
    "{\"follow_ups\": [\"...\", \"...\"]}"
)


async def generate_follow_ups(
    brief_text: str,
    outcomes: list[SqOutcome],
    invoker,  # callable: async def invoker(messages) -> response with .content
) -> list[str]:
    """Ask an LLM for up to 5 follow-up directions. LLM errors → ``[]``.

    ``invoker`` is injected so tests (and the caller) don't have to mock the
    whole ``deep_research.llm`` module. In production, phase3 passes in a
    small wrapper that calls ``safe_ainvoke`` on the ``fast`` role.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = {
        "brief": brief_text[:4000],
        "subquestions": [
            {
                "sq_id": o.sq_id,
                "status": o.status,
                "approved_count": o.approved_count,
                "rejected_count": o.rejected_count,
                "notes": o.notes,
            }
            for o in outcomes
        ],
    }
    human = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    system = SystemMessage(content=_FOLLOW_UP_SYSTEM)

    try:
        response = await invoker([system, human])
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning("self-eval follow-up invocation failed: %r", exc)
        return []

    text = (getattr(response, "content", "") or "").strip()
    if not text:
        return []
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\"follow_ups\"[^{}]*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    items = parsed.get("follow_ups") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        if len(out) >= 5:
            break
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _sq_sort_key(sq: str) -> tuple[int, str]:
    """Sort ``Q2`` before ``Q10`` numerically."""
    m = re.match(r"Q(\d+)", sq, re.IGNORECASE)
    if m:
        return (int(m.group(1)), sq)
    return (10**9, sq)
