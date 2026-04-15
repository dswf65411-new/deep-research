"""Whisper P3-3 — multi-stakeholder advocate/critic collision detector.

In the failed workspace the ``advocate`` and ``critic`` searches produced
claims, but nothing in the pipeline ever put them side-by-side and asked
"do these disagree?". This module surfaces per-subquestion collisions so
phase3 can drop them into ``gap-log.md`` as an *unresolved tensions* section.

Deliberately heuristic: clustering + LLM-driven contradiction reasoning is a
follow-up. For now a collision = "this sub-question has ≥ 1 approved claim
cited from an advocate source AND ≥ 1 from a critic source". That guarantees
the gap log at least *shows the two sides*, instead of silently flattening
them into a one-sided narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CollisionPair:
    subquestion: str
    advocate_claims: list[str] = field(default_factory=list)   # claim_ids
    critic_claims: list[str] = field(default_factory=list)     # claim_ids
    advocate_sources: list[str] = field(default_factory=list)  # source_ids
    critic_sources: list[str] = field(default_factory=list)    # source_ids


def _source_role_index(sources: Iterable) -> dict[str, str]:
    """Map source_id → role ('advocate' | 'critic' | 'perspective' | '')."""
    index: dict[str, str] = {}
    for s in sources or []:
        sid = getattr(s, "source_id", None) or (s.get("source_id") if isinstance(s, dict) else None)
        role = getattr(s, "role", None) or (s.get("role") if isinstance(s, dict) else None)
        if sid:
            index[sid] = (role or "").lower()
    return index


def collect_collisions(
    approved_claims: Iterable,
    sources: Iterable,
) -> list[CollisionPair]:
    """Group approved claims by subquestion and flag those that span both sides.

    ``approved_claims`` elements may be Pydantic ``Claim`` models or plain
    dicts (the pipeline passes both shapes in different places).
    """
    role_by_source = _source_role_index(sources)
    by_sq: dict[str, CollisionPair] = {}

    for c in approved_claims or []:
        cid = _get(c, "claim_id")
        sq = (_get(c, "subquestion") or "").strip()
        sids = _get(c, "source_ids") or []
        if not cid or not sq:
            continue

        roles = {role_by_source.get(sid, "") for sid in sids}
        has_advocate = "advocate" in roles
        has_critic = "critic" in roles

        if not (has_advocate or has_critic):
            continue

        pair = by_sq.setdefault(sq, CollisionPair(subquestion=sq))
        if has_advocate:
            pair.advocate_claims.append(cid)
            for sid in sids:
                if role_by_source.get(sid) == "advocate" and sid not in pair.advocate_sources:
                    pair.advocate_sources.append(sid)
        if has_critic:
            pair.critic_claims.append(cid)
            for sid in sids:
                if role_by_source.get(sid) == "critic" and sid not in pair.critic_sources:
                    pair.critic_sources.append(sid)

    # Only return SQs that actually have both sides.
    return [
        p for p in by_sq.values()
        if p.advocate_claims and p.critic_claims
    ]


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def format_collision_section(pairs: list[CollisionPair]) -> str:
    """Render a gap-log ``## 未解矛盾`` section.

    Returns an empty string when no collisions exist so callers can append
    conditionally.
    """
    if not pairs:
        return ""

    lines: list[str] = []
    lines.append("## 未解矛盾 (Unresolved stakeholder tensions)")
    lines.append("")
    lines.append(
        "Each sub-question below has at least one approved claim grounded in "
        "advocate-framed sources AND at least one in critic-framed sources. "
        "The final report should surface both sides rather than silently "
        "adopting one."
    )
    lines.append("")
    for p in sorted(pairs, key=lambda x: x.subquestion):
        lines.append(f"### {p.subquestion}")
        lines.append(
            f"- Advocate: {len(p.advocate_claims)} claims "
            f"({', '.join(p.advocate_claims)}) via "
            f"{', '.join(p.advocate_sources) or 'n/a'}"
        )
        lines.append(
            f"- Critic: {len(p.critic_claims)} claims "
            f"({', '.join(p.critic_claims)}) via "
            f"{', '.join(p.critic_sources) or 'n/a'}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
