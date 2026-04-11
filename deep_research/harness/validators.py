"""Pydantic-level and programmatic validators (iron rules enforcement)."""

from __future__ import annotations

from deep_research.state import Claim, Source


def validate_claims_for_phase2(claims: list[Claim]) -> list[Claim]:
    """Iron rule: only approved claims with quote_ids enter Phase 2."""
    return [
        c for c in claims
        if c.status == "approved" and len(c.quote_ids) > 0
    ]


def validate_numeric_claims(claims: list[Claim]) -> list[str]:
    """Iron rule: every numeric claim must have a number_tag."""
    violations = []
    for c in claims:
        if c.claim_type == "numeric" and c.number_tag is None:
            violations.append(
                f"{c.claim_id}: numeric claim missing number_tag"
            )
    return violations


def validate_traceability_chain(
    statements: list[dict],
    claims: list[Claim],
    sources: list[Source],
) -> list[str]:
    """Iron rule: statement → claim_id → quote_id → source_id chain must be complete."""
    claim_map = {c.claim_id: c for c in claims}
    source_ids = {s.source_id for s in sources}
    broken = []

    for st in statements:
        st_id = st.get("statement_id", "?")
        st_type = st.get("type", "")
        if st_type == "opinion":
            continue

        claim_ids = st.get("claim_ids", [])
        if not claim_ids:
            broken.append(f"{st_id}: no claim_ids")
            continue

        for cid in claim_ids:
            claim = claim_map.get(cid)
            if claim is None:
                broken.append(f"{st_id}: claim {cid} not found in ledger")
                continue
            if claim.status != "approved":
                broken.append(f"{st_id}: claim {cid} status={claim.status}")
                continue
            if not claim.quote_ids:
                broken.append(f"{st_id}: claim {cid} has no quote_ids")
                continue
            for sid in claim.source_ids:
                if sid not in source_ids:
                    broken.append(f"{st_id}: source {sid} not in registry")

    return broken


def filter_attack_agent_tools(tool_names: list[str]) -> list[str]:
    """Iron rule: sub-agents must NOT have search tools."""
    search_prefixes = ("brave_search", "serper_search", "serper_scrape", "web_fetch")
    return [t for t in tool_names if not t.startswith(search_prefixes)]
