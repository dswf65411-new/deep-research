"""CLI wrappers for grounding verification tools (Bedrock, MiniCheck, NeMo)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from deep_research.config import GROUNDING_PYTHON, GROUNDING_SCRIPTS_DIR, MINICHECK_PYTHON

PY = GROUNDING_PYTHON
SCRIPTS = str(GROUNDING_SCRIPTS_DIR)
MINICHECK_PY = MINICHECK_PYTHON


@dataclass
class GroundingResult:
    claim: str
    score: float
    verdict: str  # GROUNDED / NOT_GROUNDED
    tool_used: str


def _run_cli(cmd: list[str], input_json: dict, timeout: int = 60) -> dict | None:
    """Run a CLI tool with JSON stdin, return parsed JSON stdout or None."""
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(input_json),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


def bedrock_grounding(claims: list[str], sources: list[str], threshold: float = 0.7) -> dict | None:
    """Call Bedrock Grounding Check CLI."""
    return _run_cli(
        [PY, f"{SCRIPTS}/bedrock-guardrails.py", "--cli"],
        {"claims": claims, "sources": sources, "threshold": threshold},
    )


def minicheck_grounding(claims: list[str], sources: list[str]) -> dict | None:
    """Call MiniCheck CLI (fallback)."""
    return _run_cli(
        [MINICHECK_PY, f"{SCRIPTS}/minicheck.py", "--cli"],
        {"claims": claims, "sources": sources},
        timeout=120,
    )


def nemo_grounding(claims: list[str], sources: list[str], threshold: float = 0.7) -> dict | None:
    """Call NeMo Grounding Check CLI (second fallback)."""
    return _run_cli(
        [PY, f"{SCRIPTS}/nemo-guardrails.py", "--cli"],
        {"claims": claims, "sources": sources, "threshold": threshold},
    )


def url_health_check(urls: list[str], timeout: int = 15) -> dict | None:
    """Call urlhealth CLI."""
    return _run_cli(
        [PY, f"{SCRIPTS}/urlhealth.py", "--cli"],
        {"urls": urls, "timeout": timeout},
    )


def check_grounding_availability() -> tuple[str, str]:
    """Test which grounding tool is available.

    Returns:
        (tool_name, error_msg) — tool_name is "bedrock"/"minicheck"/"nemo"
        or ("none", combined_error) if all fail.
    """
    test_input = {
        "claims": ["The sky is blue."],
        "sources": ["The sky is blue during a clear day."],
    }
    errors = []

    result = bedrock_grounding(**test_input)
    if result and "results" in result:
        return "bedrock", ""

    errors.append("Bedrock: unavailable")

    result = minicheck_grounding(**test_input)
    if result and "results" in result:
        return "minicheck", ""

    errors.append("MiniCheck: unavailable")

    result = nemo_grounding(**test_input)
    if result and "results" in result:
        return "nemo", ""

    errors.append("NeMo: unavailable")

    return "none", "; ".join(errors)


def ground_claims(
    claims: list[str],
    sources: list[str],
    preferred_tool: str = "bedrock",
) -> list[GroundingResult]:
    """Ground a batch of claims using the preferred tool with fallback chain.

    Returns a list of GroundingResult, one per claim.
    """
    tool_chain = {
        "bedrock": [bedrock_grounding, minicheck_grounding, nemo_grounding],
        "minicheck": [minicheck_grounding, nemo_grounding, bedrock_grounding],
        "nemo": [nemo_grounding, bedrock_grounding, minicheck_grounding],
    }
    tool_names = {
        bedrock_grounding: "bedrock",
        minicheck_grounding: "minicheck",
        nemo_grounding: "nemo",
    }

    chain = tool_chain.get(preferred_tool, tool_chain["bedrock"])

    for func in chain:
        result = func(claims, sources)
        if result and "results" in result:
            tool_name = tool_names[func]
            return [
                GroundingResult(
                    claim=r.get("claim", claims[i] if i < len(claims) else ""),
                    score=r.get("grounding_score", r.get("confidence", 0.0)),
                    verdict=r.get("verdict", "GROUNDED" if r.get("supported") else "NOT_GROUNDED"),
                    tool_used=tool_name,
                )
                for i, r in enumerate(result["results"])
            ]

    # All tools failed — return empty scores
    return [
        GroundingResult(claim=c, score=0.0, verdict="UNAVAILABLE", tool_used="none")
        for c in claims
    ]
