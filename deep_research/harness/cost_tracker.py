"""Token / cost / progress tracking for a single research run.

Whisper plan P2-2 — ``--json`` mode was silent for 20 minutes on the failed
run; users had no way to tell whether the pipeline was making progress or
how much the LLM calls were costing. This module is a cheap, side-effect-
free counter that:

- ``record_llm_call(model, input_tokens, output_tokens)`` bumps call count,
  token totals, and a rough USD estimate.
- ``record_search()`` / ``record_source()`` / ``record_phase(name)`` bump
  the other counters.
- ``snapshot()`` returns a dict suitable for JSON emission.

No threading — the pipeline is single-async-loop per run, and we use a
module-level singleton accessed through a small API so downstream code
doesn't need to thread the tracker through every call site.

Rate estimates are rough. They're meant to let a caller answer "is this
about to blow through my budget?", not to bill customers. Source:
Anthropic's public pricing page as of 2026-04 for Claude Sonnet 4.6 and
Haiku 4.5.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# USD per million tokens (input, output). Only the models the pipeline
# actually uses are listed; an unknown model falls back to a conservative
# sonnet-grade estimate so we never silently under-report.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # Historical aliases we might still see in usage_metadata.
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (0.80, 4.00),
}

_FALLBACK_PRICE = (3.0, 15.0)


@dataclass
class CostSnapshot:
    phase: str = "init"
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    searches: int = 0
    sources: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 4),
            "searches": self.searches,
            "sources": self.sources,
            "per_model": self.per_model,
        }


class _Tracker:
    """Module-level singleton. Thread-safe because a pipeline may spawn
    concurrent LLM calls via ``asyncio.gather``; ``threading.Lock`` is the
    simplest way to keep counters consistent across those coroutines without
    forcing every caller to await."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = CostSnapshot()

    def reset(self) -> None:
        with self._lock:
            self._snap = CostSnapshot()

    def set_phase(self, name: str) -> None:
        with self._lock:
            self._snap.phase = name

    def record_llm_call(
        self, model: str, input_tokens: int, output_tokens: int,
    ) -> None:
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
        price_in, price_out = _MODEL_PRICES.get(model, _FALLBACK_PRICE)
        cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
        with self._lock:
            self._snap.llm_calls += 1
            self._snap.input_tokens += input_tokens
            self._snap.output_tokens += output_tokens
            self._snap.est_cost_usd += cost
            bucket = self._snap.per_model.setdefault(
                model, {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens

    def record_search(self, count: int = 1) -> None:
        with self._lock:
            self._snap.searches += max(0, int(count))

    def record_source(self, count: int = 1) -> None:
        with self._lock:
            self._snap.sources += max(0, int(count))

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            # Return a shallow copy so callers can't mutate the live state.
            return CostSnapshot(
                phase=self._snap.phase,
                llm_calls=self._snap.llm_calls,
                input_tokens=self._snap.input_tokens,
                output_tokens=self._snap.output_tokens,
                est_cost_usd=self._snap.est_cost_usd,
                searches=self._snap.searches,
                sources=self._snap.sources,
                per_model={k: dict(v) for k, v in self._snap.per_model.items()},
            )


_GLOBAL = _Tracker()


def reset() -> None:
    _GLOBAL.reset()


def set_phase(name: str) -> None:
    _GLOBAL.set_phase(name)


def record_llm_call(model: str, input_tokens: int, output_tokens: int) -> None:
    _GLOBAL.record_llm_call(model, input_tokens, output_tokens)


def record_search(count: int = 1) -> None:
    _GLOBAL.record_search(count)


def record_source(count: int = 1) -> None:
    _GLOBAL.record_source(count)


def snapshot() -> CostSnapshot:
    return _GLOBAL.snapshot()


def snapshot_dict() -> dict[str, Any]:
    return snapshot().to_dict()


# ---------------------------------------------------------------------------
# Usage-metadata extraction
# ---------------------------------------------------------------------------


def extract_usage(response: Any) -> tuple[str, int, int] | None:
    """Pull ``(model, input_tokens, output_tokens)`` from a LangChain response.

    LangChain exposes token counts via ``response.usage_metadata`` (a dict
    with ``input_tokens`` / ``output_tokens`` / ``total_tokens``) or the
    older ``response_metadata["usage"]`` path, depending on provider.
    Returns ``None`` if no usage data is available — callers treat that as
    "don't count this call".
    """
    if response is None:
        return None

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("input_tokens") is not None:
        model = _extract_model_name(response)
        return model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    rm = getattr(response, "response_metadata", None)
    if isinstance(rm, dict):
        usage = rm.get("usage") or rm.get("token_usage")
        if isinstance(usage, dict):
            input_tokens = int(
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )
            output_tokens = int(
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )
            if input_tokens or output_tokens:
                model = _extract_model_name(response)
                return model, input_tokens, output_tokens

    return None


def _extract_model_name(response: Any) -> str:
    """Best-effort model-id recovery from a LangChain response object."""
    rm = getattr(response, "response_metadata", None) or {}
    if isinstance(rm, dict):
        model = rm.get("model") or rm.get("model_name") or rm.get("model_id")
        if model:
            return str(model)
    # ChatAnthropic surfaces model on the chat object itself, not the response.
    # Falling back to an empty string is fine — the pricing helper uses the
    # sonnet-grade default when the model is unknown.
    return ""
