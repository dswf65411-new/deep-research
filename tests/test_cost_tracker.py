"""Tests for Whisper plan P2-2 — run-wide cost / progress tracker.

These tests pin the counter behaviour (calls / tokens / USD) and the
``extract_usage`` helper that pulls metadata off LangChain responses.
No actual LLM is called — we stub the response object.
"""

from __future__ import annotations

from types import SimpleNamespace

import deep_research.harness.cost_tracker as ct


def setup_function(_func):
    ct.reset()


# ---------------------------------------------------------------------------
# Counter arithmetic
# ---------------------------------------------------------------------------


def test_initial_snapshot_is_zero():
    snap = ct.snapshot_dict()
    assert snap["llm_calls"] == 0
    assert snap["input_tokens"] == 0
    assert snap["output_tokens"] == 0
    assert snap["est_cost_usd"] == 0.0
    assert snap["searches"] == 0
    assert snap["sources"] == 0
    assert snap["per_model"] == {}


def test_record_llm_call_sonnet_pricing():
    ct.record_llm_call("claude-sonnet-4-6", 1_000_000, 500_000)
    snap = ct.snapshot_dict()
    assert snap["llm_calls"] == 1
    assert snap["input_tokens"] == 1_000_000
    assert snap["output_tokens"] == 500_000
    # sonnet: $3/M input + $15/M output = 3 + 7.5 = 10.5
    assert snap["est_cost_usd"] == 10.5
    assert snap["per_model"]["claude-sonnet-4-6"] == {
        "calls": 1,
        "input_tokens": 1_000_000,
        "output_tokens": 500_000,
    }


def test_record_llm_call_haiku_pricing():
    ct.record_llm_call("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    snap = ct.snapshot_dict()
    # haiku: $0.80/M + $4/M = $4.80
    assert snap["est_cost_usd"] == 4.8


def test_unknown_model_falls_back_to_sonnet_price():
    """Prevents silent under-reporting when a new model id lands in the
    repo before the price table is updated."""
    ct.record_llm_call("claude-next-gen-9", 1_000_000, 1_000_000)
    snap = ct.snapshot_dict()
    # Fallback is (3, 15) → $18 total.
    assert snap["est_cost_usd"] == 18.0


def test_multiple_calls_accumulate():
    ct.record_llm_call("claude-sonnet-4-6", 100, 200)
    ct.record_llm_call("claude-sonnet-4-6", 300, 400)
    snap = ct.snapshot_dict()
    assert snap["llm_calls"] == 2
    assert snap["input_tokens"] == 400
    assert snap["output_tokens"] == 600


def test_search_and_source_counters():
    ct.record_search()
    ct.record_search(4)
    ct.record_source(10)
    snap = ct.snapshot_dict()
    assert snap["searches"] == 5
    assert snap["sources"] == 10


def test_negative_counts_clamped_to_zero():
    ct.record_llm_call("claude-sonnet-4-6", -100, -200)
    snap = ct.snapshot_dict()
    assert snap["input_tokens"] == 0
    assert snap["output_tokens"] == 0
    assert snap["est_cost_usd"] == 0.0


def test_set_phase_reflected_in_snapshot():
    ct.set_phase("phase1a")
    assert ct.snapshot_dict()["phase"] == "phase1a"


def test_reset_clears_everything():
    ct.record_llm_call("claude-sonnet-4-6", 100, 200)
    ct.record_search(3)
    ct.set_phase("phase2")
    ct.reset()
    snap = ct.snapshot_dict()
    assert snap["llm_calls"] == 0
    assert snap["searches"] == 0
    assert snap["phase"] == "init"


def test_snapshot_is_a_copy():
    """Mutating the returned snapshot must not affect the live tracker."""
    ct.record_llm_call("claude-sonnet-4-6", 100, 200)
    snap = ct.snapshot_dict()
    snap["llm_calls"] = 999
    assert ct.snapshot_dict()["llm_calls"] == 1


# ---------------------------------------------------------------------------
# extract_usage — LangChain response shape compatibility
# ---------------------------------------------------------------------------


def test_extract_usage_from_usage_metadata():
    resp = SimpleNamespace(
        usage_metadata={"input_tokens": 1000, "output_tokens": 200},
        response_metadata={"model": "claude-sonnet-4-6"},
    )
    out = ct.extract_usage(resp)
    assert out == ("claude-sonnet-4-6", 1000, 200)


def test_extract_usage_from_response_metadata_usage():
    resp = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "usage": {"prompt_tokens": 500, "completion_tokens": 100},
            "model_name": "claude-haiku-4-5-20251001",
        },
    )
    out = ct.extract_usage(resp)
    assert out == ("claude-haiku-4-5-20251001", 500, 100)


def test_extract_usage_returns_none_when_no_metadata():
    resp = SimpleNamespace(usage_metadata=None, response_metadata={})
    assert ct.extract_usage(resp) is None


def test_extract_usage_handles_missing_model_name():
    """If the metadata has no model, we still return the token counts with
    an empty model id so pricing can fall back."""
    resp = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
        response_metadata={},
    )
    out = ct.extract_usage(resp)
    assert out == ("", 100, 50)


def test_extract_usage_handles_none_response():
    assert ct.extract_usage(None) is None
