"""Tests for Whisper P2-5 — runtime limits (max-time / max-cost / abort)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.harness import runtime_limits as rl
from deep_research.harness.runtime_limits import LimitExceeded


def setup_function(_func):
    rl.reset()


# ---------------------------------------------------------------------------
# No limits → never raises
# ---------------------------------------------------------------------------


def test_no_limits_no_raise():
    rl.start()
    # Pass various cost values; nothing should raise.
    rl.check(cost_usd=0.0)
    rl.check(cost_usd=100.0)
    rl.check(cost_usd=9999.0)
    assert rl.tripped() is None


def test_cost_check_none_ignored():
    rl.set_max_cost_usd(1.0)
    rl.start()
    # cost_usd=None → cost limit can't fire
    rl.check(cost_usd=None)
    assert rl.tripped() is None


# ---------------------------------------------------------------------------
# Time limit
# ---------------------------------------------------------------------------


def test_time_limit_not_exceeded():
    rl.set_max_time_minutes(10.0)
    rl.start(now=1000.0)
    rl.check(now=1000.0 + 59.0)  # 59s elapsed, cap 10min
    assert rl.tripped() is None


def test_time_limit_exceeded_raises():
    rl.set_max_time_minutes(1.0)
    rl.start(now=1000.0)
    with pytest.raises(LimitExceeded, match="--max-time exceeded"):
        rl.check(now=1000.0 + 61.0)  # 61s > 60s cap
    assert rl.tripped() is not None


def test_time_limit_stays_tripped():
    """Once tripped, subsequent checks keep raising (no re-entry)."""
    rl.set_max_time_minutes(1.0)
    rl.start(now=1000.0)
    with pytest.raises(LimitExceeded):
        rl.check(now=1000.0 + 120.0)
    # Even without new time argument, we stay tripped.
    with pytest.raises(LimitExceeded):
        rl.check()


def test_zero_or_negative_max_time_treated_as_unset():
    rl.set_max_time_minutes(0)
    rl.start(now=1000.0)
    rl.check(now=1000.0 + 999.0)
    rl.set_max_time_minutes(-5)
    rl.check(now=1000.0 + 999.0)


def test_time_limit_without_start_is_noop():
    """Forgetting to call start() means the time limit simply can't fire."""
    rl.set_max_time_minutes(1.0)
    # No rl.start() call
    rl.check(now=1_000_000.0)


# ---------------------------------------------------------------------------
# Cost limit
# ---------------------------------------------------------------------------


def test_cost_limit_not_exceeded():
    rl.set_max_cost_usd(5.0)
    rl.start()
    rl.check(cost_usd=1.23)
    assert rl.tripped() is None


def test_cost_limit_exceeded_raises():
    rl.set_max_cost_usd(5.0)
    rl.start()
    with pytest.raises(LimitExceeded, match=r"--max-cost exceeded"):
        rl.check(cost_usd=5.01)


def test_cost_limit_at_cap_exceeds():
    """'>=' semantics: hitting the cap exactly is already over."""
    rl.set_max_cost_usd(5.0)
    rl.start()
    with pytest.raises(LimitExceeded):
        rl.check(cost_usd=5.0)


def test_zero_or_negative_max_cost_treated_as_unset():
    rl.set_max_cost_usd(0)
    rl.start()
    rl.check(cost_usd=9999.0)
    rl.set_max_cost_usd(-1)
    rl.check(cost_usd=9999.0)


# ---------------------------------------------------------------------------
# Abort marker
# ---------------------------------------------------------------------------


def test_abort_marker_absent_is_ok(tmp_path: Path):
    rl.set_abort_marker(tmp_path / ".abort")
    rl.start()
    rl.check()  # no marker exists, no raise


def test_abort_marker_present_raises(tmp_path: Path):
    marker = tmp_path / ".abort"
    marker.write_text("go home", encoding="utf-8")
    rl.set_abort_marker(marker)
    rl.start()
    with pytest.raises(LimitExceeded, match="aborted via marker file"):
        rl.check()


def test_write_abort_marker_creates_file(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    marker = rl.write_abort_marker(workspace)
    assert marker.exists()
    assert marker.name == ".abort"
    assert "aborted at" in marker.read_text()


def test_write_abort_marker_missing_workspace(tmp_path: Path):
    """Typo path → FileNotFoundError, not silent write to current dir."""
    with pytest.raises(FileNotFoundError):
        rl.write_abort_marker(tmp_path / "does_not_exist")


def test_abort_marker_path_is_stable(tmp_path: Path):
    """Abort-marker helper must match what write_abort_marker writes."""
    ws = tmp_path / "ws"
    ws.mkdir()
    written = rl.write_abort_marker(ws)
    assert rl.abort_marker_path(ws) == written


# ---------------------------------------------------------------------------
# Snapshot + reset
# ---------------------------------------------------------------------------


def test_snapshot_reflects_config(tmp_path: Path):
    rl.set_max_time_minutes(3.5)
    rl.set_max_cost_usd(7.25)
    rl.set_abort_marker(tmp_path / ".abort")
    rl.start(now=500.0)

    snap = rl.snapshot()
    assert snap["max_time_minutes"] == 3.5
    assert snap["max_cost_usd"] == 7.25
    assert snap["abort_marker"].endswith(".abort")
    assert snap["started_at"] == 500.0
    assert snap["tripped_reason"] is None


def test_reset_clears_everything():
    rl.set_max_time_minutes(1.0)
    rl.set_max_cost_usd(1.0)
    rl.start()
    rl.reset()
    snap = rl.snapshot()
    assert snap["max_time_minutes"] is None
    assert snap["max_cost_usd"] is None
    assert snap["abort_marker"] is None
    assert snap["started_at"] is None


def test_reset_after_trip_allows_fresh_start():
    rl.set_max_cost_usd(1.0)
    rl.start()
    with pytest.raises(LimitExceeded):
        rl.check(cost_usd=2.0)
    rl.reset()
    # After reset, we can run cleanly again.
    rl.set_max_cost_usd(100.0)
    rl.start()
    rl.check(cost_usd=2.0)
