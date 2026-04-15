"""Whisper P2-5 — soft runtime limits (wall-clock + cost) and abort marker.

Used by ``main.py`` to enforce ``--max-time``, ``--max-cost``, and ``--abort``
against long-running research pipelines. Design goals:

* Keep the limit state *outside* the LangGraph state machine so nodes don't
  need to thread it through every hop — a module-level singleton mirrors the
  pattern used by ``cost_tracker``.
* Raise a specific ``LimitExceeded`` exception the CLI can catch for a clean
  "wrote partial report, stopped early" message.
* Fail closed: once a limit fires, subsequent ``check()`` calls keep raising
  so the shutdown path can't re-enter the running graph.

Intentionally NOT a hard timeout — long-running tool calls (WebFetch, LLM)
aren't interrupted mid-flight. We only stop between phases / progress
emissions, which is where partial output can still be salvaged.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path


class LimitExceeded(RuntimeError):
    """Raised when a runtime limit (time / cost / abort) fires."""


@dataclass
class _LimitState:
    max_time_minutes: float | None = None
    max_cost_usd: float | None = None
    abort_marker: Path | None = None
    started_at: float | None = None
    tripped_reason: str | None = None


class _RuntimeLimits:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _LimitState()

    def reset(self) -> None:
        with self._lock:
            self._state = _LimitState()

    def set_max_time_minutes(self, minutes: float | None) -> None:
        with self._lock:
            self._state.max_time_minutes = (
                float(minutes) if minutes is not None and minutes > 0 else None
            )

    def set_max_cost_usd(self, usd: float | None) -> None:
        with self._lock:
            self._state.max_cost_usd = (
                float(usd) if usd is not None and usd > 0 else None
            )

    def set_abort_marker(self, path: Path | str | None) -> None:
        with self._lock:
            self._state.abort_marker = Path(path) if path else None

    def start(self, now: float | None = None) -> None:
        """Record the wall-clock start so ``check()`` can compute elapsed."""
        with self._lock:
            self._state.started_at = time.monotonic() if now is None else float(now)
            self._state.tripped_reason = None

    def tripped(self) -> str | None:
        with self._lock:
            return self._state.tripped_reason

    def check(self, cost_usd: float | None = None, now: float | None = None) -> None:
        """Raise ``LimitExceeded`` if any configured limit has fired.

        * ``cost_usd`` — current accumulated cost (caller passes from
          ``cost_tracker.snapshot_dict()`` so this module stays dependency-free).
        * ``now`` — optional monotonic clock override for tests.
        """
        with self._lock:
            st = self._state
            # Once tripped, stay tripped — avoids re-entering a running graph.
            if st.tripped_reason:
                raise LimitExceeded(st.tripped_reason)

            if st.abort_marker is not None and st.abort_marker.exists():
                reason = f"aborted via marker file: {st.abort_marker}"
                st.tripped_reason = reason
                raise LimitExceeded(reason)

            if st.max_time_minutes is not None and st.started_at is not None:
                current = time.monotonic() if now is None else float(now)
                elapsed_min = (current - st.started_at) / 60.0
                if elapsed_min >= st.max_time_minutes:
                    reason = (
                        f"--max-time exceeded: {elapsed_min:.1f}min elapsed "
                        f">= {st.max_time_minutes:.1f}min cap"
                    )
                    st.tripped_reason = reason
                    raise LimitExceeded(reason)

            if st.max_cost_usd is not None and cost_usd is not None:
                if cost_usd >= st.max_cost_usd:
                    reason = (
                        f"--max-cost exceeded: ${cost_usd:.3f} spent "
                        f">= ${st.max_cost_usd:.3f} cap"
                    )
                    st.tripped_reason = reason
                    raise LimitExceeded(reason)

    def snapshot(self) -> dict:
        """Read-only view for tests / logging."""
        with self._lock:
            st = self._state
            return {
                "max_time_minutes": st.max_time_minutes,
                "max_cost_usd": st.max_cost_usd,
                "abort_marker": str(st.abort_marker) if st.abort_marker else None,
                "started_at": st.started_at,
                "tripped_reason": st.tripped_reason,
            }


_GLOBAL = _RuntimeLimits()


def reset() -> None:
    _GLOBAL.reset()


def set_max_time_minutes(minutes: float | None) -> None:
    _GLOBAL.set_max_time_minutes(minutes)


def set_max_cost_usd(usd: float | None) -> None:
    _GLOBAL.set_max_cost_usd(usd)


def set_abort_marker(path: Path | str | None) -> None:
    _GLOBAL.set_abort_marker(path)


def start(now: float | None = None) -> None:
    _GLOBAL.start(now=now)


def check(cost_usd: float | None = None, now: float | None = None) -> None:
    _GLOBAL.check(cost_usd=cost_usd, now=now)


def tripped() -> str | None:
    return _GLOBAL.tripped()


def snapshot() -> dict:
    return _GLOBAL.snapshot()


def write_abort_marker(workspace: str | Path) -> Path:
    """Create ``<workspace>/.abort`` so a running pipeline notices and stops.

    The marker contains the wall-clock timestamp for diagnostics. A workspace
    that doesn't exist raises ``FileNotFoundError`` so callers don't silently
    write into a typo path.
    """
    ws = Path(workspace)
    if not ws.is_dir():
        raise FileNotFoundError(f"workspace not found: {ws}")
    marker = ws / ".abort"
    marker.write_text(
        f"aborted at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n", encoding="utf-8"
    )
    return marker


def abort_marker_path(workspace: str | Path) -> Path:
    """Canonical location of the per-workspace abort marker."""
    return Path(workspace) / ".abort"
