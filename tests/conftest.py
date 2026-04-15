"""Shared pytest fixtures for deep-research tests.

This file is auto-discovered by pytest — no explicit import needed.
"""

from __future__ import annotations

import pytest

from deep_research.harness import runtime_limits


@pytest.fixture(autouse=True)
def _reset_runtime_limits():
    """Reset the global runtime-limits singleton before and after every test.

    The runtime-limits module holds process-global state (max_time, max_cost,
    abort marker, tripped flag). Without this fixture, a test that sets limits
    or trips the abort marker can leak into the next test and make it fail
    in confusing ways.
    """
    runtime_limits.reset()
    yield
    runtime_limits.reset()
