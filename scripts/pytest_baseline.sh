#!/usr/bin/env bash
# pytest_baseline.sh — red/green diff against a stored baseline.
#
# Why: when resuming work or running in autonomous mode, "these 4 failures
# are pre-existing" is a dangerous claim unless you have a recorded baseline.
# This script makes it trivial to prove what the current run actually changed.
#
# Usage:
#   scripts/pytest_baseline.sh save        # store current failures as baseline
#   scripts/pytest_baseline.sh diff        # run pytest, print NEW red + NEWLY green
#   scripts/pytest_baseline.sh show        # dump the stored baseline
#   scripts/pytest_baseline.sh clear       # delete the stored baseline
#
# The baseline lives at .pytest_baseline (git-ignore it).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$HERE/.pytest_baseline"
PYTEST="${PYTEST:-$HERE/.venv/bin/python -m pytest}"

run_pytest_collect_failures() {
    # Capture failures as "path::test_name"; exit code ignored so we can
    # distinguish "pytest error" from "tests failed".
    local tmp
    tmp="$(mktemp)"
    ( cd "$HERE" && $PYTEST tests/ --tb=no -q 2>&1 ) > "$tmp" || true
    # pytest reports failures on lines starting with "FAILED ". grep must be
    # allowed to fail (no matches) without tripping set -e.
    { grep -E '^FAILED ' "$tmp" || true; } | awk '{print $2}' | sort -u
    rm -f "$tmp"
}

cmd="${1:-diff}"

case "$cmd" in
    save)
        echo "Running pytest to capture baseline failures..." >&2
        run_pytest_collect_failures > "$BASELINE"
        count="$(wc -l < "$BASELINE" | tr -d ' ')"
        echo "Baseline saved to $BASELINE — $count failing test(s)."
        ;;

    diff)
        if [[ ! -f "$BASELINE" ]]; then
            echo "ERROR: no baseline at $BASELINE. Run '$0 save' first." >&2
            exit 2
        fi
        echo "Running pytest to compare against baseline..." >&2
        current="$(mktemp)"
        run_pytest_collect_failures > "$current"

        new_red="$(comm -13 "$BASELINE" "$current")"
        new_green="$(comm -23 "$BASELINE" "$current")"

        echo
        echo "=== Baseline diff ==="
        if [[ -z "$new_red" && -z "$new_green" ]]; then
            echo "No change — same failure set as baseline."
        fi
        if [[ -n "$new_red" ]]; then
            echo
            echo "NEW FAILURES (introduced since baseline):"
            echo "$new_red" | sed 's/^/  - /'
        fi
        if [[ -n "$new_green" ]]; then
            echo
            echo "NEWLY PASSING (fixed since baseline):"
            echo "$new_green" | sed 's/^/  + /'
        fi
        rm -f "$current"

        # Exit non-zero only if we regressed (new red).
        [[ -z "$new_red" ]]
        ;;

    show)
        if [[ ! -f "$BASELINE" ]]; then
            echo "No baseline stored."
            exit 0
        fi
        cat "$BASELINE"
        ;;

    clear)
        rm -f "$BASELINE"
        echo "Baseline cleared."
        ;;

    *)
        echo "Usage: $0 {save|diff|show|clear}" >&2
        exit 1
        ;;
esac
