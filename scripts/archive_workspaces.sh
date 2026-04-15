#!/usr/bin/env bash
# archive_workspaces.sh — TTL-based workspace archival (Whisper plan P2-6).
#
# Why: `workspaces/` accumulates a directory per research run — they can hit
# hundreds of MB each. Without cleanup the repo bloats and grep/rg gets slow.
# This script tars workspaces older than N days into `workspaces/_archive/`
# so they stay available but out of the way.
#
# Usage:
#   scripts/archive_workspaces.sh                    # archive >30-day-old (default)
#   scripts/archive_workspaces.sh --days 14          # custom threshold
#   scripts/archive_workspaces.sh --dry-run          # preview only
#   scripts/archive_workspaces.sh --days 14 --dry-run
#
# Safety:
# - Never touches `_archive/` itself (the sink).
# - Skips any workspace with a `.keep` file (pin from deletion).
# - Uses gtar on macOS if available (ARM64 brew); falls back to tar.
# - After successful archive, the original dir is removed.
# - --dry-run prints actions without running them.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACES="$HERE/workspaces"
ARCHIVE="$WORKSPACES/_archive"

DAYS=30
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$WORKSPACES" ]]; then
    echo "No workspaces/ dir — nothing to do."
    exit 0
fi

# Pick tar command (gtar on macOS via brew, else tar)
TAR="${TAR:-tar}"
if command -v gtar >/dev/null 2>&1; then
    TAR="gtar"
fi

echo "Scanning $WORKSPACES for dirs older than $DAYS days..."
[[ "$DRY_RUN" -eq 1 ]] && echo "(dry-run — no changes will be made)"

mkdir -p "$ARCHIVE"

# Enumerate top-level dirs (not files, not _archive, not hidden, max depth 1)
candidates=()
while IFS= read -r -d '' d; do
    base="$(basename "$d")"
    [[ "$base" == "_archive" ]] && continue
    [[ "$base" == .* ]] && continue
    if [[ -f "$d/.keep" ]]; then
        echo "  SKIP (pinned): $base"
        continue
    fi
    candidates+=("$d")
done < <(find "$WORKSPACES" -mindepth 1 -maxdepth 1 -type d -mtime +"$DAYS" -print0)

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No workspaces older than $DAYS days. Done."
    exit 0
fi

echo "Found ${#candidates[@]} workspace(s) to archive:"
for d in "${candidates[@]}"; do
    echo "  - $(basename "$d")"
done

archived=0
for d in "${candidates[@]}"; do
    base="$(basename "$d")"
    out="$ARCHIVE/${base}.tar.gz"

    if [[ -e "$out" ]]; then
        echo "  WARN: $out already exists — skipping to avoid overwrite"
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] would: $TAR -czf $out -C $WORKSPACES $base && rm -rf $d"
    else
        echo "  Archiving $base ..."
        if ( cd "$WORKSPACES" && $TAR -czf "$out" "$base" ); then
            rm -rf "$d"
            archived=$((archived + 1))
        else
            echo "  ERROR: tar failed for $base — leaving original intact"
        fi
    fi
done

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete. Re-run without --dry-run to actually archive."
else
    echo "Archived $archived workspace(s) to $ARCHIVE/"
fi
