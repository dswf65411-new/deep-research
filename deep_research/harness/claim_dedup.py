"""Claim deduplication utilities.

Two-tier approach:
  1. normalize_for_dedup(text) — fast exact match after normalization
     (whitespace collapse, punctuation strip, NFKC unicode, lowercase)
  2. is_near_duplicate(a, b, ratio=0.92) — difflib SequenceMatcher fuzzy match
     for near-identical paraphrases that survive normalization

Usage in _extract_chunked (intra-source):
    key = normalize_for_dedup(text)
    if key in seen_claim_texts: continue
    seen_claim_texts.add(key)

Usage in _collect_claims and phase2 (cross-round / cross-source):
    if any(is_near_duplicate(text, ex) for ex in existing_texts): continue
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Punctuation characters to strip during normalization
# (Latin + CJK punctuation; CJK chars included inline so the regex stays readable)
_PUNCT_RE = re.compile(
    r'[\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019'
    r'\u300c\u300d\u3010\u3011\u300a\u300b\u3008\u3009\u2026\u2014'
    r'\-,.!?;:"\'()\[\]{}\u00b7\u2022\u25c6\u25b6\u25b7\u203b\u2605\u2606\u2666]'
)
_WHITESPACE_RE = re.compile(r"[\s\u3000\u00a0\u200b]+")


def normalize_for_dedup(text: str) -> str:
    """Normalize claim text into a canonical deduplication key.

    Steps:
    1. NFKC unicode normalization (full-width → half-width, ligatures, etc.)
    2. Lowercase
    3. Strip all whitespace (including full-width space U+3000)
    4. Strip common punctuation (Chinese and Latin)

    Returns a compact string suitable as a set/dict key.
    """
    if not text:
        return ""
    # NFKC: full-width "A" (U+FF21) → "A", "1" circled (U+2460) → "1", "km" square (U+339E) → "km"
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    return text


def is_near_duplicate(a: str, b: str, ratio: float = 0.92) -> bool:
    """Return True if *a* and *b* are near-duplicates.

    Comparison is done on normalized forms to avoid false negatives from
    minor punctuation/whitespace variations. SequenceMatcher is only called
    when both strings have at least 15 characters (short strings use exact
    normalized match only).

    Args:
        a: First claim text.
        b: Second claim text.
        ratio: Similarity threshold (0.0–1.0). Default 0.92 ≈ "almost identical".

    Returns:
        True if the two texts are near-duplicates.
    """
    if not a or not b:
        return False
    na, nb = normalize_for_dedup(a), normalize_for_dedup(b)
    if not na or not nb:
        return False
    # Exact normalized match is always a duplicate
    if na == nb:
        return True
    # Skip SequenceMatcher for very short strings (< 15 chars after normalization)
    # to avoid false positives on generic short phrases
    if len(na) < 15 or len(nb) < 15:
        return False
    # Fast-reject: if lengths differ by more than 30%, cannot exceed ratio=0.92
    len_a, len_b = len(na), len(nb)
    if min(len_a, len_b) / max(len_a, len_b) < (1 - ratio):
        return False
    return SequenceMatcher(None, na, nb).ratio() >= ratio


def dedup_claims(
    claim_texts: list[str],
    existing_texts: list[str] | None = None,
    ratio: float = 0.92,
) -> list[int]:
    """Return indices of *claim_texts* that are NOT near-duplicates.

    Checks each new text against:
    1. All previous accepted texts in claim_texts (intra-batch dedup)
    2. All texts in existing_texts (cross-round dedup)

    Args:
        claim_texts: New claim texts to filter.
        existing_texts: Already-accepted claim texts from prior rounds.
        ratio: Similarity threshold passed to is_near_duplicate.

    Returns:
        List of indices (into claim_texts) that should be kept.
    """
    existing = list(existing_texts or [])
    accepted_texts: list[str] = list(existing)
    kept: list[int] = []

    for idx, text in enumerate(claim_texts):
        if not text or not text.strip():
            continue
        norm = normalize_for_dedup(text)
        # Check against all already-accepted texts
        is_dup = False
        for prev in accepted_texts:
            if is_near_duplicate(text, prev, ratio=ratio):
                is_dup = True
                break
        if not is_dup:
            kept.append(idx)
            accepted_texts.append(text)

    return kept
