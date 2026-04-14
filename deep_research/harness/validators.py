"""Pydantic-level and programmatic validators (hard-rule enforcement).

Tier 1 = pure-Python hard rules. If code can decide it, don't hand it to an LLM.
Tier 2 = LLM verifier chain (Gemini-led; see ROLES in llm.py).

Call sites:
  - phase1a Extractor after each source is extracted (before _extract_one returns)
  - phase3 after the statement ledger is split (inside _split_one_section)
  - phase3 before audit, and before the report is produced

This file also serves as the "span-based quote infrastructure":
resolve_quote_index / verify_indexed_items are the shared helpers for
index-based quoting, used by phase1a (extracting quotes/numbers) and phase3
(splitting statements).
"""

from __future__ import annotations

import logging
import re
import unicodedata

from deep_research.state import Claim, Source

logger = logging.getLogger(__name__)


_BEDROCK_MIN = 0.3  # below this score = unverified (0.0) or clearly not grounded; must not enter Phase 2

# ---------------------------------------------------------------------------
# Off-topic metadata heuristics
# ---------------------------------------------------------------------------

# English street-address pattern: number + street name + street type (Street/Ave/Road etc.)
_STREET_ADDR_EN = re.compile(
    r"\b\d{1,6}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+"
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
    r"Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Way|Plaza|Place|Pl\.?|"
    r"Parkway|Pkwy\.?|Circle|Terrace|Trail)\b",
    re.IGNORECASE,
)
# US ZIP codes: NY 10001 / 90210-1234
_US_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
# Explicit address context keywords (English + CJK; CJK terms escaped to keep file ASCII-clean)
_ADDR_CONTEXT = re.compile(
    r"\b(?:located\s+at|headquartered\s+at|office(?:s)?\s+(?:at|in)\b|"
    r"address\s+is|"
    r"\u7e3d\u90e8\u4f4d\u65bc|"           # HQ located at
    r"\u516c\u53f8\u5730\u5740|"            # company address
    r"\u516c\u53f8\u4f4d\u65bc|"            # company located at
    r"\u8fa6\u516c\u5ba4\u4f4d\u65bc|"      # office located at
    r"\u4f4d\u65bc.{0,20}[\u5e02\u5340\u8def\u865f]"  # located at ... city/district/road/number
    r")\b",
    re.IGNORECASE,
)
# SEO marketing copy / customer service / contact info
_SEO_BOILERPLATE = re.compile(
    r"\b(?:contact\s+us|call\s+us|email\s+us|get\s+in\s+touch|"
    r"our\s+team\s+of\s+experts|dedicated\s+to\s+(?:providing|helping|serving)|"
    r"click\s+here\s+to|sign\s+up\s+(?:for\s+)?(?:free|today)|"
    r"(?:\d{3}[-.\s])?\d{3}[-.\s]\d{4})\b",  # phone number pattern
    re.IGNORECASE,
)
# Cookie / legal boilerplate
_LEGAL_BOILERPLATE = re.compile(
    r"\b(?:cookie(?:s)?\s+policy|privacy\s+policy|terms\s+of\s+service|"
    r"all\s+rights\s+reserved|copyright\s+©|©\s*\d{4})\b",
    re.IGNORECASE,
)


def _is_metadata_claim(claim_text: str) -> bool:
    """Return True if claim_text appears to be off-topic company/site metadata.

    Heuristic safety net — catches common patterns that slip through grounding
    because the text literally exists in the source (e.g. company address on About page).
    Called by validate_claims_for_phase2; also useful for early rejection.

    Does NOT attempt to judge topical relevance (that requires knowing the research
    question). It only rejects universal noise categories: addresses, contact info,
    legal boilerplate, SEO marketing copy.
    """
    if not claim_text:
        return False
    t = claim_text
    return bool(
        _STREET_ADDR_EN.search(t)
        or _US_ZIP.search(t)
        or _ADDR_CONTEXT.search(t)
        or _SEO_BOILERPLATE.search(t)
        or _LEGAL_BOILERPLATE.search(t)
    )


def validate_claims_for_phase2(claims: list[Claim]) -> list[Claim]:
    """Hard rule: only approved claims with quote_ids and bedrock_score >= 0.3 enter Phase 2.

    bedrock_score == 0.0 means grounding was never run (data-flow default).
    bedrock_score < 0.3 means the source text does not meaningfully support the claim.
    Both cases produce garbage in the final report (e.g. company addresses, NAS filenames).

    Additionally, heuristic metadata filter (_is_metadata_claim) rejects off-topic
    claims such as company addresses, contact info, and SEO boilerplate.
    """
    result = []
    for c in claims:
        if c.status != "approved":
            continue
        if not c.quote_ids:
            continue
        if c.bedrock_score < _BEDROCK_MIN:
            continue
        if _is_metadata_claim(c.claim_text):
            logger.debug(
                "validate_claims_for_phase2: removed metadata claim %s: %s",
                c.claim_id,
                c.claim_text[:80],
            )
            continue
        result.append(c)
    return result


def validate_numeric_claims(claims: list[Claim]) -> list[str]:
    """Hard rule: every numeric claim must have a number_tag."""
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
    """Hard rule: statement → claim_id → quote_id → source_id chain must be complete."""
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
    """Hard rule: sub-agents must NOT have search tools."""
    search_prefixes = ("brave_search", "serper_search", "serper_scrape", "web_fetch")
    return [t for t in tool_names if not t.startswith(search_prefixes)]


# ---------------------------------------------------------------------------
# Tier 1 — Quote / number / reference hard rules
# ---------------------------------------------------------------------------
#
# Rationale:
#   - The phase1a Extractor has the LLM emit quote/number/claim, but the
#     "original-text excerpts" it emits may be silently edited, have words
#     added, punctuation changed, or be fully hallucinated at any time.
#   - Hard rule 2 "original text first" requires quote to come verbatim from
#     the full text fetched by web search.
#   - Hard rule 3 "number citation chain" requires that each of the three
#     categories ORIGINAL/NORMALIZED/DERIVED carry a mandatory tag.
#   - Hard rule 4 "complete citation chain" requires that the
#     statement → claim_id → quote_id chain is unbroken.
#
# This layer is a pure-Python check; it does not call an LLM. phase1a verifies
# right after extraction and drops any violations; phase3 verifies again before
# the report. Catching failure as early as possible is best (a Tier 1 failure
# is a hard-rule violation and should not carry polluted data downstream).
#
# Tier 1 and Tier 2 are complementary: Tier 1 catches "LLM silently edited
# text", Tier 2 catches "LLM over-inference / lossy summarization". An item
# must pass both to be let through.


def _normalize_for_match(text: str) -> str:
    """Normalize text so substring comparison tolerates common "not really a hallucination" differences.

    Steps:
      - Unicode NFKC: full-width -> half-width, unify compatibility characters
        (fullwidth period -> ., fullwidth percent -> %)
      - Whitespace collapse: every whitespace sequence -> a single space (newline / tab / multi-space -> consistent)
      - Strip leading and trailing whitespace
      - Loosen punctuation: common CJK punctuation mapped to ASCII equivalents
        (corner brackets -> ", fullwidth comma / ideographic comma -> ,,
         fullwidth semicolon -> ;, fullwidth colon -> :, em-dash / box-drawing-dash -> -)

    Note: this function is only used to check "is it present in the original
    text"; it is never used to overwrite quote text. Quote text is still kept
    exactly as the LLM extracted it.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # CJK punctuation loosening
    cjk_punct_map = {
        "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',  # CJK corner brackets
        "\uff0c": ",", "\u3001": ",", "\uff1b": ";", "\uff1a": ":",  # fullwidth+ideographic comma/semicolon/colon
        "\u2014": "-", "\u2500": "-", "\u2013": "-",                  # em-dash / box-dash / en-dash
        "\uff08": "(", "\uff09": ")", "\u3010": "[", "\u3011": "]",  # fullwidth parens + black lenticular brackets
        "\uff01": "!", "\uff1f": "?",                                  # fullwidth exclamation/question
    }
    for k, v in cjk_punct_map.items():
        text = text.replace(k, v)
    # Whitespace collapse
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def validate_quotes_exist(
    quotes: list[dict],
    source_text: str,
    *,
    min_len: int = 8,
) -> list[str]:
    """Hard rule 2: quote text must be found in source_text (substring + normalization tolerance).

    Args:
        quotes: list of {"quote_id": ..., "text": ...} from phase1a
        source_text: raw source content (full text fetched by phase1a)
        min_len: quotes shorter than min_len chars are skipped to avoid false
                 positives from single chars / punctuation

    Returns:
        List of violation descriptions. Empty list = all passed.

    Matching strategy:
      1. Exact substring (most lenient)
      2. Substring after NFKC + punctuation + whitespace normalization
      3. Both fail → violation
    """
    if not source_text:
        return [f"validate_quotes_exist: source_text empty, cannot verify any quote"]

    norm_source = _normalize_for_match(source_text)
    violations: list[str] = []

    for q in quotes:
        qid = q.get("quote_id", "?")
        text = (q.get("text") or "").strip()
        if not text:
            violations.append(f"{qid}: empty quote text")
            continue
        if len(text) < min_len:
            continue  # too short, skip validation

        # Layer 1: exact substring
        if text in source_text:
            continue
        # Layer 2: substring after normalization
        norm_q = _normalize_for_match(text)
        if norm_q and norm_q in norm_source:
            continue

        snippet = text[:80].replace("\n", " ")
        violations.append(
            f"{qid}: quote not found in source raw content (text='{snippet}...')"
        )

    return violations


_DERIVED_FORMULA_RE = re.compile(
    # formula operators and keywords (EN + CJK; CJK escaped: U+516C U+5F0F = formula, U+8A08 U+7B97 = compute)
    r"(=|\u2248|\uff1d|\u2252|\u00d7|\u00f7|\+|\-|\*|/|\u516c\u5f0f|\u8a08\u7b97|formula|computed|derived)",
    re.IGNORECASE,
)
_NORMALIZED_ORIG_RE = re.compile(
    # match "(orig:" with ASCII ':' or full-width ':' (U+FF1A)
    r"\(\s*orig\s*[:\uff1a]",
    re.IGNORECASE,
)


def validate_number_tags(
    claims: list[Claim],
    numbers: list[dict] | None = None,
) -> list[str]:
    """Hard rule 3: every numeric claim must carry the correct tag among the three categories.

    ORIGINAL  : numbers in claim_text must match the value in the numbers ledger verbatim
    NORMALIZED: claim_text must contain an (orig: ...) marker
    DERIVED   : claim_text must contain a formula symbol (=, ×, ÷ or keywords like formula)

    Args:
        claims: claims to validate
        numbers: numbers ledger (optional). When provided, ORIGINAL values are
                 compared against the ledger; otherwise only format is checked.

    Returns:
        List of violation descriptions. Empty list = all passed.
    """
    violations: list[str] = []

    # Build the number_id → value index
    num_value_map: dict[str, str] = {}
    if numbers:
        for n in numbers:
            nid = n.get("number_id")
            val = str(n.get("value", "")).strip()
            if nid and val:
                num_value_map[nid] = val

    for c in claims:
        if c.claim_type != "numeric":
            continue

        tag = c.number_tag
        if tag is None:
            violations.append(f"{c.claim_id}: numeric claim missing number_tag")
            continue

        text = c.claim_text or ""

        if tag == "ORIGINAL":
            # value must appear in claim_text verbatim
            if num_value_map and c.quote_ids:
                related_values = [
                    num_value_map[qid] for qid in c.quote_ids if qid in num_value_map
                ]
                if related_values:
                    if not any(v in text for v in related_values):
                        joined = ", ".join(related_values)
                        violations.append(
                            f"{c.claim_id}: ORIGINAL claim_text does not contain the corresponding number value (expected one of: {joined})"
                        )

        elif tag == "NORMALIZED":
            if not _NORMALIZED_ORIG_RE.search(text):
                violations.append(
                    f"{c.claim_id}: NORMALIZED is missing the (orig: ...) marker"
                )

        elif tag == "DERIVED":
            if not _DERIVED_FORMULA_RE.search(text):
                violations.append(
                    f"{c.claim_id}: DERIVED is missing a formula / computation marker (=, \u00d7, \u00f7, formula, etc.)"
                )

    return violations


def validate_quotes_indexed(
    quotes: list[dict],
    source_text: str,
) -> list[str]:
    """Hard rule 2 (index variant): each quote must carry start/end, and source[start:end] == text.

    Runs a pure index-based check: the start/end produced by phase1a's
    _resolve_quote_index and recorded in the quote dict must satisfy
    source_text[start:end] == text. This is the strictest check, fully avoiding
    substring / normalization fuzzy matches — "silent edits" cannot slip through.

    Args:
        quotes: list of {"quote_id", "text"/"sentence", "start", "end"}
        source_text: the full source raw content

    Returns:
        List of violation descriptions; empty list = all passed.

    Note:
        Complementary to validate_quotes_exist: _exist handles "legacy / no
        index" quotes, _indexed handles "LLM + code-verified" quotes. They are
        not applied to the same quote — phase1a produces only the latter.
        _exist remains as a fallback (e.g. manual quotes in phase2/3).
    """
    if not source_text:
        return [f"validate_quotes_indexed: source_text empty, cannot verify any quote"]

    violations: list[str] = []
    for q in quotes:
        qid = q.get("quote_id") or q.get("number_id") or "?"
        text = q.get("text") or q.get("sentence") or ""
        start = q.get("start")
        end = q.get("end")

        if start is None or end is None:
            violations.append(f"{qid}: missing start/end index")
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            violations.append(f"{qid}: start/end not int (got {type(start).__name__}/{type(end).__name__})")
            continue
        if start < 0 or end > len(source_text) or start >= end:
            violations.append(
                f"{qid}: invalid span [{start}:{end}] for source len={len(source_text)}"
            )
            continue
        actual = source_text[start:end]
        if actual != text:
            snippet_a = actual[:60].replace("\n", " ")
            snippet_t = text[:60].replace("\n", " ")
            violations.append(
                f"{qid}: source[{start}:{end}]='{snippet_a}...' does not match text='{snippet_t}...'"
            )

    return violations


def validate_quote_ids_in_ledger(
    claims: list[Claim],
    quotes: list[dict],
    numbers: list[dict] | None = None,
) -> list[str]:
    """Hard rule 4: every quote_id referenced by a claim must actually exist in the quotes or numbers ledger.

    Args:
        claims: claims to validate
        quotes: quotes ledger
        numbers: numbers ledger (in phase1a, quote_ids are the union of quotes + numbers)

    Returns:
        List of violations.
    """
    valid_ids: set[str] = {q.get("quote_id") for q in quotes if q.get("quote_id")}
    if numbers:
        valid_ids.update(n.get("number_id") for n in numbers if n.get("number_id"))

    violations: list[str] = []
    for c in claims:
        for qid in c.quote_ids:
            if qid not in valid_ids:
                violations.append(
                    f"{c.claim_id}: quote_id {qid} not in ledger"
                )
    return violations


# ---------------------------------------------------------------------------
# Index-based quote resolution (compound validation) — shared utilities
# ---------------------------------------------------------------------------
#
# Design references:
#   - Anthropic Claude Citations API (2025-01): start_char_index / end_char_index,
#     0-indexed, end-exclusive (Python slice semantics). This implementation aligns.
#   - LAQuer (ACL 2025) / Attribute First, then Generate: span-based quoting as a hard constraint.
#   - LlamaIndex TextNode lesson: purely auto-maintained offsets are error-prone.
#     We use the compound "LLM output + code validation + find fallback" approach,
#     which is more robust than pure index.
#
# Three-layer validation flow:
#   1. hint hit: raw[start_hint:end_hint] == llm_text → adopt (hint correct)
#   2. find fallback: raw.find(llm_text) → if found, use the true position
#      discovered by code (LLM gave an off-by index)
#   3. reject: both fail → drop (LLM hallucinated text)
#
# Consumers:
#   - phase1a: when extracting quote/number, verify that text actually exists in source raw
#   - phase3: when splitting the statement ledger, verify that statement.text actually exists in section content

# 1:1 typographic character normalization table
# All replacements are single-char→single-char, so indices are preserved after translation.
_TYPO_TRANS = str.maketrans({
    '\u2018': "'",   # LEFT SINGLE QUOTATION MARK  → '
    '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK → '
    '\u201c': '"',   # LEFT DOUBLE QUOTATION MARK  → "
    '\u201d': '"',   # RIGHT DOUBLE QUOTATION MARK → "
    '\u2013': '-',   # EN DASH  → -
    '\u2014': '-',   # EM DASH  → -
    '\u00a0': ' ',   # NON-BREAKING SPACE → SPACE
    '\u202f': ' ',   # NARROW NO-BREAK SPACE → SPACE
})


def resolve_quote_index(
    raw: str,
    llm_text: str,
    start_hint,
    end_hint,
) -> tuple[str, int, int] | None:
    """Return (verified_text, start, end) from raw, or None if text can't be anchored.

    Args:
        raw: the reference text being validated against (source / section / any container)
        llm_text: the quoted text produced by the LLM
        start_hint / end_hint: LLM-supplied index hints (may be None or int)

    Four-layer fallback:
      1. hint correct → return raw[start:end]
      2. hint wrong but text is findable → return the found position
      3. find after typographic-quote normalization → handles HTML &rsquo; (U+2019) vs ASCII ' (U+0027) etc.
      4. whitespace-relaxed search → handles extra whitespace left over from HTML stripping

    Note:
        verified_text always satisfies raw[start:end] == verified_text (round-trippable).
    """
    if not llm_text or not raw:
        return None

    llm_text = llm_text.strip()
    if not llm_text:
        return None

    # Layer 1: hint is exact match
    if isinstance(start_hint, int) and isinstance(end_hint, int):
        if 0 <= start_hint < end_hint <= len(raw):
            candidate = raw[start_hint:end_hint]
            if candidate == llm_text:
                return (candidate, start_hint, end_hint)

    # Layer 2: exact string find
    idx = raw.find(llm_text)
    if idx >= 0:
        return (llm_text, idx, idx + len(llm_text))

    # Layer 3: typographic-char normalized find
    # _TYPO_TRANS is 1:1 char replacement → len(raw_n) == len(raw), indices preserved
    raw_n = raw.translate(_TYPO_TRANS)
    llm_n = llm_text.translate(_TYPO_TRANS)
    idx = raw_n.find(llm_n)
    if idx >= 0:
        end = idx + len(llm_n)
        return (raw[idx:end], idx, end)

    # Layer 4: whitespace-relaxed search (handles extra spaces from HTML <div> stripping)
    try:
        parts = re.split(r'(\s+)', llm_n)
        pat = ''.join(
            r'\s+' if re.match(r'^\s+$', p) else re.escape(p)
            for p in parts if p
        )
        if pat:
            m = re.search(pat, raw_n, re.DOTALL)
            if m:
                return (raw[m.start():m.end()], m.start(), m.end())
    except re.error:
        pass

    return None


def verify_indexed_items(
    raw: str,
    items: list[dict],
    text_field: str,
    *,
    chunk_offset: int = 0,
    log_prefix: str = "",
) -> list[dict]:
    """Run index-based validation over each LLM-produced item (quote / number / statement).

    Args:
        raw: the reference text being validated against (one source / one chunk / one section content)
        items: the raw LLM output list; each entry should carry text_field, start_char, end_char
        text_field: "text" for quote, "sentence" for number, "text" for statement
        chunk_offset: raw's starting offset within the full document (used in chunked mode; 0 otherwise)
        log_prefix: prefix label for warning logs (phase name)

    For each item:
      - Use resolve_quote_index to verify that text_field really exists in raw
      - Pass: overwrite text_field with the true value from raw (eliminates LLM
        silent edits) and attach start/end (converted to "global coordinates"
        = relative coord + chunk_offset)
      - Fail: drop (log a warning)

    Returns: items that passed validation (failed entries filtered out).
    """
    verified: list[dict] = []
    for it in items:
        llm_text = (it.get(text_field) or "").strip()
        if not llm_text:
            continue
        resolved = resolve_quote_index(
            raw,
            llm_text,
            it.get("start_char"),
            it.get("end_char"),
        )
        if resolved is None:
            tag = f"[{log_prefix}] " if log_prefix else ""
            logger.warning(
                "%s%s '%s' could not be located in raw — drop",
                tag,
                text_field,
                llm_text[:60].replace("\n", " "),
            )
            continue
        real_text, s, e = resolved
        new_it = {**it}
        new_it[text_field] = real_text
        new_it["start"] = s + chunk_offset
        new_it["end"] = e + chunk_offset
        verified.append(new_it)
    return verified
