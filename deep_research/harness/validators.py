"""Pydantic-level and programmatic validators (iron rules enforcement).

Tier 1 = 純 Python 硬規則。能用程式判斷的就不交給 LLM。
Tier 2 = LLM verifier chain（Gemini 主導，見 llm.py ROLES）。

呼叫時機：
  - phase1a Extractor 抽完每篇 source 後（_extract_one return 前）
  - phase3 statement ledger 切分後（_split_one_section 內）
  - phase3 audit 前 + 報告產出前

此檔兼做「span-based 引用基礎設施」：resolve_quote_index / verify_indexed_items
為 index-based 引用的共用工具，phase1a 抽 quote/number、phase3 切 statement 都用。
"""

from __future__ import annotations

import logging
import re
import unicodedata

from deep_research.state import Claim, Source

logger = logging.getLogger(__name__)


_BEDROCK_MIN = 0.3  # 低於此分數 = 未驗證(0.0) 或明顯 not grounded；不得進入 Phase 2

# ---------------------------------------------------------------------------
# Off-topic metadata heuristics
# ---------------------------------------------------------------------------

# 英文街道地址模式：門牌 + 路名 + 路型（Street/Ave/Road 等）
_STREET_ADDR_EN = re.compile(
    r"\b\d{1,6}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+"
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
    r"Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Way|Plaza|Place|Pl\.?|"
    r"Parkway|Pkwy\.?|Circle|Terrace|Trail)\b",
    re.IGNORECASE,
)
# 美國郵遞區號：NY 10001 / 90210-1234
_US_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
# 明確地址語境詞
_ADDR_CONTEXT = re.compile(
    r"\b(?:located\s+at|headquartered\s+at|office(?:s)?\s+(?:at|in)\b|"
    r"address\s+is|總部位於|公司地址|公司位於|辦公室位於|位於.{0,20}[市區路號])\b",
    re.IGNORECASE,
)
# SEO 行銷語句 / 客服 / 聯絡資訊
_SEO_BOILERPLATE = re.compile(
    r"\b(?:contact\s+us|call\s+us|email\s+us|get\s+in\s+touch|"
    r"our\s+team\s+of\s+experts|dedicated\s+to\s+(?:providing|helping|serving)|"
    r"click\s+here\s+to|sign\s+up\s+(?:for\s+)?(?:free|today)|"
    r"(?:\d{3}[-.\s])?\d{3}[-.\s]\d{4})\b",  # phone number pattern
    re.IGNORECASE,
)
# Cookie / 法律樣板
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
    """Iron rule: only approved claims with quote_ids and bedrock_score >= 0.3 enter Phase 2.

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
                "validate_claims_for_phase2: 過濾 metadata claim %s: %s",
                c.claim_id,
                c.claim_text[:80],
            )
            continue
        result.append(c)
    return result


def validate_numeric_claims(claims: list[Claim]) -> list[str]:
    """Iron rule: every numeric claim must have a number_tag."""
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
    """Iron rule: statement → claim_id → quote_id → source_id chain must be complete."""
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
    """Iron rule: sub-agents must NOT have search tools."""
    search_prefixes = ("brave_search", "serper_search", "serper_scrape", "web_fetch")
    return [t for t in tool_names if not t.startswith(search_prefixes)]


# ---------------------------------------------------------------------------
# Tier 1 — Quote / Number / Reference 硬規則
# ---------------------------------------------------------------------------
#
# 設計理由：
#   - phase1a Extractor 由 LLM 輸出 quote/number/claim，但 LLM 輸出的「原文摘
#     錄」隨時可能被偷改字 / 加詞 / 改標點 / 幻覺整段。
#   - 鐵律 2「原文先行」要求 quote 必須來自 web search 全文逐字。
#   - 鐵律 3「數字溯源」要求 ORIGINAL/NORMALIZED/DERIVED 三類各有強制標記。
#   - 鐵律 4「溯源鏈完整」要求 statement → claim_id → quote_id 鏈不斷。
#
# 本層為純 Python 校驗，不呼叫 LLM。phase1a 抽完即驗，違規即 drop；phase3
# 報告前再驗一次。failure 越早抓住越好（Tier 1 失敗 = 鐵律 violation，
# 不該往下游帶汙染資料）。
#
# 與 Tier 2 互補：Tier 1 抓「LLM 偷改字」、Tier 2 抓「LLM 推論過度 / 摘要
# 失真」。兩層都過才放行。


def _normalize_for_match(text: str) -> str:
    """正規化文本，讓 substring 比對能容忍常見的「不算幻覺」差異。

    處理：
      - Unicode NFKC：全形 → 半形，相容字符統一（。 → .，％ → %）
      - 空白壓縮：所有 whitespace 序列 → 單一空格（換行 / tab / 多空格 → 一致）
      - 去頭尾空白
      - 標點寬鬆化：常見 CJK 標點對應到 ASCII 等價物
        （「」→ "", ，→ ,, 、→ ,, ；→ ;, ：→ :, ─ → -, — → -）

    注意：本函式只用於「是否存在於原文」的比對，不會用來覆蓋 quote 文字。
    quote 文字仍以 LLM 抽出的原樣保留。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # CJK 標點寬鬆化
    cjk_punct_map = {
        "「": '"', "」": '"', "『": '"', "』": '"',
        "，": ",", "、": ",", "；": ";", "：": ":",
        "—": "-", "─": "-", "–": "-",
        "（": "(", "）": ")", "【": "[", "】": "]",
        "！": "!", "？": "?",
    }
    for k, v in cjk_punct_map.items():
        text = text.replace(k, v)
    # 空白壓縮
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def validate_quotes_exist(
    quotes: list[dict],
    source_text: str,
    *,
    min_len: int = 8,
) -> list[str]:
    """Iron rule 2: quote text 必須能在 source_text 中找到（substring + 正規化容差）。

    Args:
        quotes: list of {"quote_id": ..., "text": ...} from phase1a
        source_text: 原始 source raw content（phase1a 抓的網頁全文）
        min_len: 太短的 quote（< min_len 字元）跳過驗證 — 避免單字 / 標點誤判

    Returns:
        violation 描述清單。空 list = 全部通過。

    比對策略：
      1. 完全 substring（最寬鬆）
      2. NFKC + 標點 + 空白正規化後的 substring
      3. 都失敗 → violation
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
            continue  # 太短不驗證

        # 第 1 層：完全 substring
        if text in source_text:
            continue
        # 第 2 層：正規化後 substring
        norm_q = _normalize_for_match(text)
        if norm_q and norm_q in norm_source:
            continue

        snippet = text[:80].replace("\n", " ")
        violations.append(
            f"{qid}: quote not found in source raw content (text='{snippet}...')"
        )

    return violations


_DERIVED_FORMULA_RE = re.compile(
    r"(=|≈|＝|≒|×|÷|\+|\-|\*|/|公式|計算|formula|computed|derived)",
    re.IGNORECASE,
)
_NORMALIZED_ORIG_RE = re.compile(
    r"\(\s*orig\s*[:：]",
    re.IGNORECASE,
)


def validate_number_tags(
    claims: list[Claim],
    numbers: list[dict] | None = None,
) -> list[str]:
    """Iron rule 3: 數字 claim 三類標記必須齊全且正確。

    ORIGINAL  : claim_text 中的數字必須與 numbers ledger 的 value 逐字一致
    NORMALIZED: claim_text 必須含 (orig: ...) 標記
    DERIVED   : claim_text 必須含公式符號（=, ×, ÷ 或 公式 / formula 等關鍵字）

    Args:
        claims: 要驗證的 claims
        numbers: numbers ledger（可選）。提供時做 ORIGINAL value 比對；
                 不提供則只做格式檢查。

    Returns:
        violation 描述清單。空 list = 全部通過。
    """
    violations: list[str] = []

    # 建立 number_id → value 索引
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
            # value 必須出現在 claim_text 中（逐字）
            if num_value_map and c.quote_ids:
                related_values = [
                    num_value_map[qid] for qid in c.quote_ids if qid in num_value_map
                ]
                if related_values:
                    if not any(v in text for v in related_values):
                        joined = ", ".join(related_values)
                        violations.append(
                            f"{c.claim_id}: ORIGINAL claim_text 不含對應 number value（expected one of: {joined}）"
                        )

        elif tag == "NORMALIZED":
            if not _NORMALIZED_ORIG_RE.search(text):
                violations.append(
                    f"{c.claim_id}: NORMALIZED 缺 (orig: ...) 標記"
                )

        elif tag == "DERIVED":
            if not _DERIVED_FORMULA_RE.search(text):
                violations.append(
                    f"{c.claim_id}: DERIVED 缺公式 / 計算標記（=, ×, ÷, 公式, formula 等）"
                )

    return violations


def validate_quotes_indexed(
    quotes: list[dict],
    source_text: str,
) -> list[str]:
    """Iron rule 2（index 版）：每個 quote 必須帶 start/end，且 source[start:end] == text。

    走純 index 驗證：由 phase1a 的 _resolve_quote_index 產生並記錄到 quote dict 的
    start/end 必須滿足 source_text[start:end] == text。這是最硬的驗證，
    完全避開 substring / 正規化 fuzzy match — 不存在「偷改字」可能。

    Args:
        quotes: list of {"quote_id", "text"/"sentence", "start", "end"}
        source_text: 整篇 source raw content

    Returns:
        violation 描述清單；空 list = 全部通過。

    Note:
        與 validate_quotes_exist 互補：_exist 處理「舊格式 / 沒 index」的 quote，
        _indexed 處理「LLM + 程式驗證過的」quote。兩者不會同時用在同一筆 quote —
        phase1a 只會產生後者。_exist 保留作為 fallback（e.g. phase2/3 的手動 quote）。
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
                f"{qid}: source[{start}:{end}]='{snippet_a}...' 與 text='{snippet_t}...' 不符"
            )

    return violations


def validate_quote_ids_in_ledger(
    claims: list[Claim],
    quotes: list[dict],
    numbers: list[dict] | None = None,
) -> list[str]:
    """Iron rule 4: claim 引用的 quote_ids 必須真實存在於 quotes 或 numbers ledger。

    Args:
        claims: 要驗證的 claims
        quotes: quotes ledger
        numbers: numbers ledger（quote_ids 在 phase1a 是 quotes + numbers 的合集）

    Returns:
        violation 清單。
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
# Index-based quote resolution（複合驗證）— 共用工具
# ---------------------------------------------------------------------------
#
# 設計參考：
#   - Anthropic Claude Citations API（2025-01）：start_char_index / end_char_index，
#     0-indexed、end exclusive（Python slice 語意）。本實作對齊此語意。
#   - LAQuer (ACL 2025) / Attribute First, then Generate：span-based 引用為硬約束。
#   - LlamaIndex TextNode 教訓：純自動維護 offset 易出錯。我們走「LLM 輸出 + 程式
#     驗證 + find fallback」的複合版，比純 index 更 robust。
#
# 三層驗證流程：
#   1. hint 命中：raw[start_hint:end_hint] == llm_text → 採用（hint 正確）
#   2. find fallback：raw.find(llm_text) → 找到就改用程式真位置（LLM 算偏 index）
#   3. reject：兩者都失敗 → drop（LLM 幻覺 text）
#
# 使用者：
#   - phase1a：抽 quote/number 時驗 text 真實存在於 source raw
#   - phase3：切 statement ledger 時驗 statement.text 真實存在於 section content

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
        raw: 驗證基準文字（source / section / 任何 container）
        llm_text: LLM 輸出的引用文字
        start_hint / end_hint: LLM 給的 index hint（可能為 None 或 int）

    四層 fallback：
      1. hint 正確 → 回 raw[start:end]
      2. hint 偏但 text 能 find → 回 find 到的位置
      3. 排版引號規範化後 find → 處理 HTML &rsquo; (U+2019) vs ASCII ' (U+0027) 等
      4. 空白鬆弛搜尋 → 處理 HTML stripping 後多餘空白

    Note:
        verified_text 永遠滿足 raw[start:end] == verified_text（可 round-trip）。
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
    """對 LLM 輸出的引用清單（quote / number / statement）逐筆做 index 驗證。

    Args:
        raw: 驗證基準文字（單篇 source / 單一 chunk / 單一 section content）
        items: LLM 輸出的原始清單，每筆應含 text_field、start_char、end_char
        text_field: quote 用 "text"、number 用 "sentence"、statement 也用 "text"
        chunk_offset: raw 在完整文件中的起點（chunked 模式用，其餘模式 = 0）
        log_prefix: warning 記錄時的前綴標記（phase 名）

    對每筆：
      - 用 resolve_quote_index 驗 text_field 是否真實存在於 raw
      - 通過：把 text_field 覆寫為 raw 的真值（杜絕 LLM 偷改字），附上 start/end
        （為「全域座標」= 相對座標 + chunk_offset）
      - 失敗：drop（記 warning）

    回傳：通過驗證的 items（失敗者已濾除）。
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
                "%s%s '%s' 無法在 raw 定位 — drop",
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
