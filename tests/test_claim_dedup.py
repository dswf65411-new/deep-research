"""Tests for claim_dedup module and its integration in phase1a / phase2.

Covers:
- normalize_for_dedup: whitespace collapse, punct strip, NFKC, lowercase
- is_near_duplicate: exact-normalized match, SequenceMatcher near-match, fast-reject
- dedup_claims: intra-batch and cross-batch dedup
- _extract_chunked: uses normalize_for_dedup key
- _collect_claims: near-dup guard against existing_claims
- phase2: _dedup_approved_claims exists and filters near-dups
"""

import inspect


# ---------------------------------------------------------------------------
# normalize_for_dedup
# ---------------------------------------------------------------------------

def test_normalize_lowercases():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    assert normalize_for_dedup("WHISPER") == "whisper"


def test_normalize_strips_whitespace():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    assert normalize_for_dedup("  hello   world  ") == "helloworld"


def test_normalize_strips_chinese_punct():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    result = normalize_for_dedup("Whisper，準確率：90%。")
    assert "，" not in result
    assert "。" not in result
    assert "：" not in result


def test_normalize_nfkc_fullwidth():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    # Fullwidth ASCII → half-width
    assert normalize_for_dedup("ＡＢＣ") == "abc"


def test_normalize_strips_latin_punct():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    result = normalize_for_dedup("Hello, World! (2025)")
    assert "," not in result
    assert "!" not in result
    assert "(" not in result


def test_normalize_empty():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    assert normalize_for_dedup("") == ""


def test_normalize_same_content_different_punct():
    from deep_research.harness.claim_dedup import normalize_for_dedup
    a = normalize_for_dedup("Whisper 的準確率大約在 90%。")
    b = normalize_for_dedup("Whisper 的準確率大約在90%")
    assert a == b


# ---------------------------------------------------------------------------
# is_near_duplicate
# ---------------------------------------------------------------------------

def test_near_dup_exact_normalized_match():
    from deep_research.harness.claim_dedup import is_near_duplicate
    # Differ only in whitespace/punct → should be near-dup
    a = "Apple 在 2022 年推出了 M2 晶片，提升效能 40%"
    b = "Apple 在2022年推出了M2晶片，提升效能40%"
    assert is_near_duplicate(a, b) is True


def test_near_dup_minor_punct_variation():
    from deep_research.harness.claim_dedup import is_near_duplicate
    a = "Whisper 的準確率大約在 90%。"
    b = "Whisper 的準確率大約在90%"
    assert is_near_duplicate(a, b) is True


def test_near_dup_different_content_returns_false():
    from deep_research.harness.claim_dedup import is_near_duplicate
    assert is_near_duplicate("Whisper 準確率高達 95%", "Notta 是企業級工具") is False


def test_near_dup_empty_returns_false():
    from deep_research.harness.claim_dedup import is_near_duplicate
    assert is_near_duplicate("", "abc") is False
    assert is_near_duplicate("abc", "") is False


def test_near_dup_too_short_returns_false():
    from deep_research.harness.claim_dedup import is_near_duplicate
    # Under 15 chars after normalize — short strings use exact match only
    assert is_near_duplicate("short", "shorts") is False


def test_near_dup_custom_ratio():
    from deep_research.harness.claim_dedup import is_near_duplicate
    # At lower ratio threshold, slightly different strings should match
    a = "Whisper 是 OpenAI 的語音識別模型"
    b = "Whisper 是 OpenAI 推出的語音識別模型"
    # ratio=0.70 should catch this
    assert is_near_duplicate(a, b, ratio=0.70) is True
    # ratio=0.99 should not catch this
    assert is_near_duplicate(a, b, ratio=0.99) is False


# ---------------------------------------------------------------------------
# dedup_claims
# ---------------------------------------------------------------------------

def test_dedup_claims_removes_near_dup():
    from deep_research.harness.claim_dedup import dedup_claims
    texts = [
        "Apple 在 2022 年推出了 M2 晶片，提升效能 40%",
        "Apple 在2022年推出了M2晶片，提升效能40%",  # near-dup
        "Notta 是另一個完全不同的工具，適合企業使用",
    ]
    kept = dedup_claims(texts)
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_dedup_claims_keeps_all_when_no_dups():
    from deep_research.harness.claim_dedup import dedup_claims
    texts = [
        "Whisper 準確率高達 95%，適合轉錄",
        "Notta 提供即時字幕功能",
        "MacWhisper 是 Mac 原生應用程式",
    ]
    kept = dedup_claims(texts)
    assert kept == [0, 1, 2]


def test_dedup_claims_cross_batch_existing():
    from deep_research.harness.claim_dedup import dedup_claims
    existing = ["Whisper 準確率高達 95%，適合轉錄"]
    new_texts = [
        "Whisper 準確率高達95%，適合轉錄",  # near-dup of existing
        "Notta 提供即時字幕功能",
    ]
    kept = dedup_claims(new_texts, existing_texts=existing)
    assert 0 not in kept  # deduped against existing
    assert 1 in kept


def test_dedup_claims_empty_input():
    from deep_research.harness.claim_dedup import dedup_claims
    assert dedup_claims([]) == []


# ---------------------------------------------------------------------------
# phase1a integration
# ---------------------------------------------------------------------------

def test_extract_chunked_uses_normalize_for_dedup():
    """_extract_chunked source must use normalize_for_dedup for claim dedup."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._extract_chunked)
    assert "normalize_for_dedup" in src
    assert "seen_claim_norm" in src or "seen_claim" in src


def test_collect_claims_uses_is_near_duplicate():
    """_collect_claims must guard against existing claims using is_near_duplicate."""
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._collect_claims)
    assert "is_near_duplicate" in src
    assert "existing" in src


def test_collect_claims_dedup_against_existing():
    """New claim that is near-duplicate of existing should be dropped."""
    import deep_research.nodes.phase1a as p1a
    from deep_research.state import Claim

    existing = [
        Claim(
            claim_id="Q1-C1",
            subquestion="Q1",
            claim_text="Apple 在 2022 年推出了 M2 晶片，提升效能 40%",
            source_ids=["S001"],
        )
    ]
    # near-dup of existing Q1-C1
    extractions = [{
        "source_id": "S002",
        "claims": [{
            "subquestion": "Q1",
            "claim_text": "Apple 在2022年推出了M2晶片，提升效能40%",
            "claim_type": "qualitative",
            "quote_ids": [],
        }],
    }]
    result = p1a._collect_claims(extractions, existing_claims=existing)
    assert len(result) == 0, f"Expected 0 claims (near-dup dropped), got {len(result)}"


def test_collect_claims_keeps_different_claims():
    """Non-duplicate new claims must be kept."""
    import deep_research.nodes.phase1a as p1a
    from deep_research.state import Claim

    existing = [
        Claim(
            claim_id="Q1-C1",
            subquestion="Q1",
            claim_text="Whisper 的準確率高達 95%",
            source_ids=["S001"],
        )
    ]
    extractions = [{
        "source_id": "S002",
        "claims": [{
            "subquestion": "Q1",
            "claim_text": "Notta 提供即時字幕與摘要功能，適合會議錄音場景使用",
            "claim_type": "qualitative",
            "quote_ids": [],
        }],
    }]
    result = p1a._collect_claims(extractions, existing_claims=existing)
    assert len(result) == 1
    assert result[0].claim_id == "Q1-C2"


# ---------------------------------------------------------------------------
# phase2 integration
# ---------------------------------------------------------------------------

def test_phase2_has_dedup_approved_claims():
    import deep_research.nodes.phase2 as p2
    assert hasattr(p2, "_dedup_approved_claims")


def test_phase2_dedup_approved_claims_filters_near_dups():
    import deep_research.nodes.phase2 as p2
    from deep_research.state import Claim

    claims = [
        Claim(
            claim_id="Q1-C1",
            subquestion="Q1",
            claim_text="Apple 在 2022 年推出了 M2 晶片，提升效能 40%",
            source_ids=["S001"],
            quote_ids=["QT001"],
        ),
        Claim(
            claim_id="Q1-C2",
            subquestion="Q1",
            claim_text="Apple 在2022年推出了M2晶片，提升效能40%",  # near-dup
            source_ids=["S002"],
            quote_ids=["QT002"],
        ),
        Claim(
            claim_id="Q1-C3",
            subquestion="Q1",
            claim_text="Notta 是另一家公司的完全不同產品，定位企業市場",
            source_ids=["S003"],
            quote_ids=["QT003"],
        ),
    ]
    result = p2._dedup_approved_claims(claims)
    ids = [c.claim_id for c in result]
    assert "Q1-C1" in ids
    assert "Q1-C2" not in ids  # near-dup removed
    assert "Q1-C3" in ids


def test_phase2_dedup_preserves_cross_subq():
    """Claims from different subquestions must NOT be deduped against each other."""
    import deep_research.nodes.phase2 as p2
    from deep_research.state import Claim

    claims = [
        Claim(
            claim_id="Q1-C1",
            subquestion="Q1",
            claim_text="Whisper 是 OpenAI 開發的語音識別系統",
            source_ids=["S001"],
            quote_ids=["QT001"],
        ),
        Claim(
            claim_id="Q2-C1",
            subquestion="Q2",
            claim_text="Whisper 是 OpenAI 開發的語音識別系統",  # identical text, diff SQ
            source_ids=["S002"],
            quote_ids=["QT002"],
        ),
    ]
    result = p2._dedup_approved_claims(claims)
    ids = [c.claim_id for c in result]
    # Different SQs — both kept
    assert "Q1-C1" in ids
    assert "Q2-C1" in ids


def test_phase2_imports_is_near_duplicate():
    import deep_research.nodes.phase2 as p2
    from deep_research.harness.claim_dedup import is_near_duplicate
    assert hasattr(p2, "is_near_duplicate") or "is_near_duplicate" in inspect.getsource(p2)


def test_phase2_integrate_calls_dedup():
    import deep_research.nodes.phase2 as p2
    src = inspect.getsource(p2.phase2_integrate)
    assert "_dedup_approved_claims" in src
