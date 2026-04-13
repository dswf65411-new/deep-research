"""Tests for P1-D: Off-topic metadata claim filter.

Verifies:
- _is_metadata_claim correctly identifies company addresses, contact info, boilerplate
- Legitimate topical claims are NOT filtered
- validate_claims_for_phase2 applies the metadata filter
"""

import pytest

from deep_research.harness.validators import _is_metadata_claim, validate_claims_for_phase2
from deep_research.state import Claim


# ---------------------------------------------------------------------------
# _is_metadata_claim — should detect off-topic metadata
# ---------------------------------------------------------------------------

class TestIsMetadataClaim:
    # --- Street addresses (English) ---

    def test_en_street_address_street(self):
        assert _is_metadata_claim("Verbit is located at 460 Park Avenue, New York, NY 10022")

    def test_en_street_address_ave(self):
        assert _is_metadata_claim("Company headquarters: 1234 Oak Avenue, Suite 200")

    def test_en_street_address_road(self):
        assert _is_metadata_claim("Our office is at 99 Broadmoor Road, Boston")

    def test_us_zip_code(self):
        assert _is_metadata_claim("San Francisco, CA 94105 — visit us anytime")

    # --- Address context keywords ---

    def test_addr_context_headquartered_at(self):
        assert _is_metadata_claim("The company is headquartered at 500 Oracle Pkwy, Redwood City")

    def test_addr_context_located_at(self):
        assert _is_metadata_claim("Whisper LLC is located at 120 West 45th Street")

    def test_addr_context_office_in(self):
        assert _is_metadata_claim("We have offices in New York, London, and Tokyo")

    # --- SEO / contact boilerplate ---

    def test_contact_us(self):
        assert _is_metadata_claim("Contact us today to learn more about our services")

    def test_phone_number(self):
        assert _is_metadata_claim("Call us at 800-555-1234 for support")

    def test_sign_up_free(self):
        assert _is_metadata_claim("Sign up for free and start transcribing today")

    def test_team_of_experts(self):
        assert _is_metadata_claim("Our team of experts is ready to help with your transcription needs")

    # --- Legal boilerplate ---

    def test_privacy_policy(self):
        assert _is_metadata_claim("See our Privacy Policy for details on data handling")

    def test_copyright(self):
        assert _is_metadata_claim("© 2025 Acme Corp. All rights reserved.")

    def test_terms_of_service(self):
        assert _is_metadata_claim("By using this site you agree to our Terms of Service")

    # ---------------------------------------------------------------------------
    # Legitimate claims — must NOT be filtered
    # ---------------------------------------------------------------------------

    def test_wer_performance_claim(self):
        assert not _is_metadata_claim("Whisper Large v3 在中文語料上的 WER 為 8.3%")

    def test_model_comparison_claim(self):
        assert not _is_metadata_claim("Breeze ASR 25 在台語辨識上優於 OpenAI Whisper")

    def test_diarization_claim(self):
        assert not _is_metadata_claim("pyannote-audio 3.1 支援最多 20 位說話者分離")

    def test_platform_support_claim(self):
        assert not _is_metadata_claim("macOS 14 以上版本才支援 Apple Intelligence 語音轉錄")

    def test_price_claim(self):
        assert not _is_metadata_claim("MacWhisper Pro 的年訂閱價格為 USD 49.99")

    def test_founded_year_product_context(self):
        # "founded" in a product-launch context is borderline — should NOT be filtered
        # (the filter targets *company meta* not product release dates)
        # This claim has no address pattern so it passes
        assert not _is_metadata_claim("Whisper 於 2022 年由 OpenAI 開源發布")

    def test_taiwan_tool_claim(self):
        assert not _is_metadata_claim("雅婷轉錄器針對台灣繁體中文語境優化，支援粵語和閩南語辨識")

    def test_privacy_feature_claim(self):
        # Privacy as a feature (not legal boilerplate)
        assert not _is_metadata_claim("本機端處理模式可確保語音資料不上傳至雲端，隱私更安全")


# ---------------------------------------------------------------------------
# validate_claims_for_phase2 — metadata filter integrated into gate
# ---------------------------------------------------------------------------

def _make_claim(claim_id: str, text: str, bedrock: float = 0.7) -> Claim:
    c = Claim(claim_id=claim_id, claim_text=text, source_ids=["S001"])
    c.status = "approved"
    c.bedrock_score = bedrock
    c.quote_ids = ["S001-Q1"]
    return c


def test_address_claim_rejected_by_phase2_gate():
    """A company address claim with good bedrock must still be blocked by metadata filter."""
    addr = _make_claim("Q1-C1", "Verbit is located at 460 Park Avenue, New York, NY 10022")
    result = validate_claims_for_phase2([addr])
    assert result == [], f"Expected address claim to be rejected, got {result}"


def test_legitimate_claim_passes_phase2_gate():
    """A topical claim must not be filtered by the metadata heuristic."""
    good = _make_claim("Q1-C2", "Whisper Large v3 在中文語料上的 WER 為 8.3%")
    result = validate_claims_for_phase2([good])
    assert len(result) == 1


def test_mix_of_claims_filtered_correctly():
    """Only metadata claims are removed; topical claims pass through."""
    topical = _make_claim("Q1-C1", "MacWhisper Pro 支援本機端 Whisper 模型，無需網路")
    addr = _make_claim("Q1-C2", "MacWhisper Ltd., 55 Baker Street, London")
    boilerplate = _make_claim("Q1-C3", "Contact us today to start your free trial")
    another_good = _make_claim("Q1-C4", "pyannote 3.1 speaker diarization 平均 DER 為 12%")

    result = validate_claims_for_phase2([topical, addr, boilerplate, another_good])
    ids = {c.claim_id for c in result}
    assert "Q1-C1" in ids
    assert "Q1-C4" in ids
    assert "Q1-C2" not in ids, "Address claim must be filtered"
    assert "Q1-C3" not in ids, "SEO boilerplate claim must be filtered"
