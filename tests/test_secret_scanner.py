"""Tests for deep_research/harness/secret_scanner.py.

Covers:
- All 8+ secret patterns detect correctly (positive cases)
- Realistic text (URLs, code, long identifiers) doesn't trigger false positives
- Redact preserves surrounding text and replaces with [REDACTED_<TYPE>]
- Multiple secrets + overlapping patterns resolved correctly
"""

import pytest

from deep_research.harness.secret_scanner import (
    Secret,
    contains_secret,
    redact_secrets,
    scan_secrets,
)


# ---------------------------------------------------------------------------
# Positive cases — each pattern detects its target
# ---------------------------------------------------------------------------

class TestPositiveDetection:
    def test_discord_webhook(self):
        # Split to avoid GitHub push-protection matching a literal webhook.
        url = "https://discord" + ".com/api/webhooks/1234567890/abcdefghij_ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456"
        secrets = scan_secrets(f"webhook: {url}")
        assert len(secrets) == 1
        assert secrets[0].type == "DISCORD_WEBHOOK"
        assert secrets[0].raw == url

    def test_discord_webhook_discordapp_variant(self):
        url = "https://discordapp" + ".com/api/webhooks/1234567890/abcdefghij_ABCDEFGHIJKLMNOPQRSTUVWXYZ_123"
        assert contains_secret(url)

    def test_slack_webhook(self):
        url = "https://hooks." + "slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        secrets = scan_secrets(url)
        assert len(secrets) == 1
        assert secrets[0].type == "SLACK_WEBHOOK"

    def test_anthropic_key(self):
        key = "sk-ant-api03-abcdef0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ-hello_world"
        secrets = scan_secrets(f"ANTHROPIC_API_KEY={key}")
        assert len(secrets) == 1
        assert secrets[0].type == "ANTHROPIC_KEY"

    def test_openai_project_key(self):
        key = "sk-proj-" + "A" * 60
        secrets = scan_secrets(key)
        assert len(secrets) == 1
        assert secrets[0].type == "OPENAI_PROJECT_KEY"

    def test_openai_generic_key(self):
        key = "sk-" + "a" * 50
        secrets = scan_secrets(key)
        assert len(secrets) == 1
        assert secrets[0].type == "OPENAI_KEY"

    def test_google_api_key(self):
        key = "AIzaSyA" + "B" * 33  # AIza + 35 chars
        secrets = scan_secrets(f"key={key}&other=foo")
        assert len(secrets) == 1
        assert secrets[0].type == "GOOGLE_KEY"

    def test_github_pat_classic(self):
        key = "ghp_" + "a" * 36
        secrets = scan_secrets(f"export GITHUB_TOKEN={key}")
        assert len(secrets) == 1
        assert secrets[0].type == "GITHUB_PAT"

    def test_github_pat_fine_grained(self):
        key = "github_pat_" + "A" * 50
        secrets = scan_secrets(key)
        assert len(secrets) == 1
        assert secrets[0].type == "GITHUB_PAT_FINE"

    def test_aws_access_key_id(self):
        key = "AKIAIOSFODNN7EXAMPLE"  # 20 chars, AKIA + 16 upper
        secrets = scan_secrets(f"aws_access_key_id = {key}")
        assert any(s.type == "AWS_ACCESS_KEY_ID" for s in secrets)

    def test_aws_secret_key(self):
        text = 'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        secrets = scan_secrets(text)
        assert any(s.type == "AWS_SECRET" for s in secrets)

    def test_private_key_pem(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA1234567890abcdef==\n"
            "-----END RSA PRIVATE KEY-----"
        )
        secrets = scan_secrets(pem)
        assert len(secrets) == 1
        assert secrets[0].type == "PRIVATE_KEY_PEM"


# ---------------------------------------------------------------------------
# Negative cases — realistic text must not trigger false positives
# ---------------------------------------------------------------------------

class TestNegativeCases:
    def test_plain_github_url_not_flagged(self):
        text = "See https://github.com/user/repo/blob/main/README.md"
        assert scan_secrets(text) == []

    def test_short_sk_prefix_not_flagged(self):
        # 'sk-' followed by < 40 chars should not match (too short to be a key)
        text = "use sk-dev as the prefix for keys"
        assert scan_secrets(text) == []

    def test_akia_as_word_part_not_flagged(self):
        # 'AKIA' appearing as part of a longer word (not 4 + 16 upper) should not match
        text = "AKIASomethingLowercase"
        assert scan_secrets(text) == []

    def test_arxiv_id_not_flagged(self):
        # Realistic brief content — no secrets
        text = (
            "Check arxiv.org/abs/2310.05193 for the ResearchAgent paper. "
            "We also looked at 2504.01222 and 2508.00031."
        )
        assert scan_secrets(text) == []

    def test_aiza_short_suffix_not_flagged(self):
        # AIza prefix but suffix length wrong
        text = "AIzaTooShort"
        assert scan_secrets(text) == []

    def test_ordinary_code_not_flagged(self):
        text = (
            "def fetch_url(url: str) -> str:\n"
            "    response = httpx.get(url, timeout=3.0)\n"
            "    return response.text\n"
        )
        assert scan_secrets(text) == []


# ---------------------------------------------------------------------------
# Redaction behaviour
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redact_replaces_with_placeholder(self):
        url = "https://discord" + ".com/api/webhooks/1234567890/abcdef_ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567"
        redacted, secrets = redact_secrets(f"notify: {url} end")
        assert redacted == "notify: [REDACTED_DISCORD_WEBHOOK] end"
        assert len(secrets) == 1

    def test_redact_preserves_no_secret_text(self):
        text = "nothing to see here"
        redacted, secrets = redact_secrets(text)
        assert redacted == text
        assert secrets == []

    def test_redact_multiple_secrets_same_text(self):
        webhook = "https://discord" + ".com/api/webhooks/1234567890/" + "a" * 60
        text = (
            f"webhook={webhook} "
            f"key=sk-ant-{'x' * 60} "
            f"gh={'ghp_' + 'y' * 36}"
        )
        redacted, secrets = redact_secrets(text)
        assert "[REDACTED_DISCORD_WEBHOOK]" in redacted
        assert "[REDACTED_ANTHROPIC_KEY]" in redacted
        assert "[REDACTED_GITHUB_PAT]" in redacted
        assert len(secrets) == 3

    def test_redact_preserves_surrounding_punctuation(self):
        key = "sk-ant-" + "z" * 60
        redacted, _ = redact_secrets(f"(key={key}).")
        assert redacted == "(key=[REDACTED_ANTHROPIC_KEY])."

    def test_redact_same_secret_twice(self):
        key = "ghp_" + "b" * 36
        redacted, secrets = redact_secrets(f"{key} and again {key}")
        assert redacted == "[REDACTED_GITHUB_PAT] and again [REDACTED_GITHUB_PAT]"
        assert len(secrets) == 2


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------

class TestApi:
    def test_secret_placeholder_shape(self):
        s = Secret(type="DISCORD_WEBHOOK", start=0, end=10, raw="xxx")
        assert s.placeholder == "[REDACTED_DISCORD_WEBHOOK]"

    def test_contains_secret_true(self):
        assert contains_secret("ghp_" + "a" * 36) is True

    def test_contains_secret_false(self):
        assert contains_secret("no secret here") is False

    def test_scan_secrets_sorted_by_offset(self):
        # Two secrets, put the later one first in text
        text = f"{'sk-ant-' + 'a' * 60} then {'ghp_' + 'b' * 36}"
        secrets = scan_secrets(text)
        assert len(secrets) == 2
        assert secrets[0].start < secrets[1].start

    def test_empty_input(self):
        assert scan_secrets("") == []
        assert redact_secrets("") == ("", [])
