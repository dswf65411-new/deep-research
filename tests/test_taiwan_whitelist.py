"""Smoke tests for Taiwan Whitelist + serper_tw enforcement — issue #3 fix.

Verifies:
- _TAIWAN_DOMAIN_WHITELIST contains expected Taiwan domains
- _PLANNER_SYSTEM enforces zh-TW → serper_tw only rule
- _PLANNER_SYSTEM has Taiwan site: locking section
- brave_search accepts country and search_lang optional params
"""

import inspect

import pytest


# ---------------------------------------------------------------------------
# _TAIWAN_DOMAIN_WHITELIST
# ---------------------------------------------------------------------------

def test_taiwan_whitelist_exists():
    from deep_research.nodes.phase1a import _TAIWAN_DOMAIN_WHITELIST
    assert isinstance(_TAIWAN_DOMAIN_WHITELIST, frozenset)
    assert len(_TAIWAN_DOMAIN_WHITELIST) > 0


def test_taiwan_whitelist_core_domains():
    from deep_research.nodes.phase1a import _TAIWAN_DOMAIN_WHITELIST
    for domain in ("ithome.com.tw", "techbang.com", "mobile01.com"):
        assert domain in _TAIWAN_DOMAIN_WHITELIST, (
            f"Expected '{domain}' in _TAIWAN_DOMAIN_WHITELIST"
        )


def test_taiwan_whitelist_has_ithelp():
    from deep_research.nodes.phase1a import _TAIWAN_DOMAIN_WHITELIST
    assert "ithelp.ithome.com.tw" in _TAIWAN_DOMAIN_WHITELIST


def test_taiwan_whitelist_no_english_only_domains():
    """Whitelist must not contain generic English-only domains."""
    from deep_research.nodes.phase1a import _TAIWAN_DOMAIN_WHITELIST
    for domain in ("techcrunch.com", "wired.com", "theverge.com"):
        assert domain not in _TAIWAN_DOMAIN_WHITELIST


# ---------------------------------------------------------------------------
# _PLANNER_SYSTEM — zh-TW → serper_tw rule
# ---------------------------------------------------------------------------

def test_planner_system_enforces_zhtw_serper_tw():
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "serper_tw" in _PLANNER_SYSTEM
    # Must state zh-TW queries must only use serper_tw
    assert "zh-TW" in _PLANNER_SYSTEM and "serper_tw" in _PLANNER_SYSTEM


def test_planner_system_no_mixing_rule():
    """Must state that zh-TW must NOT mix with brave/serper_en."""
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "不可加 brave" in _PLANNER_SYSTEM or "不可與 brave" in _PLANNER_SYSTEM


def test_planner_system_serper_tw_exclusive_note():
    """serper_tw engine description must note it's exclusive to zh-TW queries."""
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "zh-TW query 專用" in _PLANNER_SYSTEM


# ---------------------------------------------------------------------------
# _PLANNER_SYSTEM — Taiwan site: locking section
# ---------------------------------------------------------------------------

def test_planner_system_has_taiwan_locking_section():
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "台灣來源鎖定" in _PLANNER_SYSTEM


def test_planner_system_has_ithome_site():
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "site:ithome.com.tw" in _PLANNER_SYSTEM


def test_planner_system_has_apple_site():
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    assert "site:apps.apple.com" in _PLANNER_SYSTEM


def test_planner_system_taiwan_queries_use_serper_tw():
    """Taiwan site: section must instruct to use serper_tw exclusively."""
    from deep_research.nodes.phase1a import _PLANNER_SYSTEM
    # Both the section header and the engine rule must appear
    assert "台灣來源鎖定" in _PLANNER_SYSTEM
    # After the section header, serper_tw must be mentioned as the required engine
    section_start = _PLANNER_SYSTEM.index("台灣來源鎖定")
    section_text = _PLANNER_SYSTEM[section_start:section_start + 600]
    assert "serper_tw" in section_text


# ---------------------------------------------------------------------------
# brave_search — country and search_lang params
# ---------------------------------------------------------------------------

def test_brave_search_signature_has_country():
    from deep_research.tools.search import brave_search
    sig = inspect.signature(brave_search)
    assert "country" in sig.parameters, "brave_search must accept 'country' param"


def test_brave_search_signature_has_search_lang():
    from deep_research.tools.search import brave_search
    sig = inspect.signature(brave_search)
    assert "search_lang" in sig.parameters, "brave_search must accept 'search_lang' param"


def test_brave_search_country_default_empty():
    from deep_research.tools.search import brave_search
    sig = inspect.signature(brave_search)
    assert sig.parameters["country"].default == ""


def test_brave_search_search_lang_default_empty():
    from deep_research.tools.search import brave_search
    sig = inspect.signature(brave_search)
    assert sig.parameters["search_lang"].default == ""
