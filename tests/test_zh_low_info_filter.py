"""Tests for Whisper plan P1-5 — low-info zh-TW / zh-CN source filter.

The failed-workspace analysis (2026-04-14) showed 84.5% of sources came from
content-mill blogs publishing ``2026 X 趨勢 / 你必須知道`` listicles. These
domains are not academic, not first-party, and not reviewed engineering
writeups — they're marketing copy.

The fix is to classify them as T5 (same bucket as UGC forums) so that:
- They still get fetched when nothing better is available;
- They never outrank genuine T1–T3 sources in ``tier_rank`` ordering;
- Domain-bias logging still catches over-concentration from one of them.
"""

from __future__ import annotations

from deep_research.harness.source_tier import classify_tier


def test_yahoo_tw_news_is_t5():
    assert classify_tier("https://tw.news.yahoo.com/some-2026-trend-article") == "T5"


def test_yahoo_stock_tw_is_t5():
    assert classify_tier("https://tw.stock.yahoo.com/news/xyz.html") == "T5"


def test_yahoo_news_is_t5():
    assert classify_tier("https://news.yahoo.com/article-123.html") == "T5"


def test_ibm_blog_is_t5():
    """ibm.com marketing / blog posts are low-signal; classify as T5.

    Note: ``developer.ibm.com`` / ``docs.ibm.com`` still hit the T1 subdomain
    prefix first, so genuine IBM developer docs are unaffected.
    """
    assert classify_tier("https://www.ibm.com/blog/2026-ai-trends") == "T5"
    assert classify_tier("https://ibm.com/topics/agents") == "T5"


def test_ibm_developer_subdomain_stays_t1():
    """Guard against over-filtering: developer.ibm.com is genuine first-party
    docs and must remain T1 via the subdomain-prefix rule."""
    assert classify_tier("https://developer.ibm.com/articles/foo/") == "T1"


def test_kknews_is_t5():
    assert classify_tier("https://kknews.cc/tech/xyzabc.html") == "T5"


def test_toutiao_is_t5():
    assert classify_tier("https://www.toutiao.com/a6123456789/") == "T5"


def test_sohu_is_t5():
    assert classify_tier("https://www.sohu.com/a/678901234_123456") == "T5"


def test_sina_cn_is_t5():
    assert classify_tier("https://tech.sina.com.cn/roll/2026-01-01/doc-abc.shtml") == "T5"


def test_cnbeta_is_t5():
    assert classify_tier("https://www.cnbeta.com/articles/tech/1234567.htm") == "T5"


def test_generic_com_still_t4():
    """Sanity: unrelated generic .com domains remain T4, not swept up in the
    blacklist."""
    assert classify_tier("https://www.some-random-blog.com/post/1") == "T4"


def test_iask_ca_is_t5():
    assert classify_tier("https://iask.ca/news/some-article.html") == "T5"
