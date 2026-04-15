"""Smoke tests for Source Tier Classification — issue #5 fix.

Verifies:
- classify_tier returns correct T1-T5 for representative domains
- T2 .edu / .ac.* TLD matching works
- T3 Taiwan whitelist is correctly classified
- T3 English media domains are correctly classified
- T5 UGC/community domains are correctly classified
- T4 is the default for unknown domains
- classify_tier is importable and used in phase1a + phase2
- tier_rank returns expected sort order
"""

import inspect
import pytest

from deep_research.harness.source_tier import classify_tier, tier_rank


# ---------------------------------------------------------------------------
# T1 — Official / first-party
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://support.apple.com/en-us/HT213186",
    "https://developer.apple.com/documentation/avfoundation",
    "https://openai.com/blog/whisper",
    "https://docs.python.org/3/",
    "https://huggingface.co/models",
    "https://platform.openai.com/docs",
    "https://developer.mozilla.org/en-US/docs/Web",
])
def test_t1_official_domains(url):
    assert classify_tier(url) == "T1", f"Expected T1 for {url}"


# ---------------------------------------------------------------------------
# T2 — Academic / peer-reviewed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://arxiv.org/abs/2501.12345",
    "https://aclanthology.org/2024.acl-main.1",
    "https://semanticscholar.org/paper/abc123",
    "https://ieeexplore.ieee.org/document/1234567",
    "https://proceedings.mlr.press/v202/paper.html",
])
def test_t2_academic_domains(url):
    assert classify_tier(url) == "T2", f"Expected T2 for {url}"


def test_t2_edu_tld():
    assert classify_tier("https://cs.stanford.edu/~pabbeel/papers/") == "T2"


def test_t2_ac_tld():
    assert classify_tier("https://www.ntu.ac.tw/research") == "T2"
    assert classify_tier("https://speech.ee.ntu.edu.tw/") == "T2"


# ---------------------------------------------------------------------------
# T2 — PapersWithCode + GitHub docs (Whisper plan P1-1)
# ---------------------------------------------------------------------------
#
# Rationale: the failed-workspace analysis showed T1/T2 coverage collapsed
# to 9% because high-signal research surfaces (PwC SOTA pages, repo READMEs,
# github.io project pages) were silently dumped into T4 alongside marketing
# blogs. These upgrades put them back into the high-quality tier that
# downstream ranking actually rewards.

@pytest.mark.parametrize("url", [
    "https://paperswithcode.com/sota/image-classification-on-imagenet",
    "https://paperswithcode.com/paper/gpt-4-technical-report",
])
def test_t2_paperswithcode(url):
    assert classify_tier(url) == "T2", f"PapersWithCode should be T2, got {classify_tier(url)}"


@pytest.mark.parametrize("url", [
    # repo root landing page is effectively the README
    "https://github.com/langchain-ai/langgraph",
    "https://github.com/langchain-ai/langgraph/",
    # explicit README blobs
    "https://github.com/langchain-ai/langgraph/blob/main/README.md",
    "https://github.com/langchain-ai/langgraph/blob/main/README",
    "https://github.com/langchain-ai/langgraph/blob/main/readme.md",
    # /docs/ tree and files
    "https://github.com/langchain-ai/langgraph/tree/main/docs",
    "https://github.com/langchain-ai/langgraph/blob/main/docs/architecture.md",
    # wiki
    "https://github.com/some/repo/wiki",
    "https://github.com/some/repo/wiki/Home",
])
def test_t2_github_docs(url):
    assert classify_tier(url) == "T2", f"GitHub docs URL should be T2: {url}"


@pytest.mark.parametrize("url", [
    "https://langchain-ai.github.io/langgraph/",
    "https://some-project.github.io/reference/api.html",
])
def test_t2_github_io(url):
    # Note: docs.*.github.io lands in T1 via the general docs. prefix rule,
    # which is fine — docs subdomains are already curated official pages.
    assert classify_tier(url) == "T2", f"github.io project pages should be T2: {url}"


@pytest.mark.parametrize("url", [
    # raw source code is not a research claim source
    "https://github.com/langchain-ai/langgraph/blob/main/src/langgraph/graph.py",
    "https://github.com/langchain-ai/langgraph/blob/main/tests/test_graph.py",
    # issues / pulls / actions are noisy, keep as T4
    "https://github.com/langchain-ai/langgraph/issues/1234",
    "https://github.com/langchain-ai/langgraph/pull/5678",
    # github.blog is editorial marketing, not docs
    "https://github.blog/2024-01-01-some-announcement/",
])
def test_github_source_and_meta_stay_t4(url):
    tier = classify_tier(url)
    assert tier == "T4", f"{url} should stay T4, got {tier}"


# ---------------------------------------------------------------------------
# T3 — Taiwan professional media
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.ithome.com.tw/review/1234",
    "https://ithelp.ithome.com.tw/articles/1234",
    "https://www.techbang.com/posts/12345",
    "https://www.kocpc.com.tw/archives/12345",
    "https://www.mobile01.com/topicdetail.php?f=1&t=1",
    "https://www.eprice.com.tw/mobile/view/1234",
    "https://inside.com.tw/articles/1234",
    "https://www.bnext.com.tw/article/1234",
])
def test_t3_taiwan_domains(url):
    assert classify_tier(url) == "T3", f"Expected T3 for {url}"


# ---------------------------------------------------------------------------
# T3 — English professional media
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.theverge.com/2025/1/1/article",
    "https://techcrunch.com/2025/01/01/story",
    "https://www.wired.com/story/something",
    "https://arstechnica.com/tech/2025",
    "https://engadget.com/article",
])
def test_t3_english_media(url):
    assert classify_tier(url) == "T3", f"Expected T3 for {url}"


# ---------------------------------------------------------------------------
# T4 — Default (unknown/generic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://brasstranscripts.com/review",
    "https://somerandomblog.com/post",
    "https://medium.com/@someone/article",
    "https://amical.software/features",
])
def test_t4_default(url):
    assert classify_tier(url) == "T4", f"Expected T4 for {url}"


def test_t4_empty_url():
    assert classify_tier("") == "T4"
    assert classify_tier("not-a-url") == "T4"


# ---------------------------------------------------------------------------
# T5 — UGC / community
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.reddit.com/r/MachineLearning/comments/abc",
    "https://stackoverflow.com/questions/1234",
    "https://www.quora.com/What-is-the-best",
    "https://www.ptt.cc/bbs/Tech_Job/M.html",
    "https://www.dcard.tw/f/tech/p/123456",
])
def test_t5_ugc(url):
    assert classify_tier(url) == "T5", f"Expected T5 for {url}"


# ---------------------------------------------------------------------------
# tier_rank ordering
# ---------------------------------------------------------------------------

def test_tier_rank_order():
    assert tier_rank("T1") < tier_rank("T2") < tier_rank("T3")
    assert tier_rank("T3") < tier_rank("T4") < tier_rank("T5")
    assert tier_rank("T5") < tier_rank("T6")


def test_tier_rank_unknown_maps_to_t4():
    assert tier_rank("TX") == tier_rank("T4")


# ---------------------------------------------------------------------------
# Integration: phase1a imports classify_tier
# ---------------------------------------------------------------------------

def test_phase1a_imports_classify_tier():
    import deep_research.nodes.phase1a as p1a
    assert hasattr(p1a, "classify_tier"), "phase1a must import classify_tier"


def test_phase1a_no_hardcoded_t4():
    """Source registry write must no longer hardcode tier='T4'."""
    import inspect
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._update_source_registry)
    assert '"T4"' not in src and "'T4'" not in src, (
        "_update_source_registry must not hardcode tier='T4'"
    )


def test_phase1a_build_sources_no_hardcoded_t4():
    import inspect
    import deep_research.nodes.phase1a as p1a
    src = inspect.getsource(p1a._build_sources)
    assert '"T4"' not in src and "'T4'" not in src, (
        "_build_sources must not hardcode tier='T4'"
    )


# ---------------------------------------------------------------------------
# Integration: phase2 imports tier_rank and passes sources
# ---------------------------------------------------------------------------

def test_phase2_imports_tier_rank():
    import deep_research.nodes.phase2 as p2
    assert hasattr(p2, "tier_rank"), "phase2 must import tier_rank"


def test_gather_source_texts_accepts_sources_param():
    import inspect
    import deep_research.nodes.phase2 as p2
    sig = inspect.signature(p2._gather_source_texts)
    assert "sources" in sig.parameters, "_gather_source_texts must accept 'sources' param"
