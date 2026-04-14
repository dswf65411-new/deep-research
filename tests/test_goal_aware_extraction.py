"""Verify the extractor's goal-aware gate (Tongyi-DeepResearch-style).

Three invariants:
1. When sq_text is provided, the LLM user_msg must contain a Research Goal
   section with the goal body (not just the SQ label).
2. A NOT_RELEVANT rational short-circuits to empty + off_topic=True.
3. A relevant page passes through normally with claims preserved.
"""

import asyncio

from deep_research.nodes import phase1a


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


def test_sq_text_is_injected_into_user_msg(monkeypatch):
    captured = {}

    async def fake_ainvoke(role, messages, max_tokens, temperature):
        captured["user_msg"] = messages[1].content
        return _FakeResp(
            '{"rational":"The source discusses LangGraph supervisor routing in detail.",'
            '"quotes":[],"numbers":[],"claims":[]}'
        )

    monkeypatch.setattr(phase1a, "safe_ainvoke_chain", fake_ainvoke)

    src = {
        "source_id": "S001",
        "url": "https://example.com",
        "title": "Example",
        "subquestion": "Q1",
        "role": "advocate",
    }
    asyncio.run(
        phase1a._extract_one_pass(src, "content text", sq_text="How does Supervisor route failures?")
    )

    assert "Research Goal" in captured["user_msg"]
    assert "How does Supervisor route failures?" in captured["user_msg"]
    assert "Q1: How does Supervisor route failures?" in captured["user_msg"]


def test_missing_sq_text_omits_goal_section(monkeypatch):
    """Backward compatibility: with sq_text='' the extractor behaves like before
    (no goal section, no relevance gate pressure from a stray Goal header)."""
    captured = {}

    async def fake_ainvoke(role, messages, max_tokens, temperature):
        captured["user_msg"] = messages[1].content
        return _FakeResp('{"rational":"ok","quotes":[],"numbers":[],"claims":[]}')

    monkeypatch.setattr(phase1a, "safe_ainvoke_chain", fake_ainvoke)

    src = {
        "source_id": "S001",
        "url": "https://example.com",
        "title": "Example",
        "subquestion": "Q1",
        "role": "advocate",
    }
    asyncio.run(phase1a._extract_one_pass(src, "content text"))

    assert "Research Goal" not in captured["user_msg"]


def test_not_relevant_short_circuits_to_off_topic(monkeypatch, tmp_path):
    async def fake_ainvoke(role, messages, max_tokens, temperature):
        return _FakeResp(
            '{"rational":"NOT_RELEVANT — this is a marketing blog about TTS, '
            'not about LangGraph supervisor routing.",'
            '"quotes":[{"quote_id":"Q1","text":"hi","start":0,"end":2}],'
            '"numbers":[],"claims":[{"claim_text":"TTS is nice","claim_type":"qualitative",'
            '"evidence_quote_ids":["Q1"],"evidence_number_ids":[]}]}'
        )

    monkeypatch.setattr(phase1a, "safe_ainvoke_chain", fake_ainvoke)

    src = {
        "source_id": "S001",
        "url": "https://murf.ai/blog",
        "title": "Best TTS 2026",
        "subquestion": "Q1",
        "role": "advocate",
        "content": "Murf AI text-to-speech blog post content here.",
        "status": "LIVE",
    }
    result = asyncio.run(
        phase1a._extract_one(src, str(tmp_path), sq_text="LangGraph supervisor routing")
    )

    assert result is not None
    assert result.get("off_topic") is True
    assert result["quotes"] == []
    assert result["numbers"] == []
    assert result["claims"] == []


def test_relevant_page_passes_through(monkeypatch, tmp_path):
    async def fake_ainvoke(role, messages, max_tokens, temperature):
        return _FakeResp(
            '{"rational":"Section 3 directly describes supervisor error handling.",'
            '"quotes":[{"quote_id":"Q1","text":"Supervisor routes","start":0,"end":17}],'
            '"numbers":[],'
            '"claims":[{"claim_text":"Supervisor routes failures","claim_type":"qualitative",'
            '"evidence_quote_ids":["Q1"],"evidence_number_ids":[]}]}'
        )

    monkeypatch.setattr(phase1a, "safe_ainvoke_chain", fake_ainvoke)

    src = {
        "source_id": "S042",
        "url": "https://langchain-ai.github.io/langgraph/",
        "title": "LangGraph Supervisor",
        "subquestion": "Q1",
        "role": "advocate",
        "content": "Supervisor routes failures back to workers based on...",
        "status": "LIVE",
        "engines": ["serper_web"],
        "fetch_method": "serper",
    }
    result = asyncio.run(
        phase1a._extract_one(src, str(tmp_path), sq_text="How does Supervisor route failures?")
    )

    assert result is not None
    assert not result.get("off_topic")
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claim_text"] == "Supervisor routes failures"
