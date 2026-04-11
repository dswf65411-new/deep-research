"""LLM provider factory — Claude / OpenAI / Gemini, auto-detect by key availability.

Model 和 context window 均為 hardcode，定期人工更新。
每個 provider 有兩個 tier：
  - "strong": 主要推理（Phase 0 規劃、Phase 1a 搜尋、Phase 2 整合、Phase 3 審計+報告）
  - "fast":   sub-task（Phase 1b 攻擊式事實核查）
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# Module-level state, set by main.py at startup
_provider: str = "claude"
_context_threshold: float = 0.3  # 預設 30%，使用者可透過 --context-threshold 覆蓋

# ---------------------------------------------------------------------------
# Model 配置（hardcode，定期更新）
# ---------------------------------------------------------------------------
#
# 更新時機：當各家推出新的 stable model 時，手動更新此處。
# 更新方式：確認新 model 的 context window（官方文件或 API 查詢），更新下方對應欄位。
#
# 選擇原則：
#   strong — 品質最高的 stable model（不選 preview），用於需要判斷力的任務
#   fast   — 速度快、成本低的 model，用於機械性核對任務
#
# 最後更新：2026-04-12

MODELS = {
    "claude": {
        "strong": {"model": "claude-sonnet-4-6", "context_limit": 1_000_000},
        "fast": {"model": "claude-haiku-4-5-20251001", "context_limit": 200_000},
    },
    "openai": {
        "strong": {"model": "gpt-4o", "context_limit": 128_000},
        "fast": {"model": "gpt-4o-mini", "context_limit": 128_000},
    },
    "gemini": {
        "strong": {"model": "gemini-2.5-pro", "context_limit": 1_048_576},
        "fast": {"model": "gemini-2.5-flash", "context_limit": 1_048_576},
    },
}

KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# ---------------------------------------------------------------------------
# Context Window 管理策略
# ---------------------------------------------------------------------------
#
# 問題：LLM 的 context window 有上限，且即使塞得下，填充率越高注意力越分散。
#   - 研究中需要核對多個 claim × 多個 source（多針場景），對精度要求極高。
#   - Lost in the Middle (Liu et al. 2023)：正確答案放在 20 篇文件中間時，
#     GPT-3.5/Claude-1.3 準確率從頭尾 ~70% 掉到中間 ~40%（U 型注意力曲線）。
#   - RULER (Hsieh et al. 2024)：多針檢索任務中，大部分模型在
#     context 填充到標稱長度 25-50% 時就開始能力下降。
#   - 2024-2025 新模型（Claude 3.5+, GPT-4o, Gemini 2.5）改善了單針表現，
#     但多針場景的公開 benchmark 仍然稀缺，無法確定精確安全閾值。
#
# 策略：Iterative Refinement (Incremental Summarization)
#   不截斷、不丟資料。決策流程：
#
#   Step 0 — 估算 input tokens:
#     粗估：len(text) // 3（中文約 1-2 tokens/字，取保守值）。
#     計算 total_tokens = tokens(fixed_prompt + draft + all_sources)。
#
#   Step 1 — 全塞判斷:
#     budget = current_model_limit × threshold
#     若 total_tokens < budget → 全塞 context，不走迴圈。
#     全塞時 LLM 同時看到所有 sources，能做跨文件比對，品質最高，無累積偏差。
#
#   Step 2 — Iterative Refinement loop:
#     total_tokens ≥ budget 時，進入迴圈模式：
#
#     2a. 每輪開始時計算：
#         fixed_cost = tokens(fixed_prompt + draft)
#         remaining = budget - fixed_cost
#
#     2b. 若 remaining > 0（正常情況）：
#         貪心法塞 sources 到 remaining 滿為止（BM25 排序後依序塞）。
#         draft = LLM(fixed_prompt + draft + source_batch)
#         新資訊放尾部、draft 放前面 → 利用 prompt prefix caching 降低成本。
#
#     2c. 若 remaining ≤ 0（fixed_cost 本身就超過 budget）：
#         不 panic，退化為一篇一篇送。每輪只送一篇 source，
#         超出 threshold 但仍在 context limit 內，精度換取可用性。
#
#     2d. 若 fixed_cost + 單篇 source > 當前 model 的 100% context limit：
#         自動切換到使用者可用的 context window 最大的 provider
#         （find_largest_available_provider），重算 budget 再繼續。
#         例：GPT-4o 128K 塞不下 → 切 Gemini 1M。
#         若切換後仍超過 100% → raise error，由使用者介入處理。
#
#     學術參考：
#       - RAPTOR (ICLR 2024)：遞歸摘要樹，從底層 chunk 向上合併
#       - Chain-of-Density (Adams et al. 2023)：多輪迭代讓摘要越來越密
#       - MemWalker (Chen et al. 2023)：LLM 在文本樹上迭代走訪收集資訊
#     相比一次性 MapReduce，Iterative Refinement 的優勢在於每步 LLM 都能
#     看到完整的累積 draft，決策品質更高；且利用 prefix caching 攤平成本。
#
#   Step 3 — BM25 排序（在 Step 2 內部使用）:
#     sources 數量多時，用 BM25 + LLM Query Expansion 決定每批的優先順序，
#     確保最相關的 sources 先被處理。Query Expansion 讓 LLM 根據研究主題
#     生成加長版查詢（同義詞、多語言對應詞），提升 BM25 的 recall。
#     大 chunk + 長 query = BM25 的 TF-IDF 統計量更穩定，語意不匹配問題小。
#
# 填充率閾值（_context_threshold）：
#   預設 0.3 (30%)。使用者可透過 CLI --context-threshold 調整。
#   30% 是保守值，未來可透過 A/B 測試（同資料全塞 vs iterative）校準。
#
# 實作位於 deep_research/context.py（TODO）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider / Threshold 設定
# ---------------------------------------------------------------------------

def auto_detect_provider() -> str:
    """Auto-detect LLM provider by key availability. Priority: claude > gemini > openai."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "找不到任何 LLM API key。請在 .env 中設定至少一個：\n"
        "  ANTHROPIC_API_KEY（Claude）\n"
        "  GEMINI_API_KEY（Gemini）\n"
        "  OPENAI_API_KEY（OpenAI）\n"
        "執行 make init 重新設定。"
    )


def _detect_provider_from_model_name(model_name: str) -> str:
    """Infer provider from a full model name like 'gemini-2.5-pro' or 'gpt-4o'."""
    if model_name.startswith("gemini") or model_name.startswith("gemma"):
        return "gemini"
    elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
        return "openai"
    elif model_name.startswith("claude"):
        return "claude"
    raise ValueError(
        f"無法從模型名稱 '{model_name}' 判斷 provider。"
        f"請用 claude/gemini/openai 或完整模型版號（如 gemini-2.5-pro）。"
    )


def set_provider(provider_or_model: str) -> None:
    """Set the global LLM provider. Called once at startup.

    支援三種輸入：
      1. "auto" — 按 claude > gemini > openai 順序偵測 key
      2. "claude" / "gemini" / "openai" — 用該家 hardcode 的模型
      3. 完整模型版號如 "gemini-2.5-pro" — 直接用這個 model（覆蓋 strong tier）
    """
    global _provider

    provider_names = set(MODELS.keys())

    if provider_or_model == "auto":
        _provider = auto_detect_provider()

    elif provider_or_model in provider_names:
        key_name = KEY_ENV[provider_or_model]
        if not os.environ.get(key_name):
            raise RuntimeError(f"選擇了 --model {provider_or_model} 但 {key_name} 未設定。請在 .env 中加入。")
        _provider = provider_or_model

    else:
        # 完整模型版號 → 推斷 provider，覆蓋 strong tier
        detected = _detect_provider_from_model_name(provider_or_model)
        key_name = KEY_ENV[detected]
        if not os.environ.get(key_name):
            raise RuntimeError(f"模型 {provider_or_model} 需要 {key_name}，但未設定。請在 .env 中加入。")
        _provider = detected
        # 覆蓋 strong model name（context_limit 維持 hardcode 值）
        MODELS[_provider]["strong"]["model"] = provider_or_model
        logger.info(f"User override: {_provider}/strong → {provider_or_model}")

    logger.info(
        f"Provider: {_provider} | "
        f"strong={MODELS[_provider]['strong']['model']} "
        f"(ctx: {MODELS[_provider]['strong']['context_limit']:,}) | "
        f"fast={MODELS[_provider]['fast']['model']} "
        f"(ctx: {MODELS[_provider]['fast']['context_limit']:,})"
    )


def get_provider() -> str:
    return _provider


def set_context_threshold(threshold: float) -> None:
    """Set the context fill-rate threshold (0.0 ~ 1.0).

    當 input tokens 超過 model context limit × threshold 時，
    切換為 Iterative Refinement 模式而非全塞。
    預設 0.3 (30%)。使用者可透過 CLI --context-threshold 覆蓋。
    """
    global _context_threshold
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"context-threshold 必須在 (0.0, 1.0] 之間，收到: {threshold}")
    _context_threshold = threshold


def get_context_threshold() -> float:
    return _context_threshold


# ---------------------------------------------------------------------------
# LLM 實例化
# ---------------------------------------------------------------------------

def get_llm(
    tier: str = "strong",
    max_tokens: int = 8192,
    temperature: float = 0.2,
    provider: str | None = None,
) -> BaseChatModel:
    """Return a chat model instance.

    Args:
        tier: "strong" (主力推理) or "fast" (sub-task)
        max_tokens: max output tokens
        temperature: sampling temperature
        provider: override provider (default: use global _provider)
    """
    p = provider or _provider
    model_name = MODELS[p][tier]["model"]

    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider: {p}")


# ---------------------------------------------------------------------------
# Context limit 查詢
# ---------------------------------------------------------------------------

def get_context_limit(provider: str | None = None, tier: str = "strong") -> int:
    """Return the context window limit (input tokens) for a provider+tier."""
    p = provider or _provider
    return MODELS[p][tier]["context_limit"]


def find_largest_available_provider(tier: str = "strong") -> str | None:
    """Find the available provider with the largest context window.

    Skips the current provider. Returns None if no other provider has a key.
    """
    candidates = []
    for provider in MODELS:
        if provider == _provider:
            continue
        key_name = KEY_ENV.get(provider, "")
        if not os.environ.get(key_name):
            continue
        limit = MODELS[provider][tier]["context_limit"]
        candidates.append((limit, provider))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def get_model_info(provider: str | None = None, tier: str = "strong") -> dict:
    """Return full model info: {"model": "...", "context_limit": N}."""
    p = provider or _provider
    return dict(MODELS[p][tier])
