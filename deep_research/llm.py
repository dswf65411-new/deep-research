"""LLM provider factory — Claude / OpenAI / Gemini, auto-detect by key availability.

Model 和 context window 均為 hardcode，定期人工更新。
每個 provider 有兩個 tier：
  - "strong": 主要推理（Phase 0 規劃、Phase 1a 搜尋、Phase 2 整合、Phase 3 審計+報告）
  - "fast":   sub-task（Phase 1b 攻擊式事實核查）
"""

from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

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
# Rate limiting（主動節流，避免觸發 provider 的 per-minute 上限）
# ---------------------------------------------------------------------------
#
# 雙層防護：
#   1. InMemoryRateLimiter — client 側 token bucket，主動在 get_llm 時注入。
#      實例必須跨呼叫共享（module-level dict），不然每次 get_llm 都會新建 bucket
#      等於沒節流。
#   2. safe_ainvoke — tenacity 反應式 retry + asyncio.wait_for hard timeout。
#      Anthropic SDK 內建 max_retries 有時漏抓（LangChain #25309 hang bug），
#      這層補兜底。
#
# 速率設計：
#   Anthropic Tier 1 的 Sonnet 預設 30K ITPM。假設每次平均輸入 3K tokens：
#     30000 / 3000 = 10 req/min ≈ 0.17 req/s
#   保守抓 0.25 req/s（≈ 15 req/min），配 max_bucket_size=5 容許短時 burst。
#   Gemini 1M context、Google 預設 TPM 較寬，放寬到 0.5。
#   數值想調快就改下面的 dict。
_RATE_LIMITERS: dict[str, InMemoryRateLimiter] = {
    "claude": InMemoryRateLimiter(
        requests_per_second=3.0,   # Tier 2+ allows 1000 RPM; 3 req/s = 180 RPM
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    ),
    "gemini": InMemoryRateLimiter(
        requests_per_second=2.0,   # Flash quota generous; 2 req/s = 120 RPM
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    ),
    "openai": InMemoryRateLimiter(
        requests_per_second=1.0,
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    ),
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
        temperature: sampling temperature（Gemini 會被強制覆蓋為 1.0，見下方）
        provider: override provider (default: use global _provider)

    Note on temperature:
        Gemini 2.5 系列的 temperature 語意和 Claude/OpenAI 不同。
        預設值為 1.0，推理任務調低反而會退化品質（Google 官方建議）。
        因此不論呼叫端傳什麼，Gemini 一律覆蓋為 1.0。
        Claude/OpenAI 則遵守呼叫端指定的 temperature。
    """
    p = provider or _provider
    model_name = MODELS[p][tier]["model"]

    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limiter=_RATE_LIMITERS["claude"],
            max_retries=5,
        )

    elif p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limiter=_RATE_LIMITERS["openai"],
            max_retries=5,
        )

    elif p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Gemini 固定 1.0：調低會退化推理品質
        gemini_kwargs: dict = dict(
            model=model_name,
            max_tokens=max_tokens,
            temperature=1.0,
            rate_limiter=_RATE_LIMITERS["gemini"],
            max_retries=5,
        )
        # Flash 模型用於 structured JSON 抽取，不需要 thinking token
        # 啟用 thinking 時，幾乎所有 token budget 被 reasoning 吃掉，content 截斷
        if "flash" in model_name:
            gemini_kwargs["thinking_budget"] = 0
        return ChatGoogleGenerativeAI(**gemini_kwargs)

    raise ValueError(f"Unknown provider: {p}")


# ---------------------------------------------------------------------------
# Safe ainvoke（反應式 retry + hard timeout）
# ---------------------------------------------------------------------------

# 單次呼叫最長等待（秒）。Anthropic LLM 產出長 output 偶爾會跑 60-90 秒，
# 訂 180 秒給足空間；同時防止 LangChain #25309 hang bug 讓整個 pipeline 卡死。
_AINVOKE_HARD_TIMEOUT = 180.0


def _is_retryable_error(exc: BaseException) -> bool:
    """判斷 exception 是不是值得 retry 的瞬態錯誤。

    為什麼不用 retry_if_exception_type？
      各家 SDK 的 RateLimitError/TimeoutError 類別不同，且 LangChain 有時會
      把底層 exception 包成 generic 類型。檢查類別名稱 + 訊息關鍵字最 robust。
    """
    err_name = type(exc).__name__.lower()
    if "ratelimit" in err_name or "rate_limit" in err_name:
        return True
    if "timeout" in err_name:
        return True
    # asyncio.TimeoutError（wait_for 觸發）也要 retry
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "429", "resource exhausted", "quota")):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_random_exponential(min=4, max=120),  # 4s → 最多 2 分鐘
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _invoke_with_retry(llm: BaseChatModel, messages: list, **kwargs):
    return await asyncio.wait_for(
        llm.ainvoke(messages, **kwargs),
        timeout=_AINVOKE_HARD_TIMEOUT,
    )


async def safe_ainvoke(llm: BaseChatModel, messages: list, **kwargs):
    """Rate-limit-aware LLM invocation.

    四層防護：
      1. InMemoryRateLimiter（get_llm 時注入）— 主動節流，避開上限
      2. SDK 內建 max_retries=5         — provider SDK 自己的 429 retry
      3. tenacity exp backoff            — 上面兩層沒接住時兜底
      4. asyncio.wait_for hard timeout   — 防 LangChain #25309 hang bug

    大部分情況 (1) + (2) 就能擋掉 429，(3)(4) 是保險。
    """
    return await _invoke_with_retry(llm, messages, **kwargs)


# ---------------------------------------------------------------------------
# Role-based fallback chain
# ---------------------------------------------------------------------------
#
# 設計理由（基於 Vectara HHEM-2.3 leaderboard 2026-03-20 數據）：
#   - 「寫作 / 規劃 / 評估」類任務 → Claude 主導（instruction following 強）
#   - 「抽取 / 核對 / 驗證」類任務 → Gemini 主導（grounded summarization
#     幻覺率 7.0%，比 Claude Opus 4.6 的 12.2% 低近 2 倍）
#
# Fallback 鏈設計：
#   - 同類任務有 3 家供應商可切換，避免 API 故障 / rate limit 卡死
#   - 每家挑 Vectara HHEM 上幻覺率最低且 context 夠長的版本
#   - reasoning / thinking model 不選 — Vectara 數據顯示推理模型（o3-pro
#     23.3%、o4-mini 18.6%）幻覺反而更高
#
# 觸發 fallback 的條件（在 safe_ainvoke_chain 內判斷）：
#   - safe_ainvoke 重試 6 次後仍失敗（rate limit / 5xx / timeout）
#   - 程式 bug 類錯誤（ValueError / TypeError）不 fallback，直接 raise

ROLES: dict[str, list[tuple[str, str]]] = {
    # 寫作 / 規劃 / Judge：Claude 主導，同家先降級再跨家，最後 Gemini 兜底
    # 若 Anthropic credits 耗盡，fallback 到 Gemini 確保 pipeline 不死掉
    "writer": [
        ("claude", "claude-opus-4-6"),
        ("claude", "claude-sonnet-4-6"),
        ("claude", "claude-haiku-4-5-20251001"),
        ("gemini", "gemini-2.5-flash"),  # final fallback when Anthropic credits exhausted
    ],
    # 抽取 / 核對 / 驗證 / 規劃：Gemini Flash 主導（快速 structured output），跨家 fallback
    # 注意：gemini-2.5-pro 的 thinking tokens 會吃掉 max_tokens，導致 empty content
    # 所以 verifier chain 改用 gemini-2.5-flash（無 thinking overhead，速度快 3-5x）
    "verifier": [
        ("gemini", "gemini-2.5-flash"),
        ("claude", "claude-haiku-4-5-20251001"),
        ("claude", "claude-sonnet-4-6"),
    ],
}

# Context limits for fallback-chain models (used by iterative_refine budget calc)
ROLE_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
}


def _is_recoverable_via_fallback(exc: BaseException) -> bool:
    """Errors that switching to another provider might fix.

    程式 bug（ValueError / TypeError / KeyError）不 fallback，因為換家也救不了。
    其餘（API error / network / parse）都試 fallback。
    """
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    return True


def _create_llm(
    provider: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    """Create a chat model instance with rate limiter, no global state lookup."""
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limiter=_RATE_LIMITERS["claude"],
            max_retries=5,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limiter=_RATE_LIMITERS["openai"],
            max_retries=5,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_kw: dict = dict(
            model=model_name,
            max_tokens=max_tokens,
            temperature=1.0,  # Gemini 固定 1.0
            rate_limiter=_RATE_LIMITERS["gemini"],
            max_retries=5,
        )
        if "flash" in model_name:
            gemini_kw["thinking_budget"] = 0
        return ChatGoogleGenerativeAI(**gemini_kw)
    raise ValueError(f"Unknown provider: {provider}")


def _available_chain(role: str) -> list[tuple[str, str]]:
    """Return fallback chain filtered by available API keys."""
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}. Valid: {list(ROLES.keys())}")
    chain = ROLES[role]
    available = [(p, m) for p, m in chain if os.environ.get(KEY_ENV[p])]
    if not available:
        keys_needed = ", ".join(KEY_ENV[p] for p, _ in chain)
        raise RuntimeError(
            f"No available LLM for role '{role}'. "
            f"Set at least one of: {keys_needed}"
        )
    return available


def get_llm_for_role(
    role: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Return the primary (first available) LLM for a role.

    Use this when you need a single LLM instance (e.g. for iterative_refine
    internal logic that needs to pre-compute context limits). For invocation
    with automatic fallback, use safe_ainvoke_chain instead.
    """
    available = _available_chain(role)
    provider, model = available[0]
    return _create_llm(provider, model, max_tokens, temperature)


def get_role_context_limit(role: str) -> int:
    """Return context limit of role's primary (first available) model."""
    available = _available_chain(role)
    _, model = available[0]
    return ROLE_MODEL_CONTEXT_LIMITS.get(model, 200_000)


async def safe_ainvoke_chain(
    role: str,
    messages: list,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    **kwargs,
):
    """Invoke role's fallback chain — try each provider until one succeeds.

    對每個 provider 的 LLM 呼叫 safe_ainvoke（內含 rate limit / retry / timeout）。
    safe_ainvoke 失敗 → 切下一家。所有家都失敗 → 拋最後一個 exception。

    Fallback 不適用於：ValueError / TypeError 這類程式 bug（換家也救不了）。
    """
    available = _available_chain(role)
    last_exc: BaseException | None = None

    for i, (provider, model) in enumerate(available):
        try:
            llm = _create_llm(provider, model, max_tokens, temperature)
            result = await safe_ainvoke(llm, messages, **kwargs)
            if i > 0:
                logger.warning(
                    "[role=%s] fallback #%d (%s/%s) succeeded after %d failures",
                    role, i, provider, model, i,
                )
            return result
        except Exception as e:
            if not _is_recoverable_via_fallback(e):
                # 程式 bug — 立刻 raise，不浪費時間切下家
                raise
            last_exc = e
            remaining = len(available) - i - 1
            if remaining > 0:
                logger.warning(
                    "[role=%s] %s/%s failed (%s: %s) — trying next (%d remaining)",
                    role, provider, model, type(e).__name__, str(e)[:200], remaining,
                )
            else:
                logger.error(
                    "[role=%s] all %d providers exhausted, last error: %s",
                    role, len(available), e,
                )

    raise last_exc if last_exc else RuntimeError(f"Role {role} chain exhausted")


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


def get_available_providers() -> list[str]:
    """Return all providers with valid API keys, current provider first.

    Used by PoLL (Panel of LLM evaluators) to create diverse model panels.
    Returns at most 3 providers: the current one + any others with keys.
    """
    available = []
    current = _provider
    # Current provider first (key already validated at startup)
    available.append(current)
    # Then others that have API keys configured
    for p in MODELS:
        if p != current and os.environ.get(KEY_ENV.get(p, "")):
            available.append(p)
    return available
