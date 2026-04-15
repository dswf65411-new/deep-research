"""LLM provider factory — Claude / OpenAI / Gemini, auto-detect by key availability.

Model and context window are hardcoded and updated manually on a regular basis.
Each provider exposes two tiers:
  - "strong": primary reasoning (Phase 0 plan, Phase 1a search, Phase 2 integrate, Phase 3 audit+report)
  - "fast":   sub-task (Phase 1b attack-style fact-check)
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
_context_threshold: float = 0.3  # default 30%; user can override via --context-threshold

# ---------------------------------------------------------------------------
# Model configuration (hardcoded, refreshed periodically)
# ---------------------------------------------------------------------------
#
# When to update: whenever each vendor ships a new stable model, update this block by hand.
# How to update: confirm the new model's context window (official docs or API query) and
#   refresh the matching fields below.
#
# Selection principles:
#   strong — highest-quality stable model (no previews); used for tasks needing judgement
#   fast   — fast, low-cost model used for mechanical verify tasks
#
# Last updated: 2026-04-12

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
# Rate limiting (proactive throttling to avoid provider per-minute limits)
# ---------------------------------------------------------------------------
#
# Two layers of defence:
#   1. InMemoryRateLimiter — client-side token bucket, injected in get_llm.
#      The instance must be shared across calls (module-level dict); otherwise
#      every get_llm would create a new bucket, defeating the throttle.
#   2. safe_ainvoke — tenacity reactive retry + asyncio.wait_for hard timeout.
#      The Anthropic SDK's built-in max_retries occasionally misses (LangChain
#      #25309 hang bug); this layer is the safety net.
#
# Rate design:
#   Anthropic Tier 1 Sonnet defaults to 30K ITPM. Assuming an average 3K input tokens:
#     30000 / 3000 = 10 req/min ≈ 0.17 req/s
#   Picking a conservative 0.25 req/s (≈ 15 req/min) with max_bucket_size=5 allows
#   short bursts.
#   Gemini has 1M context and Google's default TPM is looser, so we relax to 0.5.
#   To go faster, edit the dict below.
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
# Context window management strategy
# ---------------------------------------------------------------------------
#
# Problem: the LLM context window has a hard limit, and even when text fits,
#   attention spreads thinner as the fill rate rises.
#   - Research requires cross-checking many claims × many sources (multi-needle
#     scenario), which demands high precision.
#   - Lost in the Middle (Liu et al. 2023): when the correct answer sits in the
#     middle of 20 documents, GPT-3.5 / Claude-1.3 accuracy drops from ~70% at
#     the head/tail to ~40% in the middle (U-shaped attention curve).
#   - RULER (Hsieh et al. 2024): on multi-needle retrieval tasks, most models
#     start degrading once context is filled to 25-50% of the stated length.
#   - Newer 2024-2025 models (Claude 3.5+, GPT-4o, Gemini 2.5) improve on the
#     single-needle case, but public multi-needle benchmarks remain scarce, so
#     there is no confirmed safe threshold.
#
# Strategy: Iterative Refinement (Incremental Summarization)
#   No truncation, no data loss. Decision flow:
#
#   Step 0 — estimate input tokens:
#     Rough estimate: len(text) // 3 (Chinese is about 1-2 tokens per char; take
#     the conservative value).
#     Compute total_tokens = tokens(fixed_prompt + draft + all_sources).
#
#   Step 1 — stuff-everything check:
#     budget = current_model_limit × threshold
#     If total_tokens < budget → stuff all context; skip the loop.
#     Stuffing lets the LLM see all sources at once, enabling cross-document
#     comparison with the highest quality and no accumulated bias.
#
#   Step 2 — Iterative Refinement loop:
#     When total_tokens ≥ budget, switch to the loop:
#
#     2a. At the start of each round:
#         fixed_cost = tokens(fixed_prompt + draft)
#         remaining = budget - fixed_cost
#
#     2b. If remaining > 0 (normal case):
#         Greedily pack sources (in BM25-ranked order) until remaining is used up.
#         draft = LLM(fixed_prompt + draft + source_batch)
#         New info goes at the tail, draft at the front, to exploit prompt
#         prefix caching and cut cost.
#
#     2c. If remaining ≤ 0 (fixed_cost alone exceeds budget):
#         No panic; degrade to one source per round. Each round exceeds the
#         threshold but stays within the context limit, trading precision for
#         availability.
#
#     2d. If fixed_cost + a single source > 100% of the current model's context
#         limit: automatically switch to the provider with the largest available
#         context window (find_largest_available_provider), recompute the budget,
#         and continue. Example: GPT-4o 128K can't fit → switch to Gemini 1M.
#         If even 100% of the new provider can't fit → raise error for the user.
#
#     Academic references:
#       - RAPTOR (ICLR 2024): recursive summarization tree, merging from leaf chunks upward
#       - Chain-of-Density (Adams et al. 2023): iterative refinement that makes summaries denser
#       - MemWalker (Chen et al. 2023): LLM iteratively walks a text tree collecting info
#     Compared with one-shot MapReduce, Iterative Refinement lets the LLM see
#     the full accumulated draft at every step, producing better decisions and
#     amortising cost through prefix caching.
#
#   Step 3 — BM25 ranking (used inside Step 2):
#     When there are many sources, BM25 + LLM Query Expansion sets the per-batch
#     priority so the most relevant sources are processed first. Query Expansion
#     has the LLM produce a longer query (synonyms, multilingual equivalents)
#     based on the research topic, improving BM25 recall.
#     Large chunk + long query = BM25 TF-IDF statistics are steadier, and semantic
#     mismatch is minor.
#
# Fill-rate threshold (_context_threshold):
#   Default 0.3 (30%). Users can tune via CLI --context-threshold.
#   30% is the conservative baseline; future A/B testing (stuff-everything vs
#   iterative with the same data) can calibrate this.
#
# Implementation lives in deep_research/context.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider / threshold configuration
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
        "No LLM API key found. Set at least one in .env:\n"
        "  ANTHROPIC_API_KEY (Claude)\n"
        "  GEMINI_API_KEY (Gemini)\n"
        "  OPENAI_API_KEY (OpenAI)\n"
        "Run `make init` to reconfigure."
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
        f"Cannot infer provider from model name '{model_name}'. "
        f"Use claude/gemini/openai or a full model version string (e.g. gemini-2.5-pro)."
    )


def set_provider(provider_or_model: str) -> None:
    """Set the global LLM provider. Called once at startup.

    Accepts three kinds of input:
      1. "auto" — detect key in the order claude > gemini > openai
      2. "claude" / "gemini" / "openai" — use that vendor's hardcoded model
      3. A full model version string like "gemini-2.5-pro" — use this model
         directly (overrides the strong tier)
    """
    global _provider

    provider_names = set(MODELS.keys())

    if provider_or_model == "auto":
        _provider = auto_detect_provider()

    elif provider_or_model in provider_names:
        key_name = KEY_ENV[provider_or_model]
        if not os.environ.get(key_name):
            raise RuntimeError(f"--model {provider_or_model} was selected but {key_name} is not set. Add it to .env.")
        _provider = provider_or_model

    else:
        # Full model version string → infer provider, override strong tier
        detected = _detect_provider_from_model_name(provider_or_model)
        key_name = KEY_ENV[detected]
        if not os.environ.get(key_name):
            raise RuntimeError(f"Model {provider_or_model} requires {key_name}, which is not set. Add it to .env.")
        _provider = detected
        # Override strong model name (context_limit stays on its hardcoded value)
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

    When input tokens exceed model context limit × threshold, switch to
    Iterative Refinement mode instead of stuffing everything.
    Default 0.3 (30%). Users can override via CLI --context-threshold.
    """
    global _context_threshold
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"context-threshold must be in (0.0, 1.0], got: {threshold}")
    _context_threshold = threshold


def get_context_threshold() -> float:
    return _context_threshold


# ---------------------------------------------------------------------------
# LLM instantiation
# ---------------------------------------------------------------------------

def get_llm(
    tier: str = "strong",
    max_tokens: int = 8192,
    temperature: float = 0.2,
    provider: str | None = None,
) -> BaseChatModel:
    """Return a chat model instance.

    Args:
        tier: "strong" (primary reasoning) or "fast" (sub-task)
        max_tokens: max output tokens
        temperature: sampling temperature (Gemini is forcibly overridden to 1.0; see note below)
        provider: override provider (default: use global _provider)

    Note on temperature:
        The Gemini 2.5 family has different temperature semantics than Claude/OpenAI.
        Its default is 1.0, and lowering it for reasoning tasks actually degrades
        quality (per Google's official guidance). Therefore Gemini is always
        overridden to 1.0 regardless of what the caller passes.
        Claude/OpenAI honour the caller-specified temperature.
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
        # Gemini fixed at 1.0: lowering degrades reasoning quality
        gemini_kwargs: dict = dict(
            model=model_name,
            max_tokens=max_tokens,
            temperature=1.0,
            rate_limiter=_RATE_LIMITERS["gemini"],
            max_retries=5,
        )
        # Flash model is used for structured JSON extraction; no thinking tokens needed.
        # With thinking enabled, nearly all of the token budget goes to reasoning and content is truncated.
        if "flash" in model_name:
            gemini_kwargs["thinking_budget"] = 0
        return ChatGoogleGenerativeAI(**gemini_kwargs)

    raise ValueError(f"Unknown provider: {p}")


# ---------------------------------------------------------------------------
# Safe ainvoke (reactive retry + hard timeout)
# ---------------------------------------------------------------------------

# Maximum wait time per call (seconds). Anthropic LLMs producing long outputs
# occasionally run 60-90 seconds, so we set 180s to leave room; it also
# prevents the LangChain #25309 hang bug from stalling the whole pipeline.
_AINVOKE_HARD_TIMEOUT = 180.0


def _is_retryable_error(exc: BaseException) -> bool:
    """Decide whether an exception is a transient error worth retrying.

    Why not retry_if_exception_type?
      Each SDK has its own RateLimitError/TimeoutError classes, and LangChain
      sometimes wraps the underlying exception into a generic type. Checking
      the class name plus message keywords is the most robust approach.
    """
    err_name = type(exc).__name__.lower()
    if "ratelimit" in err_name or "rate_limit" in err_name:
        return True
    if "timeout" in err_name:
        return True
    # asyncio.TimeoutError (raised by wait_for) should also retry
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "429", "resource exhausted", "quota")):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_random_exponential(min=4, max=120),  # 4s → up to 2 minutes
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

    Four layers of defence:
      1. InMemoryRateLimiter (injected in get_llm) — proactive throttling to stay under limits
      2. SDK built-in max_retries=5                — provider SDK's own 429 retry
      3. tenacity exp backoff                      — fallback when the above two miss
      4. asyncio.wait_for hard timeout             — guards against LangChain #25309 hang bug

    In most cases (1) + (2) absorb 429s; (3)(4) are insurance.

    Emits a usage-metadata record to the run-wide cost tracker (Whisper P2-2)
    so ``main.py`` can surface progress and rough USD burn in ``--json``
    mode. Failures in the tracker are swallowed — a metrics bug must never
    take down a real LLM call.
    """
    response = await _invoke_with_retry(llm, messages, **kwargs)
    try:
        from deep_research.harness import cost_tracker as _cost
        usage = _cost.extract_usage(response)
        if usage:
            model, itoks, otoks = usage
            if not model:
                # ChatAnthropic stores the model on the chat object, not the
                # response. Recover it from the caller if possible so the
                # pricing table gets the right key.
                model = getattr(llm, "model", "") or getattr(llm, "model_name", "") or ""
            _cost.record_llm_call(model, itoks, otoks)
    except Exception:
        logger.debug("cost tracker record_llm_call failed", exc_info=True)
    return response


# ---------------------------------------------------------------------------
# Role-based fallback chain
# ---------------------------------------------------------------------------
#
# Rationale (based on Vectara HHEM-2.3 leaderboard data, 2026-03-20):
#   - "Writing / plan / evaluation" tasks → Claude leads (strong instruction following)
#   - "Extract / verify / validate" tasks → Gemini leads (grounded summarization
#     hallucination rate 7.0%, nearly 2x lower than Claude Opus 4.6's 12.2%)
#
# Fallback chain design:
#   - Each task category has 3 vendors to switch between, avoiding API outages /
#     rate-limit stalls
#   - For each vendor, pick the version with the lowest hallucination on
#     Vectara HHEM plus a long-enough context
#   - Do NOT pick reasoning / thinking models — Vectara data shows reasoning
#     models (o3-pro 23.3%, o4-mini 18.6%) actually hallucinate more
#
# Fallback trigger conditions (decided inside safe_ainvoke_chain):
#   - safe_ainvoke still fails after 6 retries (rate limit / 5xx / timeout)
#   - Program bug errors (ValueError / TypeError) do NOT fallback; raise directly

ROLES: dict[str, list[tuple[str, str]]] = {
    # Writing / plan / judge: Claude leads; degrade within vendor first, then cross-vendor, with Gemini as the final safety net.
    # If Anthropic credits are exhausted, fallback to Gemini so the pipeline stays alive.
    "writer": [
        ("claude", "claude-opus-4-6"),
        ("claude", "claude-sonnet-4-6"),
        ("claude", "claude-haiku-4-5-20251001"),
        ("gemini", "gemini-2.5-flash"),  # final fallback when Anthropic credits exhausted
    ],
    # Extract / verify / validate / plan: Gemini Flash leads (fast structured output) with cross-vendor fallback.
    # Note: gemini-2.5-pro's thinking tokens consume max_tokens and produce empty content,
    # so the verifier chain uses gemini-2.5-flash (no thinking overhead, 3-5x faster).
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

    Program bugs (ValueError / TypeError / KeyError) do NOT fallback because
    switching vendors won't help. Everything else (API error / network / parse)
    is retried via fallback.
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
            temperature=1.0,  # Gemini fixed at 1.0
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

    For each provider's LLM, call safe_ainvoke (which contains rate limit /
    retry / timeout). If safe_ainvoke fails → switch to the next vendor. If
    every vendor fails → raise the last exception.

    Fallback does NOT apply to: program bugs like ValueError / TypeError
    (switching vendors can't save them).
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
                # Program bug — raise immediately, don't waste time trying the next vendor
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
# Context-limit queries
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
