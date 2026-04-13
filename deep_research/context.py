"""Context Window 管理 — Iterative Refinement / Incremental Summarization.

核心職責：
  1. 估算 token 數量
  2. 判斷全塞 or Iterative Refinement
  3. BM25 + Query Expansion 排序 sources
  4. 迴圈增量整合 sources 進 draft
  5. 統整 topic + refs + clarifications → full_research_topic

設計細節見 llm.py 頂部的 Context Window 管理策略註解。
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.llm import (
    get_context_limit,
    get_context_threshold,
    get_llm,
    get_provider,
    find_largest_available_provider,
    safe_ainvoke,
    safe_ainvoke_chain,
    get_role_context_limit,
    _available_chain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token 估算
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """粗估 token 數。中文約 1-2 tokens/字，英文約 1 token/4 chars，取保守值。"""
    return len(text) // 3


# ---------------------------------------------------------------------------
# 參考文件讀取（支援文字、圖片、PDF）
# ---------------------------------------------------------------------------

def read_reference_files(paths: list[str]) -> list[dict]:
    """讀取參考檔案，回傳統一格式。

    ============================================================
    ✅ 支援格式
    ============================================================

    【文字類 — 走 UTF-8 直讀，副檔名無限制，只要能 UTF-8 decode】
      - Markdown：  .md, .markdown
      - 純文字：    .txt, .log
      - 結構化：    .json, .yaml, .yml, .toml, .xml, .csv, .tsv
      - 原始碼：    .py, .js, .ts, .go, .rs, .java, .c, .cpp, .sh
      - 網頁：      .html, .htm
      - 任何其他 UTF-8 可讀的純文字檔
      → 格式：{"type": "text", "name": ..., "content": ...}

    【PDF — 用 pymupdf 提取純文字（圖表/排版會遺失）】
      - .pdf
      → 格式：{"type": "text", "name": ..., "content": 提取的純文字}
      - ⚠️ 需 pip install pymupdf，否則 content 會是錯誤訊息字串
      - ⚠️ 掃描檔 PDF（無 OCR 文字層）提取結果為空

    【圖片 — base64 編碼送 LLM 視覺理解】
      - PNG、JPEG / JPG、GIF、WebP、BMP、TIFF、SVG（任何 mime = image/*）
      → 格式：{"type": "image", "name": ..., "mime": ..., "data": base64}
      - ⚠️ 實務上 LLM provider 通常只認 PNG / JPEG / GIF / WebP
        TIFF / BMP / SVG 可能被 LLM 拒絕或誤判

    【Workspace 目錄 — 研究追加功能】
      - 傳入的 path 是目錄時，讀裡面的 final-report.md
      → 格式：{"type": "text", "name": "{dir}/final-report.md", "content": ...}
      - 用途：把前一次研究的報告當作新研究的 context

    ============================================================
    ❌ 不支援格式（會被 skip + 印 warning）
    ============================================================

      - Office 文件：.docx, .xlsx, .pptx（zip 容器，走文字 fallback 會 UnicodeDecodeError）
      - 壓縮檔：.zip, .tar, .gz, .7z
      - 音訊：.mp3, .wav, .m4a, .ogg
      - 影片：.mp4, .mov, .avi, .webm
      - 非 UTF-8 文字檔（Big5、GBK、Shift-JIS）— 編碼硬寫死 utf-8

    若需支援上述格式，需擴充此函式的分支邏輯。

    Raises:
        ValueError: 當使用者傳入明確不支援的格式時，立刻拋錯（fail-fast），
                    而不是 skip 後讓使用者後面才發現 ref 沒被讀到。
    Returns:
        list of ref dicts
    """
    refs = []
    for p in paths:
        path = Path(p)

        if path.is_dir():
            # Workspace 目錄 — 讀 final-report.md
            report = path / "final-report.md"
            if report.exists():
                refs.append({
                    "type": "text",
                    "name": f"{path.name}/final-report.md",
                    "content": report.read_text(encoding="utf-8"),
                })
            else:
                raise ValueError(
                    f"❌ 目錄 {path} 內沒有 final-report.md。\n"
                    f"   目錄參考僅支援讀取先前研究的 workspace（必須包含 final-report.md）。"
                )
            continue

        if not path.is_file():
            raise ValueError(f"❌ 參考檔案不存在：{path}")

        # ─── Fail-fast：明確不支援的格式立刻拋錯 ───────────────────
        # 這些格式就算走 UTF-8 fallback 也必定失敗（zip 容器、二進位）
        # 與其 skip 讓使用者後面才發現少了 ref，不如當下就明確告知
        UNSUPPORTED_EXTS = {
            # Office 文件（zip 容器，無法 UTF-8 讀）
            ".docx": "Word 文件", ".xlsx": "Excel 試算表", ".pptx": "PowerPoint 簡報",
            ".doc":  "Word 文件（舊版）", ".xls": "Excel 試算表（舊版）",
            ".ppt":  "PowerPoint 簡報（舊版）", ".odt": "OpenDocument 文字",
            ".ods":  "OpenDocument 試算表", ".odp": "OpenDocument 簡報",
            ".rtf":  "RTF 格式文件",
            # 電子書
            ".epub": "EPUB 電子書", ".mobi": "Kindle 電子書", ".azw3": "Kindle 電子書",
            # 壓縮檔
            ".zip": "ZIP 壓縮檔", ".tar": "TAR 封存檔", ".gz": "Gzip 壓縮檔",
            ".bz2": "Bzip2 壓縮檔", ".7z": "7-Zip 壓縮檔", ".rar": "RAR 壓縮檔",
            # 音訊
            ".mp3": "MP3 音訊", ".wav": "WAV 音訊", ".m4a": "M4A 音訊",
            ".ogg": "OGG 音訊", ".flac": "FLAC 音訊", ".aac": "AAC 音訊",
            # 影片
            ".mp4": "MP4 影片", ".mov": "MOV 影片", ".avi": "AVI 影片",
            ".mkv": "MKV 影片", ".webm": "WebM 影片", ".flv": "FLV 影片",
            # 其他二進位
            ".exe": "執行檔", ".dll": "動態連結庫", ".so": "共享函式庫",
            ".dmg": "macOS 磁碟映像", ".iso": "ISO 映像檔",
        }
        suffix = path.suffix.lower()
        if suffix in UNSUPPORTED_EXTS:
            raise ValueError(
                f"❌ 不支援的檔案格式：{path.name}（{UNSUPPORTED_EXTS[suffix]}）\n"
                f"   ✅ 目前僅支援：\n"
                f"      • 純文字類：.md, .txt, .json, .yaml, .csv, .html, 程式原始碼等\n"
                f"      • PDF：.pdf（只讀文字，圖表/圖片會被忽略）\n"
                f"      • 圖片：.png, .jpg, .jpeg, .gif, .webp\n"
                f"   如需提供 Office 文件，請先轉為 PDF 或貼成純文字。"
            )

        mime, _ = mimetypes.guess_type(str(path))

        if mime and mime.startswith("image/"):
            # 圖片 → base64
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            refs.append({
                "type": "image",
                "name": path.name,
                "mime": mime,
                "data": b64,
            })

        elif mime == "application/pdf":
            # ─── PDF：提取純文字，並明確告知使用者限制 ────────────
            # 使用者常誤以為 LLM 能「看」PDF 的圖表/排版，實際只讀到文字。
            # 在讀取當下就用 stderr 印提醒，確保使用者知道限制。
            print(
                f"📄 偵測到 PDF：{path.name}\n"
                f"   ⚠️  只會讀取『文字內容』。以下資訊會被忽略：\n"
                f"       • 圖片、圖表、示意圖\n"
                f"       • 表格的排版結構（文字仍可讀到，但可能錯位）\n"
                f"       • 掃描檔若無 OCR 文字層，將完全無法讀取\n"
                f"   如有關鍵圖表，建議另存為 PNG/JPG 一併提供。",
                file=sys.stderr,
            )
            text = _extract_pdf_text(path)
            if not text.strip():
                raise ValueError(
                    f"❌ 無法從 PDF 提取任何文字：{path.name}\n"
                    f"   可能原因：掃描檔無 OCR 文字層 / 加密 / 檔案損毀。"
                )
            refs.append({
                "type": "text",
                "name": path.name,
                "content": text,
            })

        else:
            # 其他檔案：嘗試用 UTF-8 讀（純文字副檔名會走到這裡）
            try:
                content = path.read_text(encoding="utf-8")
                refs.append({
                    "type": "text",
                    "name": path.name,
                    "content": content,
                })
            except UnicodeDecodeError:
                # 讀不到表示檔案不是 UTF-8 文字（可能是未列入黑名單的二進位格式
                # 或是 Big5/GBK 等非 UTF-8 編碼），一樣拋錯
                raise ValueError(
                    f"❌ 無法以 UTF-8 讀取 {path.name}\n"
                    f"   可能是未被明確列入支援清單的二進位檔、\n"
                    f"   或是非 UTF-8 編碼的文字檔（如 Big5, GBK）。\n"
                    f"   請先轉為 UTF-8 或另存為支援的格式。"
                )

    return refs


def _extract_pdf_text(path: Path) -> str:
    """從 PDF 提取文字。"""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pymupdf 未安裝，無法提取 PDF 文字。請執行 pip install pymupdf")
        return f"(PDF 檔案 {path.name}，需安裝 pymupdf 才能提取文字)"
    except Exception as e:
        logger.warning(f"PDF 提取失敗 {path}: {e}")
        return f"(PDF 提取失敗: {e})"


def refs_to_message_content(refs: list[dict]) -> list[dict]:
    """將 refs 轉成 LangChain 的 multimodal content blocks。

    用於 HumanMessage(content=blocks) 格式，支援文字+圖片混合。
    """
    blocks = []
    for ref in refs:
        if ref["type"] == "text":
            blocks.append({
                "type": "text",
                "text": f"\n--- 參考文件：{ref['name']} ---\n{ref['content']}",
            })
        elif ref["type"] == "image":
            blocks.append({
                "type": "text",
                "text": f"\n--- 參考圖片：{ref['name']} ---",
            })
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{ref['mime']};base64,{ref['data']}"},
            })
    return blocks


# ---------------------------------------------------------------------------
# 研究任務書統整（topic + refs + clarifications → full_research_topic）
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """你是研究需求分析師。請將以下資訊統整為一份結構化的研究任務書（research brief）。

這份任務書將作為整個研究流程的核心指令，所有後續的搜尋、分析、報告都依據這份任務書執行。
請用第三人稱描述委託者的需求，確保任何人讀到這份任務書都能獨立理解完整的研究需求。

## 統整規則
1. 將使用者的原始主題、參考文件中的關鍵資訊、澄清問答中的所有細節，融合成一份完整敘述
2. 參考文件中的具體數據、結論、觀點要保留並標註來源檔名
3. 消除矛盾：如果參考文件和問答有衝突，以問答（使用者最新意見）為準
4. 補充隱含需求：從問答中推斷出使用者沒有明說但顯然需要的面向
5. 圖片內容用文字描述其關鍵資訊（數據、架構圖的結構、趨勢等）

## 輸出格式

### 研究目標
（一段話，清楚說明為什麼做這個研究、期望得到什麼）

### 核心問題
（條列，這個研究需要回答的具體問題）

### 範圍與限制
（時間範圍、地域、技術邊界、排除項目、預算等硬性約束）

### 評估標準
（用什麼維度判斷好壞、怎麼比較、成功的定義）

### 已知背景
（從參考文件和問答中提取的已知事實，避免重複搜尋）

### 產出要求
（報告格式、受眾、深度偏好、特殊要求）"""


async def synthesize_research_topic(
    topic: str,
    refs: list[dict],
    clarifications: list[dict],
) -> str:
    """統整 topic + refs + clarifications → full_research_topic.

    一次性呼叫，在 Phase 0 澄清完成後執行。
    產出的 full_research_topic 會作為整個研究的 fixed context。

    Anti-pattern 防護（LLM 專注原則）：
      - 場景：使用者可能丟入整份長 PDF / 大型文字檔當 refs，
              一次性呼叫無 Iterative Refinement 保護，易觸發 LiM。
      - 策略：文字 refs 總量軟上限 30K tokens 警告，硬上限 50K tokens 截斷。
              圖片 refs 不動（圖片 token 計費方式不同，交給模型處理）。
    """
    import logging
    log = logging.getLogger(__name__)

    # refs 大小保護：文字類累積估 tokens，超過上限截斷
    safe_refs: list[dict] = []
    if refs:
        SOFT_LIMIT = 30_000
        HARD_LIMIT = 50_000
        accumulated = 0
        truncated_count = 0
        for ref in refs:
            if ref.get("type") == "text":
                content = ref.get("content", "")
                ref_tokens = estimate_tokens(content)
                if accumulated + ref_tokens <= HARD_LIMIT:
                    safe_refs.append(ref)
                    accumulated += ref_tokens
                else:
                    # 只留頭尾各 ~1500 tokens (4500 chars) 摘要
                    remaining = HARD_LIMIT - accumulated
                    if remaining > 3000:
                        keep_chars = (remaining - 500) * 3 // 2
                        head = content[:keep_chars]
                        tail = content[-keep_chars:] if len(content) > keep_chars * 2 else ""
                        if tail:
                            new_content = f"{head}\n\n...[中段省略 ~{ref_tokens - remaining} tokens 以符合 50K 上限]...\n\n{tail}"
                        else:
                            new_content = head + "\n\n...[尾段省略以符合 50K 上限]..."
                        safe_refs.append({**ref, "content": new_content})
                        accumulated = HARD_LIMIT
                    truncated_count += 1
            else:
                # 圖片直接放行
                safe_refs.append(ref)
        if accumulated > SOFT_LIMIT:
            log.warning(
                "[synthesize] refs 文字總量 %d tokens 超過軟上限 %d — 建議拆分或先自行摘要",
                accumulated,
                SOFT_LIMIT,
            )
        if truncated_count:
            log.warning(
                "[synthesize] 已截斷 %d 份 refs（超過 50K tokens 硬上限）",
                truncated_count,
            )

    # 組裝 multimodal content blocks
    content_blocks: list[dict] = [
        {"type": "text", "text": f"## 原始研究主題\n\n{topic}"},
    ]

    # 參考文件（文字+圖片）
    if safe_refs:
        ref_blocks = refs_to_message_content(safe_refs)
        content_blocks.extend(ref_blocks)

    # 澄清問答
    if clarifications:
        qa_text = "\n## 澄清問答\n\n"
        for i, qa in enumerate(clarifications, 1):
            qa_text += f"**Q{i}:** {qa['question']}\n**A{i}:** {qa['answer']}\n\n"
        content_blocks.append({"type": "text", "text": qa_text})

    # Prompt caching 標記
    # Anthropic: cache_control 在 system message 上（整個 SYNTHESIZE_PROMPT 會被 cache）
    # OpenAI: 自動 prefix caching（prompt > 1024 tokens 時自動啟用）
    # Gemini: 自動 prefix caching
    system_msg = SystemMessage(content=SYNTHESIZE_PROMPT)
    human_msg = HumanMessage(content=content_blocks)

    # role="writer" — 規劃 / 統整類，Claude Opus 主導，內含 fallback chain
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[system_msg, human_msg],
        max_tokens=8192,
        temperature=0.2,
    )
    return response.content


# ---------------------------------------------------------------------------
# BM25 排序（Query Expansion + ranking）
# ---------------------------------------------------------------------------

async def _expand_query(topic: str) -> str:
    """LLM 生成加長版 query，提升 BM25 的 recall。

    包含：同義詞、相關術語、多語言對應、領域慣用詞。
    成本很低（一次 verifier chain call，output ~200 tokens）。
    """
    response = await safe_ainvoke_chain(
        role="verifier",
        messages=[
            SystemMessage(content="""根據研究主題，生成一段用於資訊檢索的加長查詢。
包含：同義詞、相關術語、英文對應詞、領域慣用詞、相關概念。
目的是提升 BM25 關鍵字匹配的 recall。
直接輸出查詢文字，不要解釋。"""),
            HumanMessage(content=topic),
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.content


def _rank_sources_bm25(sources: list[str], query: str) -> list[str]:
    """用 BM25 依相關性排序 sources。

    Args:
        sources: list of source text strings
        query: expanded query string

    Returns:
        sorted sources (most relevant first)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 未安裝，跳過排序。請執行 pip install rank-bm25")
        return sources

    if not sources or not query:
        return sources

    # Tokenize（簡單分詞：中文逐字、英文空格分）
    def tokenize(text: str) -> list[str]:
        import re
        # 中英文混合分詞：英文按空格+標點，中文逐字
        tokens = []
        for segment in re.split(r'(\s+)', text):
            segment = segment.strip()
            if not segment:
                continue
            # 如果包含 CJK 字元，逐字切
            if any('\u4e00' <= c <= '\u9fff' for c in segment):
                tokens.extend(list(segment))
            else:
                tokens.extend(segment.lower().split())
        return tokens

    tokenized_sources = [tokenize(s) for s in sources]
    tokenized_query = tokenize(query)

    bm25 = BM25Okapi(tokenized_sources)
    scores = bm25.get_scores(tokenized_query)

    # 按 score 降序排
    ranked = sorted(zip(scores, sources), reverse=True)
    return [s for _, s in ranked]


# ---------------------------------------------------------------------------
# Iterative Refinement 核心迴圈
# ---------------------------------------------------------------------------

ITERATIVE_SYSTEM = """你是深度研究分析師。你的任務是根據研究任務書，將新的搜尋結果整合進目前的研究草稿中。

## 整合規則

1. 逐篇審閱本輪新資訊，判斷是否與研究任務相關
2. 相關且有價值 → 整合進草稿的適當位置
3. 與草稿現有內容重複 → 跳過，但如果新資訊有更精確的數據或更新的日期，替換舊的
4. 與草稿現有內容矛盾 → 兩者都保留，標記 [矛盾待查] 並註明來源
5. 不相關或低品質 → 跳過，不要加入

## 溯源要求

每個事實、數據、觀點都必須標註來源：
- 格式：「內容文字 [來源：檔名或URL]」
- 數字必須逐字引用原文，不可四捨五入或改寫
- 推論必須標記 [推論] 並說明依據哪些事實推導

## 草稿結構

維持以下結構，新內容插入到對應段落的末尾（不改動已有段落的順序和內容）：
1. 每個核心問題一個段落
2. 段落內按「事實 → 數據 → 分析 → 矛盾/待查」排列
3. 段落末尾可以有 [待補充] 標記

## 輸出

直接輸出更新後的完整草稿。不要輸出解釋、不要輸出 diff、不要說「我更新了什麼」。
只輸出草稿本身。"""


async def iterative_refine(
    sources: list[str],
    full_research_topic: str,
    system_prompt: str = "",
    extra_context: str = "",
    tier: str = "strong",
    provider: str | None = None,
    role: str | None = None,
) -> str:
    """Iterative Refinement 核心：將所有 sources 增量整合進 draft。

    決策流程：
      1. total_tokens < budget → 全塞，一次 LLM call
      2. total_tokens >= budget → 迴圈，每輪塞到 budget 為止
      3. fixed_cost > budget → 一篇一篇送
      4. fixed_cost + 單篇 > 100% context → role 模式 raise；tier 模式換最大 provider

    Args:
        sources: list of source text strings（搜尋結果原文）
        full_research_topic: 統整後的研究任務書
        system_prompt: 自訂 system prompt（預設用 ITERATIVE_SYSTEM）。
            各 phase 傳入各自的 prompt：
            - Phase 1b: 攻擊式事實核查 prompt
            - Phase 2: 報告整合 prompt（預設）
            - Phase 3: 最終審計 prompt
        extra_context: 額外固定 context（如待核對 claims、statements），
            會放在 research_topic 之後、draft 之前。
        tier: LLM tier ("strong" or "fast") — 僅在 role=None 時使用
        provider: override provider — 僅在 role=None 時使用
        role: 角色化 fallback chain 模式（推薦）。
            "writer"   → Claude Opus → Sonnet → GPT-pro（規劃 / 整合 / 報告）
            "verifier" → Gemini → GPT-mini → Claude Haiku（抽取 / 核對 / 審計）
            設定後，內部走 safe_ainvoke_chain（自動 fallback），
            context_limit / cache 格式以 chain 第一個 model 為準。

    Returns:
        最終的 draft / 核查結果 / 審計結果
    """
    if not system_prompt:
        system_prompt = ITERATIVE_SYSTEM

    if role is not None:
        # Role 模式：用 chain 第一個 model 的 context limit，cache 格式取第一家 provider
        primary_provider, _primary_model = _available_chain(role)[0]
        p = primary_provider
        context_limit = get_role_context_limit(role)
    else:
        p = provider or get_provider()
        context_limit = get_context_limit(p, tier)

    threshold = get_context_threshold()
    budget = int(context_limit * threshold)

    # extra_context 放大器檢查 —
    # iterative_refine 每輪都會重傳 extra_context，分批模式下實際消耗 = extra × 輪數。
    # 呼叫端若塞了整份登記表 / 全部 claims / 全部 statements，這裡會放大成雜訊主因。
    # 軟上限 2K tokens（警告），硬上限 4K tokens（截斷保護）。
    if extra_context:
        extra_tokens = estimate_tokens(extra_context)
        if extra_tokens > 2000:
            logger.warning(
                f"iterative_refine: extra_context 過大 ({extra_tokens:,} tokens)。"
                f"每輪都會重傳 → 分批模式下實際消耗 = {extra_tokens:,} × 輪數。"
                f"建議呼叫端拆成多個獨立 call（per-section / per-group）或精簡 context。"
            )
        if extra_tokens > 4000:
            char_cap = 12000  # ~4000 tokens
            extra_context = (
                extra_context[:char_cap]
                + "\n\n...[extra_context 超過 4K tokens 硬上限，已截斷]..."
            )
            logger.warning(
                f"extra_context 硬截斷至 ~{estimate_tokens(extra_context):,} tokens"
            )

    # 估算 fixed 部分的 tokens
    fixed_prompt_tokens = estimate_tokens(system_prompt + full_research_topic + extra_context)
    total_source_tokens = sum(estimate_tokens(s) for s in sources)
    total_tokens = fixed_prompt_tokens + total_source_tokens

    logger.info(
        f"Context 決策: fixed={fixed_prompt_tokens:,} sources={total_source_tokens:,} "
        f"total={total_tokens:,} budget={budget:,} ({threshold:.0%} of {context_limit:,})"
    )

    # --- Step 1: 全塞判斷 ---
    if total_tokens < budget:
        logger.info("全塞模式：所有 sources 一次送入")
        return await _single_pass(sources, full_research_topic, system_prompt, extra_context, tier, p, role)

    # --- Step 2+: Iterative Refinement ---
    logger.info(f"Iterative Refinement 模式：{len(sources)} sources 分批處理")

    # BM25 排序（最相關的先處理）
    expanded_query = await _expand_query(full_research_topic)
    sorted_sources = _rank_sources_bm25(sources, expanded_query)

    draft = ""
    processed = 0

    while processed < len(sorted_sources):
        # 計算本輪 budget
        draft_tokens = estimate_tokens(draft)
        fixed_cost = fixed_prompt_tokens + draft_tokens
        remaining = budget - fixed_cost

        if remaining <= 0:
            # fixed_cost 超過 budget → 一篇一篇送
            remaining_for_one = context_limit - fixed_cost  # 用 100% limit 而非 threshold

            if remaining_for_one <= 0:
                if role is not None:
                    # Role 模式：fallback chain 自己會挑可用的 provider，
                    # 此處若仍超 limit 表示研究任務書過長，無解。
                    raise RuntimeError(
                        f"[role={role}] fixed_prompt + draft ({fixed_cost:,} tokens) 超過 "
                        f"chain primary model 的 100% context limit ({context_limit:,} tokens)。"
                        f"請縮短研究任務書或調高 --context-threshold。"
                    )
                # tier 模式：連 100% context 都不夠 → 換最大 provider
                larger = find_largest_available_provider(tier)
                if larger:
                    larger_limit = get_context_limit(larger, tier)
                    remaining_for_one = larger_limit - fixed_cost
                    if remaining_for_one <= 0:
                        raise RuntimeError(
                            f"fixed_prompt + draft ({fixed_cost:,} tokens) 超過最大可用 provider "
                            f"({larger}, {larger_limit:,} tokens) 的 100% context limit。"
                            f"請縮短研究任務書或調高 --context-threshold。"
                        )
                    logger.warning(f"切換到 {larger} (context: {larger_limit:,}) 處理剩餘 sources")
                    p = larger
                else:
                    raise RuntimeError(
                        f"fixed_prompt + draft ({fixed_cost:,} tokens) 超過 {p} 的 100% context limit "
                        f"({context_limit:,} tokens)，且無更大的 provider 可用。"
                    )

            # 一篇一篇送
            source = sorted_sources[processed]
            source_tokens = estimate_tokens(source)
            if source_tokens > remaining_for_one:
                # 單篇太長，截取能放的部分（這是唯一允許截斷的地方）
                char_limit = remaining_for_one * 3  # 反向估算字元數
                source = source[:char_limit] + "\n\n[...此來源因篇幅過長被截斷...]"
                logger.warning(f"Source {processed+1} 過長 ({source_tokens:,} tokens)，截斷至 {remaining_for_one:,} tokens")

            draft = await _refine_once(draft, [source], full_research_topic, system_prompt, extra_context, tier, p, role)
            processed += 1
        else:
            # 正常情況：貪心塞多篇
            batch = []
            batch_tokens = 0
            while processed < len(sorted_sources):
                source = sorted_sources[processed]
                source_tokens = estimate_tokens(source)
                if batch and batch_tokens + source_tokens > remaining:
                    break  # 這批滿了
                batch.append(source)
                batch_tokens += source_tokens
                processed += 1

            draft = await _refine_once(draft, batch, full_research_topic, system_prompt, extra_context, tier, p, role)

        logger.info(f"已處理 {processed}/{len(sorted_sources)} sources, draft: {estimate_tokens(draft):,} tokens")

    return draft


async def _single_pass(
    sources: list[str],
    full_research_topic: str,
    system_prompt: str,
    extra_context: str,
    tier: str,
    provider: str,
    role: str | None = None,
) -> str:
    """全塞模式：一次 LLM call 處理所有 sources。"""
    all_sources = "\n\n---\n\n".join(
        f"### 來源 {i+1}\n{s}" for i, s in enumerate(sources)
    )

    # Prompt caching 設計：
    #   Anthropic: system message + human message 前段（research_topic）會被 cache
    #              需要在 content block 加 cache_control
    #   OpenAI:    自動 prefix caching（prompt > 1024 tokens 時啟用，無需額外參數）
    #   Gemini:    自動 prefix caching
    system_content = _build_system_with_cache(system_prompt, provider)

    extra_section = f"\n\n---\n\n{extra_context}" if extra_context else ""
    human_text = f"""## 研究任務書

{full_research_topic}{extra_section}

---

## 搜尋結果（共 {len(sources)} 篇）

{all_sources}"""

    human_content = _build_human_with_cache(human_text, full_research_topic, provider)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    if role is not None:
        response = await safe_ainvoke_chain(
            role=role,
            messages=messages,
            max_tokens=16384,
            temperature=0.2,
        )
    else:
        llm = get_llm(tier=tier, max_tokens=16384, temperature=0.2, provider=provider)
        response = await safe_ainvoke(llm, messages)
    return response.content


async def _refine_once(
    draft: str,
    source_batch: list[str],
    full_research_topic: str,
    system_prompt: str,
    extra_context: str,
    tier: str,
    provider: str,
    role: str | None = None,
) -> str:
    """一輪 Iterative Refinement：draft + 一批 sources → 更新後的 draft。"""
    batch_text = "\n\n---\n\n".join(
        f"### 來源 {i+1}\n{s}" for i, s in enumerate(source_batch)
    )

    draft_section = draft if draft else "（尚無結果，這是第一輪）"

    # Prompt caching 設計：
    #   fixed_prompt（SYSTEM + 研究任務書 + extra_context）每輪不變 → cached
    #   draft 每輪成長但前段不變 → 部分 cached
    #   source_batch 每輪全新 → 不 cached
    system_content = _build_system_with_cache(system_prompt, provider)

    extra_section = f"\n\n---\n\n{extra_context}" if extra_context else ""
    human_text = f"""## 研究任務書

{full_research_topic}{extra_section}

---

## 目前累積結果

{draft_section}

---

## 本輪新資訊（共 {len(source_batch)} 篇）

{batch_text}"""

    human_content = _build_human_with_cache(human_text, full_research_topic, provider)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    if role is not None:
        response = await safe_ainvoke_chain(
            role=role,
            messages=messages,
            max_tokens=16384,
            temperature=0.2,
        )
    else:
        llm = get_llm(tier=tier, max_tokens=16384, temperature=0.2, provider=provider)
        response = await safe_ainvoke(llm, messages)
    return response.content


# ---------------------------------------------------------------------------
# Prompt Caching helpers
# ---------------------------------------------------------------------------

def _build_system_with_cache(system_text: str, provider: str) -> str | list[dict]:
    """Build system message content with provider-specific cache control.

    Anthropic: 使用 cache_control 標記，讓 system prompt 被 cache。
               cache_control: {"type": "ephemeral"} 表示這個 block 應被 cache，
               TTL 為 5 分鐘（Anthropic 自動管理）。
    OpenAI/Gemini: 自動 prefix caching，不需要額外參數，回傳純字串。
    """
    if provider == "claude":
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system_text


def _build_human_with_cache(
    full_text: str,
    research_topic: str,
    provider: str,
) -> str | list[dict]:
    """Build human message content with cache control on the fixed prefix.

    將 research_topic 標記為 cacheable prefix：
      - 它在每輪迴圈中都不變
      - 它通常是最長的固定部分
      - cache 命中後，後面的 draft + sources 是增量付費

    Anthropic: 拆成兩個 content block，第一個（research_topic）加 cache_control。
    OpenAI/Gemini: 自動 prefix caching，不需要拆分。
    """
    if provider == "claude":
        # 找到 research_topic 在 full_text 中的結束位置，拆成兩段
        topic_end = full_text.find(research_topic) + len(research_topic)
        prefix = full_text[:topic_end]
        suffix = full_text[topic_end:]
        blocks = [
            {
                "type": "text",
                "text": prefix,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        if suffix.strip():
            blocks.append({"type": "text", "text": suffix})
        return blocks
    return full_text
