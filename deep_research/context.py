"""Context Window management — Iterative Refinement / Incremental Summarization.

Core responsibilities:
  1. Estimate token count
  2. Decide between fit-all-at-once vs. Iterative Refinement
  3. BM25 + Query Expansion ranking of sources
  4. Loop incrementally integrating sources into the draft
  5. Integrate topic + refs + clarifications -> full_research_topic

See the Context Window management strategy comments at the top of llm.py for design details.
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
from deep_research.prompts_shared import FOCUSED_EXEC_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count estimate. Chinese is about 1-2 tokens/char; English is about 1 token/4 chars; take the conservative value."""
    return len(text) // 3


# ---------------------------------------------------------------------------
# Reference file reading (supports text, images, PDF)
# ---------------------------------------------------------------------------

def read_reference_files(paths: list[str]) -> list[dict]:
    """Read reference files and return them in a unified format.

    ============================================================
    Supported formats
    ============================================================

    [Text — read directly as UTF-8; no extension restriction so long as it UTF-8 decodes]
      - Markdown:  .md, .markdown
      - Plain text:    .txt, .log
      - Structured:    .json, .yaml, .yml, .toml, .xml, .csv, .tsv
      - Source code:    .py, .js, .ts, .go, .rs, .java, .c, .cpp, .sh
      - Web:      .html, .htm
      - Any other UTF-8 readable plain text file
      -> Format: {"type": "text", "name": ..., "content": ...}

    [PDF — extract plain text with pymupdf (images/layout are lost)]
      - .pdf
      -> Format: {"type": "text", "name": ..., "content": extracted plain text}
      - Requires pip install pymupdf; otherwise content is an error message string
      - Scanned PDFs (no OCR text layer) produce empty output

    [Images — base64 encoded and sent to the LLM for visual understanding]
      - PNG, JPEG / JPG, GIF, WebP, BMP, TIFF, SVG (anything with mime = image/*)
      -> Format: {"type": "image", "name": ..., "mime": ..., "data": base64}
      - In practice, LLM providers usually accept only PNG / JPEG / GIF / WebP;
        TIFF / BMP / SVG may be rejected or misidentified

    [Workspace directory — research follow-up feature]
      - When the path is a directory, read the final-report.md inside it
      -> Format: {"type": "text", "name": "{dir}/final-report.md", "content": ...}
      - Use case: feed a previous research report as context for new research

    ============================================================
    Unsupported formats (skipped + warning printed)
    ============================================================

      - Office documents: .docx, .xlsx, .pptx (zip containers, will UnicodeDecodeError via text fallback)
      - Archives: .zip, .tar, .gz, .7z
      - Audio: .mp3, .wav, .m4a, .ogg
      - Video: .mp4, .mov, .avi, .webm
      - Non-UTF-8 text files (Big5, GBK, Shift-JIS) — encoding is hard-coded to utf-8

    To support any of the above, you must extend the branching logic in this function.

    Raises:
        ValueError: when the user passes an explicitly unsupported format,
                    raise immediately (fail-fast) rather than skipping and
                    letting the user discover that a ref was dropped later.
    Returns:
        list of ref dicts
    """
    refs = []
    for p in paths:
        path = Path(p)

        if path.is_dir():
            # Workspace directory — read final-report.md
            report = path / "final-report.md"
            if report.exists():
                refs.append({
                    "type": "text",
                    "name": f"{path.name}/final-report.md",
                    "content": report.read_text(encoding="utf-8"),
                })
            else:
                raise ValueError(
                    f"Directory {path} does not contain final-report.md.\n"
                    f"   Directory references only support reading a previous research workspace (which must contain final-report.md)."
                )
            continue

        if not path.is_file():
            raise ValueError(f"Reference file does not exist: {path}")

        # Fail-fast: raise immediately on explicitly unsupported formats
        # These formats are guaranteed to fail even through the UTF-8 fallback (zip containers, binaries).
        # Rather than skipping and letting the user discover the missing ref later, notify them now.
        UNSUPPORTED_EXTS = {
            # Office documents (zip containers, cannot be read as UTF-8)
            ".docx": "Word document", ".xlsx": "Excel spreadsheet", ".pptx": "PowerPoint presentation",
            ".doc":  "Word document (legacy)", ".xls": "Excel spreadsheet (legacy)",
            ".ppt":  "PowerPoint presentation (legacy)", ".odt": "OpenDocument text",
            ".ods":  "OpenDocument spreadsheet", ".odp": "OpenDocument presentation",
            ".rtf":  "RTF document",
            # E-books
            ".epub": "EPUB e-book", ".mobi": "Kindle e-book", ".azw3": "Kindle e-book",
            # Archives
            ".zip": "ZIP archive", ".tar": "TAR archive", ".gz": "Gzip archive",
            ".bz2": "Bzip2 archive", ".7z": "7-Zip archive", ".rar": "RAR archive",
            # Audio
            ".mp3": "MP3 audio", ".wav": "WAV audio", ".m4a": "M4A audio",
            ".ogg": "OGG audio", ".flac": "FLAC audio", ".aac": "AAC audio",
            # Video
            ".mp4": "MP4 video", ".mov": "MOV video", ".avi": "AVI video",
            ".mkv": "MKV video", ".webm": "WebM video", ".flv": "FLV video",
            # Other binary
            ".exe": "executable", ".dll": "dynamic-link library", ".so": "shared library",
            ".dmg": "macOS disk image", ".iso": "ISO image",
        }
        suffix = path.suffix.lower()
        if suffix in UNSUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported file format: {path.name} ({UNSUPPORTED_EXTS[suffix]})\n"
                f"   Currently supported:\n"
                f"      - Plain text: .md, .txt, .json, .yaml, .csv, .html, source code, etc.\n"
                f"      - PDF: .pdf (text only; charts/images are ignored)\n"
                f"      - Images: .png, .jpg, .jpeg, .gif, .webp\n"
                f"   To supply an Office document, first convert it to PDF or paste it as plain text."
            )

        mime, _ = mimetypes.guess_type(str(path))

        if mime and mime.startswith("image/"):
            # Image -> base64
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            refs.append({
                "type": "image",
                "name": path.name,
                "mime": mime,
                "data": b64,
            })

        elif mime == "application/pdf":
            # PDF: extract plain text and explicitly tell the user about the limitations
            # Users often assume the LLM can "see" charts/layout in a PDF; in reality only text is read.
            # Print a reminder to stderr on read so the user is aware of the limitations.
            print(
                f"PDF detected: {path.name}\n"
                f"   Only the 'text content' will be read. The following is ignored:\n"
                f"       - Images, charts, diagrams\n"
                f"       - Table layout structure (text can still be read, but may be misaligned)\n"
                f"       - Scanned files without an OCR text layer cannot be read at all\n"
                f"   If critical charts exist, it is recommended to also provide them as PNG/JPG.",
                file=sys.stderr,
            )
            text = _extract_pdf_text(path)
            if not text.strip():
                raise ValueError(
                    f"Unable to extract any text from PDF: {path.name}\n"
                    f"   Possible causes: scanned file without OCR text layer / encrypted / corrupted file."
                )
            refs.append({
                "type": "text",
                "name": path.name,
                "content": text,
            })

        else:
            # Other files: try reading as UTF-8 (plain-text extensions fall here)
            try:
                content = path.read_text(encoding="utf-8")
                refs.append({
                    "type": "text",
                    "name": path.name,
                    "content": content,
                })
            except UnicodeDecodeError:
                # Unreadable means the file is not UTF-8 text (possibly a binary format not in the blacklist
                # or a non-UTF-8 encoding such as Big5/GBK); raise in either case.
                raise ValueError(
                    f"Unable to read {path.name} as UTF-8.\n"
                    f"   It may be a binary file not explicitly listed as supported,\n"
                    f"   or a non-UTF-8-encoded text file (e.g. Big5, GBK).\n"
                    f"   Please convert it to UTF-8 or save it as a supported format."
                )

    return refs


# ---------------------------------------------------------------------------
# Project directory scan — feeds user's local project into phase0 brief
# ---------------------------------------------------------------------------

# Extensions worth reading for "what is this project?" — docs first, then
# manifests, then source. PDFs / images are excluded because a recursive
# project scan should stay lightweight (use --ref for those).
_PROJECT_SCAN_EXTS: set[str] = {
    # Docs (highest priority)
    ".md", ".markdown", ".txt", ".rst",
    # Manifests / config
    ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini",
    # Source
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".php", ".sh",
}

# Directory names to skip wholesale — dependency / build / cache dirs that
# would blow the scan budget with machine-generated noise and rarely carry
# project intent.
_PROJECT_SCAN_SKIP_DIRS: set[str] = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", ".next", ".cache",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode",
}

# Scan limits — keeps the LLM synthesize context sane. Whisper plan P0-5
# calls for 50 files / 10K chars-per-file / 200K total; see the plan for
# rationale.
_PROJECT_SCAN_MAX_FILES = 50
_PROJECT_SCAN_PER_FILE_CHARS = 10_000
_PROJECT_SCAN_TOTAL_CHARS = 200_000


def _score_project_file(path: Path, root: Path) -> tuple[int, int, str]:
    """Sort key so docs outrank source, and shallower paths outrank deep ones.

    Returns (priority, depth, name). Lower priority wins.
      - 0: README / OVERVIEW / CLAUDE.md at root (project-describing docs)
      - 1: other .md / .txt / .rst
      - 2: manifests (.toml / .json / .yaml / etc.)
      - 3: source files
    """
    name = path.name.lower()
    suffix = path.suffix.lower()
    depth = len(path.relative_to(root).parts)

    doc_stems = {"readme", "overview", "claude", "agents", "contributing", "architecture"}
    if suffix in {".md", ".markdown", ".txt", ".rst"} and path.stem.lower() in doc_stems:
        return (0, depth, name)
    if suffix in {".md", ".markdown", ".txt", ".rst"}:
        return (1, depth, name)
    if suffix in {".toml", ".json", ".yaml", ".yml", ".cfg", ".ini"}:
        return (2, depth, name)
    return (3, depth, name)


def scan_project_dir(root: str) -> list[dict]:
    """Recursively read a user's project directory into ref dicts.

    Honours ``_PROJECT_SCAN_SKIP_DIRS`` (won't descend into node_modules etc.),
    ``_PROJECT_SCAN_EXTS`` (only text-like extensions), per-file cap of
    ``_PROJECT_SCAN_PER_FILE_CHARS``, total-budget cap of
    ``_PROJECT_SCAN_TOTAL_CHARS``, and a hard file-count cap of
    ``_PROJECT_SCAN_MAX_FILES``.

    Returns a list of ref dicts shaped like ``read_reference_files`` output —
    ``{"type": "text", "name": <relative path>, "content": <text>}`` — so
    ``synthesize_research_topic`` can consume them unchanged.

    Files are ordered by priority (docs first, then manifests, then source),
    tiebroken by depth (shallower wins) then filename. Truncated files get a
    trailing marker so the LLM knows the tail was dropped.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"--project-dir is not a directory: {root}")

    candidates: list[Path] = []
    for path in root_path.rglob("*"):
        if any(part in _PROJECT_SCAN_SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _PROJECT_SCAN_EXTS:
            continue
        candidates.append(path)

    candidates.sort(key=lambda p: _score_project_file(p, root_path))

    refs: list[dict] = []
    total_chars = 0
    for path in candidates:
        if len(refs) >= _PROJECT_SCAN_MAX_FILES:
            break
        if total_chars >= _PROJECT_SCAN_TOTAL_CHARS:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary / non-UTF-8 / permission error — silently skip; project
            # scan is best-effort, not fail-fast (unlike --ref).
            continue
        truncated = False
        if len(text) > _PROJECT_SCAN_PER_FILE_CHARS:
            text = text[:_PROJECT_SCAN_PER_FILE_CHARS] + "\n\n...[truncated]..."
            truncated = True
        remaining = _PROJECT_SCAN_TOTAL_CHARS - total_chars
        if len(text) > remaining:
            text = text[:remaining] + "\n\n...[truncated to fit total budget]..."
            truncated = True
        rel = path.relative_to(root_path).as_posix()
        refs.append({
            "type": "text",
            "name": f"project-dir/{rel}",
            "content": text,
        })
        total_chars += len(text)
    return refs


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF."""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pymupdf is not installed; unable to extract PDF text. Run pip install pymupdf.")
        return f"(PDF file {path.name}; pymupdf must be installed to extract text)"
    except Exception as e:
        logger.warning(f"PDF extraction failed {path}: {e}")
        return f"(PDF extraction failed: {e})"


def refs_to_message_content(refs: list[dict]) -> list[dict]:
    """Convert refs into LangChain multimodal content blocks.

    Used in the HumanMessage(content=blocks) form, supporting mixed text+image.
    """
    blocks = []
    for ref in refs:
        if ref["type"] == "text":
            blocks.append({
                "type": "text",
                "text": f"\n--- Reference document: {ref['name']} ---\n{ref['content']}",
            })
        elif ref["type"] == "image":
            blocks.append({
                "type": "text",
                "text": f"\n--- Reference image: {ref['name']} ---",
            })
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{ref['mime']};base64,{ref['data']}"},
            })
    return blocks


# ---------------------------------------------------------------------------
# Research brief integration (topic + refs + clarifications -> full_research_topic)
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = FOCUSED_EXEC_PROMPT + """You are a research-requirements analyst. Integrate the information below into a structured research brief.

This brief will be the core instruction for the entire research pipeline; all subsequent search, analysis, and reporting will be executed according to it.
Describe the requester's needs in the third person so that anyone reading this brief can independently understand the full research requirement.

## Integration rules
1. Fuse the user's original topic, the key information from the reference documents, and every detail from the clarification Q&A into a complete narrative.
2. Keep the specific data points, conclusions, and opinions from the reference documents and tag them with the source filename.
3. Eliminate contradictions: when reference documents and the Q&A conflict, follow the Q&A (the user's latest view).
4. Fill in implicit requirements: infer aspects the user did not state explicitly but clearly needs from the Q&A.
5. Describe image content in words (data values, architecture diagram structure, trends, etc.).

## Output format

### Research objective
(One paragraph clearly explaining why this research is being conducted and what is expected.)

### Core questions
(Bulleted list of the specific questions this research must answer.)

### Scope and constraints
(Time range, region, technical boundaries, exclusions, budget, and any other hard constraints.)

### Evaluation criteria
(Which dimensions to use for judgment, how to compare, and the definition of success.)

### Known context
(Known facts extracted from the reference documents and Q&A to avoid redundant searching.)

### Output requirements
(Report format, audience, depth preference, and any special requirements.)"""


async def synthesize_research_topic(
    topic: str,
    refs: list[dict],
    clarifications: list[dict],
) -> str:
    """Integrate topic + refs + clarifications -> full_research_topic.

    One-shot call, executed after Phase 0 clarification completes.
    The resulting full_research_topic serves as fixed context for the entire research run.

    Anti-pattern protection (LLM focus principle):
      - Scenario: the user may drop an entire long PDF / large text file as refs;
              a single one-shot call has no Iterative Refinement protection and is prone to LiM.
      - Strategy: warn at a soft limit of 30K tokens total text refs, truncate at a hard limit of 50K tokens.
              Image refs are left alone (image tokens are billed differently; leave it to the model).
    """
    import logging
    log = logging.getLogger(__name__)

    # refs size protection: accumulate text-token count and truncate above the limit
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
                    # Keep only ~1500 tokens (4500 chars) at head and tail
                    remaining = HARD_LIMIT - accumulated
                    if remaining > 3000:
                        keep_chars = (remaining - 500) * 3 // 2
                        head = content[:keep_chars]
                        tail = content[-keep_chars:] if len(content) > keep_chars * 2 else ""
                        if tail:
                            new_content = f"{head}\n\n...[middle omitted ~{ref_tokens - remaining} tokens to fit the 50K limit]...\n\n{tail}"
                        else:
                            new_content = head + "\n\n...[tail omitted to fit the 50K limit]..."
                        safe_refs.append({**ref, "content": new_content})
                        accumulated = HARD_LIMIT
                    truncated_count += 1
            else:
                # Pass images through
                safe_refs.append(ref)
        if accumulated > SOFT_LIMIT:
            log.warning(
                "[synthesize] refs total text %d tokens exceeds soft limit %d — consider splitting or summarizing first",
                accumulated,
                SOFT_LIMIT,
            )
        if truncated_count:
            log.warning(
                "[synthesize] %d refs were truncated (exceeded the 50K-token hard limit)",
                truncated_count,
            )

    # Assemble multimodal content blocks
    content_blocks: list[dict] = [
        {"type": "text", "text": f"## Original research topic\n\n{topic}"},
    ]

    # Reference documents (text + image)
    if safe_refs:
        ref_blocks = refs_to_message_content(safe_refs)
        content_blocks.extend(ref_blocks)

    # Clarification Q&A
    if clarifications:
        qa_text = "\n## Clarification Q&A\n\n"
        for i, qa in enumerate(clarifications, 1):
            qa_text += f"**Q{i}:** {qa['question']}\n**A{i}:** {qa['answer']}\n\n"
        content_blocks.append({"type": "text", "text": qa_text})

    # Prompt caching markers
    # Anthropic: cache_control on the system message (the whole SYNTHESIZE_PROMPT will be cached)
    # OpenAI: automatic prefix caching (automatically enabled when prompt > 1024 tokens)
    # Gemini: automatic prefix caching
    system_msg = SystemMessage(content=SYNTHESIZE_PROMPT)
    human_msg = HumanMessage(content=content_blocks)

    # role="writer" — planning / integration category, Claude Opus-led, with a fallback chain
    response = await safe_ainvoke_chain(
        role="writer",
        messages=[system_msg, human_msg],
        max_tokens=8192,
        temperature=0.2,
    )
    return response.content


# ---------------------------------------------------------------------------
# BM25 ranking (Query Expansion + ranking)
# ---------------------------------------------------------------------------

async def _expand_query(topic: str) -> str:
    """Use the LLM to generate an expanded query that improves BM25 recall.

    Includes: synonyms, related terms, cross-language equivalents, domain jargon.
    Cost is low (a single verifier chain call; output ~200 tokens).
    """
    response = await safe_ainvoke_chain(
        role="verifier",
        messages=[
            SystemMessage(content="""Based on the research topic, generate one expanded query string for information retrieval.
Include: synonyms, related terms, English equivalents, domain jargon, related concepts.
The goal is to improve the recall of BM25 keyword matching.
Output the query text directly; do not explain."""),
            HumanMessage(content=topic),
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.content


def _rank_sources_bm25(sources: list[str], query: str) -> list[str]:
    """Rank sources by BM25 relevance.

    Args:
        sources: list of source text strings
        query: expanded query string

    Returns:
        sorted sources (most relevant first)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 is not installed; skipping ranking. Run pip install rank-bm25.")
        return sources

    if not sources or not query:
        return sources

    # Tokenize (simple tokenization: one Chinese char per token, English split on whitespace)
    def tokenize(text: str) -> list[str]:
        import re
        # Mixed-language tokenization: English splits on whitespace + punctuation, Chinese one char per token
        tokens = []
        for segment in re.split(r'(\s+)', text):
            segment = segment.strip()
            if not segment:
                continue
            # If it contains CJK characters, split per character
            if any('\u4e00' <= c <= '\u9fff' for c in segment):
                tokens.extend(list(segment))
            else:
                tokens.extend(segment.lower().split())
        return tokens

    tokenized_sources = [tokenize(s) for s in sources]
    tokenized_query = tokenize(query)

    bm25 = BM25Okapi(tokenized_sources)
    scores = bm25.get_scores(tokenized_query)

    # Sort by score descending
    ranked = sorted(zip(scores, sources), reverse=True)
    return [s for _, s in ranked]


# ---------------------------------------------------------------------------
# Iterative Refinement core loop
# ---------------------------------------------------------------------------

ITERATIVE_SYSTEM = FOCUSED_EXEC_PROMPT + """You are a deep-research analyst. Your task is to integrate new search results into the current research draft according to the research brief.

## Integration rules

1. Review this round's new information one item at a time and decide whether it is relevant to the research task.
2. Relevant and valuable -> integrate into the appropriate place in the draft.
3. Duplicates existing draft content -> skip; but if the new information has more precise numbers or a newer date, replace the old content.
4. Conflicts with existing draft content -> keep both, mark with [conflict pending] and note the source.
5. Irrelevant or low-quality -> skip; do not add.

## Citation requirements

Every fact, datum, and opinion must be tagged with its source:
- Format: "content text [source: filename or URL]"
- Numbers must be quoted verbatim from the original; no rounding or rewording.
- Inferences must be labelled [inference] and state the facts they derive from.

## Draft structure

Preserve the following structure; new content is inserted at the end of the relevant section (do not change the order or content of existing sections):
1. One section per core question.
2. Within each section, arrange content as "facts -> data -> analysis -> conflicts/pending".
3. The end of a section may carry a [to be supplemented] marker.

## Output

Output the updated, complete draft directly. Do not output explanations, do not output a diff, do not say "what I updated".
Output only the draft itself."""


async def iterative_refine(
    sources: list[str],
    full_research_topic: str,
    system_prompt: str = "",
    extra_context: str = "",
    tier: str = "strong",
    provider: str | None = None,
    role: str | None = None,
) -> str:
    """Iterative Refinement core: incrementally integrate all sources into the draft.

    Decision flow:
      1. total_tokens < budget -> fit all at once in a single LLM call
      2. total_tokens >= budget -> loop, packing up to budget each round
      3. fixed_cost > budget -> send one source per round
      4. fixed_cost + single source > 100% context -> role mode raises; tier mode switches to the largest provider

    Args:
        sources: list of source text strings (original search result text)
        full_research_topic: the integrated research brief
        system_prompt: custom system prompt (defaults to ITERATIVE_SYSTEM).
            Each phase supplies its own prompt:
            - Phase 1b: adversarial fact-check prompt
            - Phase 2: report integration prompt (default)
            - Phase 3: final audit prompt
        extra_context: additional fixed context (e.g. claims or statements to verify);
            placed after research_topic and before the draft.
        tier: LLM tier ("strong" or "fast") — used only when role=None
        provider: override provider — used only when role=None
        role: role-based fallback chain mode (recommended).
            "writer"   -> Claude Opus -> Sonnet -> GPT-pro (planning / integration / reporting)
            "verifier" -> Gemini -> GPT-mini -> Claude Haiku (extraction / verification / audit)
            Once set, the internal path uses safe_ainvoke_chain (automatic fallback);
            context_limit / cache format are taken from the first model in the chain.

    Returns:
        the final draft / verification result / audit result
    """
    if not system_prompt:
        system_prompt = ITERATIVE_SYSTEM

    if role is not None:
        # Role mode: use the chain's first model's context limit; cache format uses the first provider
        primary_provider, _primary_model = _available_chain(role)[0]
        p = primary_provider
        context_limit = get_role_context_limit(role)
    else:
        p = provider or get_provider()
        context_limit = get_context_limit(p, tier)

    threshold = get_context_threshold()
    budget = int(context_limit * threshold)

    # extra_context amplifier check —
    # iterative_refine re-sends extra_context every round; in batched mode actual consumption = extra * rounds.
    # If the caller stuffs in the entire ledger / all claims / all statements, this becomes the main source of noise.
    # Soft limit 2K tokens (warning), hard limit 4K tokens (truncation protection).
    if extra_context:
        extra_tokens = estimate_tokens(extra_context)
        if extra_tokens > 2000:
            logger.warning(
                f"iterative_refine: extra_context is too large ({extra_tokens:,} tokens). "
                f"It is re-sent every round -> in batched mode actual consumption = {extra_tokens:,} * rounds. "
                f"Consider splitting the caller into multiple independent calls (per-section / per-group) or trimming the context."
            )
        if extra_tokens > 4000:
            char_cap = 12000  # ~4000 tokens
            extra_context = (
                extra_context[:char_cap]
                + "\n\n...[extra_context exceeds the 4K-token hard limit; truncated]..."
            )
            logger.warning(
                f"extra_context hard-truncated to ~{estimate_tokens(extra_context):,} tokens"
            )

    # Estimate tokens of the fixed portion
    fixed_prompt_tokens = estimate_tokens(system_prompt + full_research_topic + extra_context)
    total_source_tokens = sum(estimate_tokens(s) for s in sources)
    total_tokens = fixed_prompt_tokens + total_source_tokens

    logger.info(
        f"Context decision: fixed={fixed_prompt_tokens:,} sources={total_source_tokens:,} "
        f"total={total_tokens:,} budget={budget:,} ({threshold:.0%} of {context_limit:,})"
    )

    # --- Step 1: fit-all-at-once check ---
    if total_tokens < budget:
        logger.info("Fit-all-at-once mode: all sources sent in a single call")
        return await _single_pass(sources, full_research_topic, system_prompt, extra_context, tier, p, role)

    # --- Step 2+: Iterative Refinement ---
    logger.info(f"Iterative Refinement mode: {len(sources)} sources processed in batches")

    # BM25 ranking (most relevant processed first)
    expanded_query = await _expand_query(full_research_topic)
    sorted_sources = _rank_sources_bm25(sources, expanded_query)

    draft = ""
    processed = 0

    while processed < len(sorted_sources):
        # Compute this round's budget
        draft_tokens = estimate_tokens(draft)
        fixed_cost = fixed_prompt_tokens + draft_tokens
        remaining = budget - fixed_cost

        if remaining <= 0:
            # fixed_cost exceeds budget -> send one source per round
            remaining_for_one = context_limit - fixed_cost  # use the 100% limit, not the threshold

            if remaining_for_one <= 0:
                if role is not None:
                    # Role mode: the fallback chain itself picks an available provider;
                    # if it still exceeds the limit here, the research brief is too long — no way out.
                    raise RuntimeError(
                        f"[role={role}] fixed_prompt + draft ({fixed_cost:,} tokens) exceeds "
                        f"the chain primary model's 100% context limit ({context_limit:,} tokens). "
                        f"Shorten the research brief or raise --context-threshold."
                    )
                # Tier mode: even 100% context is not enough -> switch to the largest provider
                larger = find_largest_available_provider(tier)
                if larger:
                    larger_limit = get_context_limit(larger, tier)
                    remaining_for_one = larger_limit - fixed_cost
                    if remaining_for_one <= 0:
                        raise RuntimeError(
                            f"fixed_prompt + draft ({fixed_cost:,} tokens) exceeds the 100% context limit of the largest available provider "
                            f"({larger}, {larger_limit:,} tokens). "
                            f"Shorten the research brief or raise --context-threshold."
                        )
                    logger.warning(f"Switching to {larger} (context: {larger_limit:,}) to process remaining sources")
                    p = larger
                else:
                    raise RuntimeError(
                        f"fixed_prompt + draft ({fixed_cost:,} tokens) exceeds {p}'s 100% context limit "
                        f"({context_limit:,} tokens), and no larger provider is available."
                    )

            # Send one source per round
            source = sorted_sources[processed]
            source_tokens = estimate_tokens(source)
            if source_tokens > remaining_for_one:
                # Single source too long; take what fits (this is the only place truncation is allowed)
                char_limit = remaining_for_one * 3  # reverse-estimate character count
                source = source[:char_limit] + "\n\n[...this source was truncated because it was too long...]"
                logger.warning(f"Source {processed+1} too long ({source_tokens:,} tokens); truncated to {remaining_for_one:,} tokens")

            draft = await _refine_once(draft, [source], full_research_topic, system_prompt, extra_context, tier, p, role)
            processed += 1
        else:
            # Normal case: greedily pack multiple sources
            batch = []
            batch_tokens = 0
            while processed < len(sorted_sources):
                source = sorted_sources[processed]
                source_tokens = estimate_tokens(source)
                if batch and batch_tokens + source_tokens > remaining:
                    break  # this batch is full
                batch.append(source)
                batch_tokens += source_tokens
                processed += 1

            draft = await _refine_once(draft, batch, full_research_topic, system_prompt, extra_context, tier, p, role)

        logger.info(f"Processed {processed}/{len(sorted_sources)} sources, draft: {estimate_tokens(draft):,} tokens")

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
    """Fit-all-at-once mode: a single LLM call handles all sources."""
    all_sources = "\n\n---\n\n".join(
        f"### Source {i+1}\n{s}" for i, s in enumerate(sources)
    )

    # Prompt caching design:
    #   Anthropic: system message + the prefix of the human message (research_topic) are cached;
    #              cache_control must be added to the content block.
    #   OpenAI:    automatic prefix caching (enabled when prompt > 1024 tokens; no extra parameter needed)
    #   Gemini:    automatic prefix caching
    system_content = _build_system_with_cache(system_prompt, provider)

    extra_section = f"\n\n---\n\n{extra_context}" if extra_context else ""
    human_text = f"""## Research brief

{full_research_topic}{extra_section}

---

## Search results ({len(sources)} total)

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
    """One round of Iterative Refinement: draft + a batch of sources -> updated draft."""
    batch_text = "\n\n---\n\n".join(
        f"### Source {i+1}\n{s}" for i, s in enumerate(source_batch)
    )

    draft_section = draft if draft else "(No results yet; this is the first round.)"

    # Prompt caching design:
    #   fixed_prompt (SYSTEM + research brief + extra_context) is stable every round -> cached
    #   the draft grows each round but its prefix is stable -> partially cached
    #   source_batch is fresh every round -> not cached
    system_content = _build_system_with_cache(system_prompt, provider)

    extra_section = f"\n\n---\n\n{extra_context}" if extra_context else ""
    human_text = f"""## Research brief

{full_research_topic}{extra_section}

---

## Current accumulated draft

{draft_section}

---

## New information this round ({len(source_batch)} total)

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

    Anthropic: use the cache_control marker so the system prompt is cached.
               cache_control: {"type": "ephemeral"} means this block should be
               cached with a 5-minute TTL (managed automatically by Anthropic).
    OpenAI/Gemini: automatic prefix caching; no extra parameter needed, return a plain string.
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

    Mark research_topic as the cacheable prefix because:
      - It does not change across loop rounds.
      - It is usually the longest fixed portion.
      - After the cache hits, subsequent draft + sources are paid for incrementally.

    Anthropic: split into two content blocks; the first (research_topic) gets cache_control.
    OpenAI/Gemini: automatic prefix caching; no splitting needed.
    """
    if provider == "claude":
        # Find where research_topic ends within full_text and split into two segments
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
