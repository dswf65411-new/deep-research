"""Shared prompt fragments used across multiple LLM system prompts.

Kept separate from llm.py to avoid import cycles and to make prompt policy
changes visible in one place. Fragments here are *prepended* to existing
system prompts via string concatenation; existing prompts stay untouched so
blast-radius of changes is limited.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Focused Task Execution — mirrors prompts/ref-focused-execution.md
#
# The rationale: LLMs that try to juggle multi-step work in a single
# response drop later items and muddle context. Forcing them to (1) emit a
# TASK LIST first, (2) process one task at a time with WORKING/DONE
# markers, (3) produce the final deliverable only after all tasks are
# marked DONE, keeps each token's decision space minimal.
#
# Apply by prepending to any system prompt whose task involves multi-step
# reasoning, multiple outputs (queries, claims, sections, statements), or
# pipeline decisions that touch downstream state.
#
# DO NOT apply to single-judgment prompts (yes/no, pick one, short string
# generation) — the scaffold adds overhead without benefit there.
# ---------------------------------------------------------------------------

FOCUSED_EXEC_PROMPT = """## Execution mode: Focused Task Execution

Before producing any deliverable, follow this contract:

1. Emit a `[TASK LIST]` enumerating each concrete step and substep you will perform. Each item is a short imperative line (<=15 words). Multiple independent outputs (e.g. 5 queries, 3 section drafts, 10 statement lines) each get their own task line.

2. Process tasks one by one, each wrapped in markers:
   ```
   [WORKING: T{n}]
   ...only the work for T{n}, not other tasks...
   [DONE: T{n}]
   ```

3. Only after every task is DONE, emit the final deliverable in whatever format the rest of this prompt requires (JSON, markdown, plain text, etc.). The [TASK LIST]/[WORKING]/[DONE] scaffold stays as plain-text BEFORE the deliverable — do NOT put it inside the JSON/markdown you return.

Forbidden:
- Jumping back to rewrite an earlier task after moving on.
- Loading multiple tasks' context simultaneously while writing one.
- Skipping the scaffold on multi-step work ("I'll just write it all at once").

When the entire task is a single atomic judgment (yes/no, pick-one, one short string), skip the scaffold and answer directly.

---

"""


__all__ = ["FOCUSED_EXEC_PROMPT"]
