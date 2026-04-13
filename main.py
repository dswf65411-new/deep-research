#!/usr/bin/env python3
"""CLI entry point for deep-research LangGraph workflow.

Interaction modes:
    --ask    (default) Interactive: multi-round clarify + plan approval
    --noask            Autonomous: no user interaction

Execution modes:
    Direct CLI:  uses input() for interaction, single process
    Skill mode (--json):  turn-based JSON protocol via workspace/.state.json

Turn-based protocol (--json):
    {"status":"NEEDS_INPUT","workspace":"...","type":"clarify","questions":["..."],"round":1,"total_asked":3}
    {"status":"NEEDS_INPUT","workspace":"...","type":"clarify_retry","questions":["..."],"missing_indices":[0,2]}
    {"status":"NEEDS_INPUT","workspace":"...","type":"approve","plan_summary":"..."}
    {"status":"DONE","workspace":"..."}
    {"status":"DONE_ASK_FOLLOWUP","workspace":"...","message":"..."}
    {"status":"ERROR","error":"..."}

Exit codes:
    0 = normal (turn-based pipeline may still need more turns; read stdout JSON
        `status` field to know the next step):
          "DONE"              — research complete
          "NEEDS_INPUT"       — pipeline paused, waiting for user input
          "DONE_ASK_FOLLOWUP" — done, caller may offer a follow-up round
    1 = error (stdout JSON has {"status": "ERROR", "error": "..."})

Rationale: exit code 10 previously signalled NEEDS_INPUT, but Claude Code's
Bash tool rendered any non-zero exit as a red "Error" and misled users.
Now only true failures return non-zero; the stdout JSON is the single source
of truth for pipeline state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from deep_research.context import read_reference_files  # noqa: E402 — after load_dotenv

DEFAULT_MAX_QUESTIONS = 10


def format_references_as_context(refs: list[dict]) -> str:
    """Format reference files into a text context string for clarification phase.

    Args:
        refs: list from context.read_reference_files() — each has type/name/content or data
    """
    if not refs:
        return ""
    parts = []
    for ref in refs:
        if ref["type"] == "text":
            parts.append(f"### 參考資料：{ref['name']}\n\n{ref['content']}")
        elif ref["type"] == "image":
            parts.append(f"### 參考圖片：{ref['name']}（圖片內容，將由 LLM 視覺理解）")
    return "\n\n---\n\n".join(parts)


def refs_to_clarification(refs: list[dict]) -> dict | None:
    """Convert reference files into a single clarification entry for the Q&A phase.

    Args:
        refs: list from context.read_reference_files()
    """
    if not refs:
        return None
    context = format_references_as_context(refs)
    filenames = ", ".join(r["name"] for r in refs)
    return {
        "question": f"使用者提供的參考資料（{filenames}）",
        "answer": context,
    }


# ---------------------------------------------------------------------------
# Workspace state persistence
# ---------------------------------------------------------------------------

def save_state(workspace: str, state: dict) -> None:
    path = Path(workspace) / ".state.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(workspace: str) -> dict | None:
    path = Path(workspace) / ".state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Run full graph (Phase 0 plan → Phase 1a → 1b → 2 → 3)
# ---------------------------------------------------------------------------

async def run_graph(
    topic: str,
    depth: str,
    budget: int,
    ask_mode: bool,
    clarifications: list[dict] | None = None,
    refs: list[dict] | None = None,
    interactive_cli: bool = False,
) -> str:
    """Run the research graph. Returns workspace path.

    Args:
        refs: 參考文件（from context.read_reference_files），傳入 graph state
              供 phase0 的 synthesize_research_topic 使用（支援文字+圖片+PDF）。
    """
    import uuid
    from langgraph.types import Command
    from deep_research.graph import build_deep_research

    graph = build_deep_research()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    effective_clarifications = list(clarifications or [])

    initial_state = {
        "topic": topic,
        "depth": depth,
        "search_budget": budget,
        "search_count": 0,
        "iteration_count": 0,
        "ask_mode": ask_mode,
        "clarifications": effective_clarifications,
        "refs": refs or [],
        "sources": [],
        "claims": [],
        "report_sections": [],
        "execution_log": [],
        "blockers": [],
        "coverage_status": {},
    }

    _log(f"[deep-research] 啟動研究：{topic}")
    _log(f"[deep-research] 模式={'互動' if ask_mode else '自動'}, 深度={depth}, 預算={budget}")

    async for event in graph.astream(initial_state, config, stream_mode="updates"):
        for node_name, update in event.items():
            if isinstance(update, dict):
                for log_entry in update.get("execution_log", []):
                    _log(f"  [{node_name}] {log_entry}")

    # Check for plan approval interrupt
    snapshot = graph.get_state(config)
    if snapshot.next and "human_approval" in snapshot.next:
        if not ask_mode:
            resume_value = {"approved": True}
        elif interactive_cli:
            # Infinite loop until user explicitly approves
            while True:
                plan = snapshot.values.get("plan", "")
                print("\n" + "=" * 60)
                print("研究計畫：")
                print("=" * 60)
                print(plan)
                print("\n是否開始研究？(y/n): ", end="", flush=True)
                confirm = input().strip().lower()

                if confirm in ("y", "yes", "是"):
                    resume_value = {"approved": True}
                    break
                elif confirm in ("n", "no", "否"):
                    print("請說明需要修改的地方: ", end="", flush=True)
                    revision = input().strip()
                    if not revision:
                        print("⚠️  請提供修改說明")
                        continue
                    # Re-generate plan with revision
                    from deep_research.nodes.phase0 import phase0_plan_standalone
                    new_plan = await phase0_plan_standalone(
                        topic=topic, depth=depth, budget=budget,
                        clarifications=clarifications or [],
                        workspace_path=snapshot.values.get("workspace_path", ""),
                    )
                    # Update snapshot plan for next display
                    snapshot.values["plan"] = new_plan
                    continue
                else:
                    print("⚠️  請回答 y（是）或 n（否）")
                    continue
        else:
            resume_value = {"approved": True}

        async for event in graph.astream(
            Command(resume=resume_value), config, stream_mode="updates"
        ):
            for node_name, update in event.items():
                if isinstance(update, dict):
                    for log_entry in update.get("execution_log", []):
                        _log(f"  [{node_name}] {log_entry}")

    final_state = graph.get_state(config)
    state_values = final_state.values if hasattr(final_state, "values") else {}
    workspace = state_values.get("workspace_path", "")

    _log(f"\n[deep-research] 研究完成")
    if workspace:
        _log(f"[deep-research] 報告位於：{workspace}/final-report.md")

    return workspace


# ---------------------------------------------------------------------------
# CLI interactive mode
# ---------------------------------------------------------------------------

async def cli_interactive(
    topic: str, depth: str, budget: int, ask_mode: bool, max_questions: int,
    ref_paths: list[str] | None = None,
) -> None:
    """Interactive CLI with multi-round clarification + follow-up loop."""

    current_topic = topic
    current_refs = read_reference_files(ref_paths) if ref_paths else []

    while True:
        # --- Clarification phase ---
        clarifications = await _cli_clarify_loop(
            current_topic, ask_mode, max_questions, current_refs,
        )

        # --- Run research ---
        # refs 傳入 graph state → phase0 用於 synthesize_research_topic（支援圖片+PDF）
        workspace = await run_graph(
            topic=current_topic, depth=depth, budget=budget,
            ask_mode=ask_mode, clarifications=clarifications,
            refs=current_refs, interactive_cli=True,
        )

        if workspace:
            print(f"\n完成。報告位於：{workspace}/final-report.md")

        if not ask_mode:
            break

        # --- Follow-up loop ---
        print(f"\n{'=' * 60}")
        print("有沒有想要以這份研究為基礎，深入研究的方向？")
        print("（直接輸入想深入的方向，或輸入 n 結束）")
        print(f"{'=' * 60}")
        print("  > ", end="", flush=True)
        followup = input().strip()

        if not followup or followup.lower() in ("n", "no", "否", "沒有"):
            break

        # New cycle: previous workspace's report becomes a reference
        current_topic = followup
        current_refs = read_reference_files([workspace]) if workspace else []  # workspace dir → reads final-report.md
        _log(f"\n[deep-research] 開始追加研究：{followup}")
        _log(f"[deep-research] 參考前次研究：{workspace}")


async def _cli_clarify_loop(
    topic: str, ask_mode: bool, max_questions: int, refs: list[dict],
) -> list[dict]:
    """Run multi-round clarification for CLI mode. Returns clarifications list."""
    from deep_research.nodes.phase0 import (
        generate_questions, validate_answers, judge_clarity,
    )

    clarifications: list[dict] = []

    # Add reference files as initial context
    ref_entry = refs_to_clarification(refs)
    if ref_entry:
        clarifications.append(ref_entry)

    if not ask_mode:
        return clarifications

    total_asked = 0
    round_num = 0
    judge_suggested: list[str] | None = None  # Layer 3: 上輪 Judge 針對失敗維度的追問

    while True:
        round_num += 1
        remaining = max_questions - total_asked
        if remaining <= 0:
            _log(f"[deep-research] 已達問題上限 {max_questions}，進入規劃")
            break

        # Adaptive Escalation: 優先使用 Judge 針對缺失維度的追問
        if judge_suggested:
            questions = judge_suggested[:remaining]
            reasoning = "根據上一輪 Judge 評估的不足維度生成的針對性追問"
            judge_suggested = None
        else:
            questions, reasoning = await generate_questions(
                topic, clarifications, max_questions=remaining, round_num=round_num,
            )

        if not questions:
            _log("[deep-research] LLM 認為主題已足夠明確")
            break

        total_asked += len(questions)

        print(f"\n{'=' * 60}")
        print(f"第 {round_num} 輪澄清（已問 {total_asked - len(questions)} 題，本輪 {len(questions)} 題，上限 {max_questions}）")
        print(f"{'=' * 60}")

        round_answers = {}
        for i, q in enumerate(questions):
            while True:
                print(f"\n  [{i+1}] {q}")
                print("  回答: ", end="", flush=True)
                ans = input().strip()
                if ans:
                    round_answers[str(i)] = ans
                    break
                print("  ⚠️  請提供回答（不能為空）")

        valid, _ = validate_answers(questions, round_answers)
        clarifications.extend(valid)

        is_clear, judge_reason, suggested = await judge_clarity(topic, clarifications)
        if is_clear:
            _log(f"[deep-research] Judge 判定：主題已足夠清楚 — {judge_reason}")
            break
        else:
            _log(f"[deep-research] Judge 判定：還需要更多澄清 — {judge_reason}")
            if remaining - len(questions) <= 0:
                _log(f"[deep-research] 但已達問題上限 {max_questions}，繼續進行")
                break
            judge_suggested = suggested  # 保存到下輪使用

    return clarifications


# ---------------------------------------------------------------------------
# Skill mode: turn-based JSON protocol
# ---------------------------------------------------------------------------

async def skill_mode(
    topic: str | None,
    depth: str,
    budget: int,
    ask_mode: bool,
    max_questions: int,
    resume_workspace: str | None,
    answer: str | None,
) -> None:
    """Handle --json skill mode."""
    from deep_research.nodes.phase0 import (
        generate_questions, validate_answers, judge_clarity,
        phase0_plan_standalone,
    )

    if resume_workspace:
        state = load_state(resume_workspace)
        if state is None:
            _json_out({"status": "ERROR", "error": f"找不到 state：{resume_workspace}/.state.json"})
            sys.exit(1)

        phase = state.get("phase", "unknown")

        try:
            parsed_answer = json.loads(answer) if answer else {}
        except json.JSONDecodeError:
            parsed_answer = answer

        if phase in ("clarify", "clarify_retry"):
            questions = state.get("questions", [])

            if phase == "clarify_retry":
                # Retry: only validate the missing questions
                missing_indices = state.get("missing_indices", [])
                retry_questions = [questions[i] for i in missing_indices]
                valid, still_missing = validate_answers(retry_questions, parsed_answer)

                if still_missing:
                    # Still missing — ask again (does NOT count toward total)
                    real_missing = [missing_indices[i] for i in still_missing]
                    state["missing_indices"] = real_missing
                    state["phase"] = "clarify_retry"
                    save_state(resume_workspace, state)

                    retry_qs = [questions[i] for i in real_missing]
                    _json_out({
                        "status": "NEEDS_INPUT",
                        "workspace": resume_workspace,
                        "type": "clarify_retry",
                        "questions": retry_qs,
                        "missing_indices": real_missing,
                        "message": "以下問題的回答為空或格式不正確，請重新回答：",
                    })
                    sys.exit(0)

                state["clarifications"] = state.get("clarifications", []) + valid

            else:
                # Normal clarify answer
                valid, missing = validate_answers(questions, parsed_answer)

                if missing:
                    # Some answers missing — retry those (does NOT count toward total)
                    state["phase"] = "clarify_retry"
                    state["missing_indices"] = missing
                    state["clarifications"] = state.get("clarifications", []) + valid
                    save_state(resume_workspace, state)

                    missing_qs = [questions[i] for i in missing]
                    _json_out({
                        "status": "NEEDS_INPUT",
                        "workspace": resume_workspace,
                        "type": "clarify_retry",
                        "questions": missing_qs,
                        "missing_indices": missing,
                        "message": "以下問題的回答為空或格式不正確，請重新回答：",
                    })
                    sys.exit(0)

                state["clarifications"] = state.get("clarifications", []) + valid

            # All answers valid — check with Judge
            is_clear, judge_reason, suggested = await judge_clarity(
                state["topic"], state["clarifications"]
            )

            total_asked = state.get("total_asked", 0)
            mq = state.get("max_questions", max_questions)
            remaining = mq - total_asked

            if not is_clear and remaining > 0 and suggested:
                # Judge says need more — generate next round
                round_num = state.get("round", 1) + 1
                next_questions = suggested[:remaining]
                total_asked += len(next_questions)

                state["phase"] = "clarify"
                state["questions"] = next_questions
                state["round"] = round_num
                state["total_asked"] = total_asked
                save_state(resume_workspace, state)

                _json_out({
                    "status": "NEEDS_INPUT",
                    "workspace": resume_workspace,
                    "type": "clarify",
                    "questions": next_questions,
                    "round": round_num,
                    "total_asked": total_asked,
                    "max_questions": mq,
                    "judge_reasoning": judge_reason,
                })
                sys.exit(0)

            # Clear enough (or hit limit) → generate plan
            await _do_plan_and_maybe_pause(state, ask_mode)

        elif phase == "approve":
            if isinstance(parsed_answer, dict) and parsed_answer.get("approved"):
                await _run_full(state)
            elif isinstance(parsed_answer, dict) and parsed_answer.get("approved") is False:
                # User rejected — needs revision description
                revision = parsed_answer.get("revision", "")
                if not revision:
                    # No revision provided — ask again
                    _json_out({
                        "status": "NEEDS_INPUT",
                        "workspace": resume_workspace,
                        "type": "approve_revision",
                        "message": "請說明需要修改研究計畫的哪些部分：",
                    })
                    sys.exit(0)
                # Re-generate plan with revision context
                state["clarifications"] = state.get("clarifications", []) + [
                    {"question": "使用者對研究計畫的修改要求", "answer": revision}
                ]
                plan = await phase0_plan_standalone(
                    topic=state["topic"],
                    depth=state["depth"],
                    budget=state["budget"],
                    clarifications=state["clarifications"],
                    workspace_path=state["workspace"],
                )
                state["phase"] = "approve"
                state["plan"] = plan
                save_state(state["workspace"], state)
                _json_out({
                    "status": "NEEDS_INPUT",
                    "workspace": state["workspace"],
                    "type": "approve",
                    "plan_summary": plan,
                    "message": "已根據修改意見重新生成計畫，請確認：",
                })
                sys.exit(0)
            elif phase == "approve_revision" if False else isinstance(parsed_answer, str):
                # String answer = revision text, re-generate
                state["clarifications"] = state.get("clarifications", []) + [
                    {"question": "使用者對研究計畫的修改要求", "answer": parsed_answer}
                ]
                plan = await phase0_plan_standalone(
                    topic=state["topic"],
                    depth=state["depth"],
                    budget=state["budget"],
                    clarifications=state["clarifications"],
                    workspace_path=state["workspace"],
                )
                state["phase"] = "approve"
                state["plan"] = plan
                save_state(state["workspace"], state)
                _json_out({
                    "status": "NEEDS_INPUT",
                    "workspace": state["workspace"],
                    "type": "approve",
                    "plan_summary": plan,
                    "message": "已根據修改意見重新生成計畫，請確認：",
                })
                sys.exit(0)
            else:
                await _run_full(state)

        elif phase == "approve_revision":
            # User provided revision text
            revision = parsed_answer if isinstance(parsed_answer, str) else str(parsed_answer)
            state["clarifications"] = state.get("clarifications", []) + [
                {"question": "使用者對研究計畫的修改要求", "answer": revision}
            ]
            plan = await phase0_plan_standalone(
                topic=state["topic"],
                depth=state["depth"],
                budget=state["budget"],
                clarifications=state["clarifications"],
                workspace_path=state["workspace"],
            )
            state["phase"] = "approve"
            state["plan"] = plan
            save_state(state["workspace"], state)
            _json_out({
                "status": "NEEDS_INPUT",
                "workspace": state["workspace"],
                "type": "approve",
                "plan_summary": plan,
                "message": "已根據修改意見重新生成計畫，請確認：",
            })
            sys.exit(0)

        elif phase == "followup":
            # User wants follow-up research (or says no)
            followup_text = parsed_answer if isinstance(parsed_answer, str) else str(parsed_answer)

            if followup_text.lower().strip() in ("n", "no", "否", "沒有", ""):
                _json_out({"status": "DONE", "workspace": state["workspace"]})
                sys.exit(0)

            # Start new research with follow-up topic + previous report as ref
            from deep_research.tools.workspace import create_workspace
            prev_workspace = state["workspace"]
            new_workspace = create_workspace(followup_text)

            new_state = {
                "topic": followup_text,
                "depth": state.get("depth", "deep"),
                "budget": state.get("budget", 150),
                "ask_mode": state.get("ask_mode", True),
                "max_questions": state.get("max_questions", max_questions),
                "workspace": new_workspace,
                "ref_workspace": prev_workspace,
                "clarifications": [],
                "total_asked": 0,
                "round": 0,
            }

            if state.get("ask_mode", True):
                # Start clarification for the follow-up topic
                ref_report = read_reference_report(prev_workspace)
                initial_clarifications = []
                if ref_report:
                    initial_clarifications.append({
                        "question": "前次研究報告（作為本次研究的參考背景）",
                        "answer": ref_report,
                    })
                new_state["clarifications"] = initial_clarifications

                mq = new_state["max_questions"]
                questions, reasoning = await generate_questions(
                    followup_text, initial_clarifications,
                    max_questions=mq, round_num=1,
                )

                if questions:
                    new_state["phase"] = "clarify"
                    new_state["questions"] = questions
                    new_state["round"] = 1
                    new_state["total_asked"] = len(questions)
                    save_state(new_workspace, new_state)

                    _json_out({
                        "status": "NEEDS_INPUT",
                        "workspace": new_workspace,
                        "type": "clarify",
                        "questions": questions,
                        "reasoning": reasoning,
                        "round": 1,
                        "total_asked": len(questions),
                        "max_questions": mq,
                        "ref_workspace": prev_workspace,
                    })
                    sys.exit(0)
                else:
                    await _do_plan_and_maybe_pause(new_state, True)
            else:
                await _run_full(new_state)

        else:
            _json_out({"status": "ERROR", "error": f"未知 phase：{phase}"})
            sys.exit(1)

    else:
        # New run
        if not topic:
            _json_out({"status": "ERROR", "error": "缺少研究主題"})
            sys.exit(1)

        from deep_research.tools.workspace import create_workspace
        workspace_path = create_workspace(topic)

        base_state = {
            "topic": topic,
            "depth": depth,
            "budget": budget,
            "ask_mode": ask_mode,
            "max_questions": max_questions,
            "workspace": workspace_path,
            "clarifications": [],
            "total_asked": 0,
            "round": 0,
        }

        if ask_mode:
            questions, reasoning = await generate_questions(
                topic, [], max_questions=max_questions, round_num=1,
            )

            if questions:
                base_state["phase"] = "clarify"
                base_state["questions"] = questions
                base_state["round"] = 1
                base_state["total_asked"] = len(questions)
                save_state(workspace_path, base_state)

                _json_out({
                    "status": "NEEDS_INPUT",
                    "workspace": workspace_path,
                    "type": "clarify",
                    "questions": questions,
                    "reasoning": reasoning,
                    "round": 1,
                    "total_asked": len(questions),
                    "max_questions": max_questions,
                })
                sys.exit(0)
            else:
                await _do_plan_and_maybe_pause(base_state, ask_mode)
        else:
            workspace = await run_graph(
                topic=topic, depth=depth, budget=budget,
                ask_mode=False, interactive_cli=False,
            )
            _json_out({"status": "DONE", "workspace": workspace})


async def _do_plan_and_maybe_pause(state: dict, ask_mode: bool) -> str:
    from deep_research.nodes.phase0 import phase0_plan_standalone

    plan = await phase0_plan_standalone(
        topic=state["topic"],
        depth=state["depth"],
        budget=state["budget"],
        clarifications=state.get("clarifications", []),
        workspace_path=state["workspace"],
    )

    if ask_mode:
        state["phase"] = "approve"
        state["plan"] = plan
        save_state(state["workspace"], state)

        _json_out({
            "status": "NEEDS_INPUT",
            "workspace": state["workspace"],
            "type": "approve",
            "plan_summary": plan,
        })
        sys.exit(0)
    else:
        return await _run_full(state)


async def _run_full(state: dict) -> str:
    # Reference files are already baked into clarifications
    workspace = await run_graph(
        topic=state["topic"],
        depth=state["depth"],
        budget=state["budget"],
        ask_mode=False,
        clarifications=state.get("clarifications", []),
        interactive_cli=False,
    )

    if state.get("ask_mode", True):
        # Ask if user wants follow-up research
        state["phase"] = "followup"
        state["workspace"] = workspace
        save_state(workspace, state)

        _json_out({
            "status": "DONE_ASK_FOLLOWUP",
            "workspace": workspace,
            "message": "研究完成。有沒有想要以這份研究為基礎，深入研究的方向？（輸入想研究的方向，或 'n' 結束）",
        })
        sys.exit(0)
    else:
        _json_out({"status": "DONE", "workspace": workspace})
        return workspace


def _json_out(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(
        description="deep-research: LangGraph deep research workflow"
    )
    parser.add_argument("topic", nargs="*", help="研究主題")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--ask", action="store_true", default=True, help="互動模式（預設）")
    mode_group.add_argument("--noask", action="store_true", help="自動模式")

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--model", type=str, default="auto",
                        help="LLM 模型選擇：auto（預設，按 claude>gemini>openai 偵測）、"
                             "claude/gemini/openai（用該家最強模型）、"
                             "或完整模型版號如 gemini-3.1-pro")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f"澄清問題上限（預設 {DEFAULT_MAX_QUESTIONS}）")

    parser.add_argument("--context-threshold", type=float, default=0.3,
                        help="Context 填充率閾值（0.0-1.0，預設 0.3）。"
                             "超過此比例時切換為 Iterative Refinement 模式。"
                             "值越小越保守（精度高但慢），值越大越激進（快但可能 lost in middle）")
    parser.add_argument("--ref", type=str, nargs="+", metavar="FILE",
                        help="參考資料（前次研究 workspace 目錄、文件、筆記等，可多個）")
    parser.add_argument("--resume", type=str, metavar="WORKSPACE")
    parser.add_argument("--answer", type=str)
    parser.add_argument("--json", action="store_true", help="JSON 輸出（skill 用）")

    parser.add_argument("--auto", action="store_true", help="（已棄用，請用 --noask）")

    args = parser.parse_args()

    if args.auto:
        args.noask = True

    ask_mode = not args.noask
    topic = " ".join(args.topic) if args.topic else None

    if not topic and not args.resume:
        parser.error("請提供研究主題，或用 --resume 繼續執行")

    depth = "quick" if args.quick else "standard" if args.standard else "deep"
    budget = args.budget or {"quick": 30, "standard": 60, "deep": 150}[depth]

    from deep_research.llm import set_provider, set_context_threshold
    set_provider(args.model)
    set_context_threshold(args.context_threshold)

    try:
        if args.json or args.resume:
            asyncio.run(skill_mode(
                topic=topic, depth=depth, budget=budget,
                ask_mode=ask_mode, max_questions=args.max_questions,
                resume_workspace=args.resume, answer=args.answer,
            ))
        else:
            asyncio.run(cli_interactive(
                topic=topic, depth=depth, budget=budget,
                ask_mode=ask_mode, max_questions=args.max_questions,
                ref_paths=args.ref,
            ))
    except KeyboardInterrupt:
        _log("\n[deep-research] 使用者中斷")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        if args.json or args.resume:
            _json_out({"status": "ERROR", "error": str(e)})
        else:
            _log(f"\n[deep-research] 錯誤：{e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
