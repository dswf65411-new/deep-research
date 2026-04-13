"""Main StateGraph definition — assembles all phases into the ResearchGraph.

Topology (ask mode, CLI direct):
  [START] → Phase0-Plan → Human Approval (interrupt) → Phase1a → Phase1b
                                                          ↑         │
                                                          └── fail ──┘
                                                                    │ pass
                                                                    ↓
                                                              Phase2 → Phase3 → [END]

Topology (noask mode, or skill mode after external clarify+approve):
  [START] → Phase0-Plan → Phase1a → Phase1b → ... → Phase3 → [END]

Note: Clarification (ask questions) is handled outside the graph by main.py.
      The graph receives clarifications via initial state and uses them in planning.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from deep_research.harness.gates import gate_check
from deep_research.nodes.phase0 import phase0_plan
from deep_research.nodes.phase1a import phase1a_search
from deep_research.nodes.phase1b import phase1b_verify, trigger_fallback_node
from deep_research.nodes.phase2 import phase2_integrate
from deep_research.nodes.phase3 import phase3_report
from deep_research.state import ResearchState


# ---------------------------------------------------------------------------
# Human-in-the-loop: interrupt after plan for approval (ask mode only)
# ---------------------------------------------------------------------------

async def human_approval(state: ResearchState) -> dict:
    """Pause execution and wait for human to approve the research plan.

    Uses LangGraph's interrupt() to suspend the graph.
    The caller resumes with Command(resume={"approved": True}) or
    Command(resume={"approved": False, "revised_plan": "..."}).
    """
    plan_summary = state.get("plan", "")

    response = interrupt({
        "type": "approve",
        "question": "請確認研究計畫，或提出修改。",
        "plan_summary": plan_summary,
        "workspace": state.get("workspace_path", ""),
    })

    if isinstance(response, dict):
        if response.get("approved"):
            return {}
        if "revised_plan" in response:
            return {"plan": response["revised_plan"]}

    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_plan(state: ResearchState) -> str:
    """After plan: ask mode → human approval, noask → straight to search."""
    if state.get("ask_mode", True):
        return "human_approval"
    return "phase1a"


def route_after_fallback(state: ResearchState) -> str:
    """Route after trigger_fallback_node.

    Priority:
    1. needs_refetch non-empty → focus-mode phase1a (via increment_iter)
    2. phase1b passed normally → phase2
    3. Max retries hit → phase2 (force through)
    4. Else → normal phase1a retry (via increment_iter)
    """
    needs = state.get("needs_refetch", [])
    if needs:
        return "increment_iter"

    result = state.get("phase1b_result", "fail")
    if result in ("pass", "max_retries"):
        return "phase2"

    gate = gate_check(state)
    if gate in ("pass", "max_retries"):
        return "phase2"

    return "increment_iter"


async def increment_iteration(state: ResearchState) -> dict:
    """Increment the iteration counter before looping back to Phase 1a."""
    current = state.get("iteration_count", 0)
    return {
        "iteration_count": current + 1,
        "execution_log": [f"迭代 {current + 1} → 回到 Phase 1a"],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_deep_research(checkpointer=None) -> StateGraph:
    """Build and compile the full research graph.

    Args:
        checkpointer: LangGraph checkpointer for persistence.
            Defaults to InMemorySaver if None.

    Returns:
        Compiled StateGraph ready for .ainvoke() or .astream().
    """
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("phase0_plan", phase0_plan)
    builder.add_node("human_approval", human_approval)
    builder.add_node("phase1a", phase1a_search)
    builder.add_node("phase1b", phase1b_verify)
    builder.add_node("trigger_fallback", trigger_fallback_node)
    builder.add_node("increment_iter", increment_iteration)
    builder.add_node("phase2", phase2_integrate)
    builder.add_node("phase3", phase3_report)

    # START → plan
    builder.add_edge(START, "phase0_plan")

    # Plan → route by ask_mode (approve or skip)
    builder.add_conditional_edges(
        "phase0_plan",
        route_after_plan,
        {
            "human_approval": "human_approval",
            "phase1a": "phase1a",
        },
    )

    # Approval → search
    builder.add_edge("human_approval", "phase1a")

    # Search → verify → fallback trigger
    builder.add_edge("phase1a", "phase1b")
    builder.add_edge("phase1b", "trigger_fallback")

    # Fallback trigger → conditional (focus-mode loop or proceed to phase2)
    builder.add_conditional_edges(
        "trigger_fallback",
        route_after_fallback,
        {
            "phase2": "phase2",
            "increment_iter": "increment_iter",
        },
    )
    builder.add_edge("increment_iter", "phase1a")

    # Phase 2 → Phase 3 → END
    builder.add_edge("phase2", "phase3")
    builder.add_edge("phase3", END)

    # Compile
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)
