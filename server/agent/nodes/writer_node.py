from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from ...llm import make_llm
from ..state import GraphState

# ---------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------
WRITER_SYSTEM = """
You are a senior data-science copilot writing concise, technical Markdown summaries.

Core principles:
- Derive all reasoning ONLY from structured facts and the user's intent.
- Be analytical but natural.
- If user specifies a task family (classification, clustering, etc.), lock recommendations to that.
- Include: Dataset Overview, Recommended Algorithms, Preprocessing, Evaluation Strategy, and Next Steps.
"""

CONCEPT_SYSTEM = """
You are a concise data-science mentor. For conceptual DS/ML questions (without dataset),
answer with:
1) One-line definition
2) Practical intuition
3) 3–5 common examples
4) When to use vs. when not to use
Keep it under 10 sentences.
"""

# LLMs for writing and concept detection
_writer = make_llm(system_prompt=WRITER_SYSTEM, temperature=0.45)
_concept_llm = make_llm(system_prompt=CONCEPT_SYSTEM, temperature=0.2)

# ---------------------------------------------------------------------
# LLM-based concept gate setup
# ---------------------------------------------------------------------
_CLASSIFIER_SYSTEM = """
You are a strict router for a data-science assistant.

Return EXACTLY one token: CONCEPT or DATA.

Rules:
- CONCEPT = when the user asks about ML/DS theory, definitions, algorithms, comparisons,
  intuition, formulas, or best practices (e.g., "What is gradient descent?", "Explain ROC vs AUC").
- DATA = when the user wants analysis or computation on a dataset (e.g., "Run EDA", "Suggest model for this CSV").
- Always return one of these tokens — no other text.
"""

_classifier_llm = make_llm(system_prompt=_CLASSIFIER_SYSTEM, temperature=0.0)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _latest_user_text(messages: List[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


# ---------------------------------------------------------------------
# Main writer node
# ---------------------------------------------------------------------
def writer_node(state: GraphState) -> GraphState:
    """
    Modes:
      1. Concept Q&A (no dataset facts) — LLM explains concept.
      2. Dataset summary — structured EDA/modeling guidance.
      3. Fallback — ask user for dataset or target.
    """
    messages = getattr(state, "messages", []) or []
    facts: Dict[str, Any] = getattr(state, "facts", {}) or {}
    tool_outs: List[Dict[str, Any]] = getattr(state, "tool_outputs", []) or []
    user_text = _latest_user_text(messages)
    has_any_facts = bool(facts) or bool(tool_outs)

    # -----------------------------------------------------------------
    # NEW: LLM-based concept vs data classification
    # -----------------------------------------------------------------
    verdict = "DATA"
    if user_text:
        try:
            res = _classifier_llm.invoke(user_text)
            verdict = (res.content or "").strip().upper()
        except Exception:
            verdict = "DATA"

    # -----------------------------------------------------------------
    # MODE 1 — Concept Q&A
    # -----------------------------------------------------------------
    if verdict.startswith("CONCEPT") and not has_any_facts:
        prompt = f"User question:\n{user_text}\n"
        msg = _concept_llm.invoke(prompt)
        answer = (msg.content or "").strip() or "Could you clarify what concept you want explained?"
        state.final_answer = answer
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # -----------------------------------------------------------------
    # MODE 2 — Dataset summary (your existing behavior)
    # -----------------------------------------------------------------
    if has_any_facts:
        facts_txt = "\n".join(f"- {k}: {v}" for k, v in facts.items()) if isinstance(facts, dict) else str(facts)
        tool_txt = "\n".join(str(t) for t in tool_outs)
        prompt = (
            "You are summarizing structured dataset insights.\n\n"
            f"Facts:\n{facts_txt}\n\nTool Outputs:\n{tool_txt}\n\n"
            "Provide: Dataset Overview, Recommended Algorithms, Preprocessing, "
            "Evaluation Strategy, and Next Steps."
        )
        msg = _writer.invoke(prompt)
        state.final_answer = (msg.content or "").strip()
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # -----------------------------------------------------------------
    # MODE 3 — Fallback (no facts + not conceptual)
    # -----------------------------------------------------------------
    fallback = (
        "I couldn’t compute dataset facts yet. Please attach a CSV/XLSX "
        "and specify your task (classification, regression, clustering, etc.) "
        "and target column if applicable."
    )
    state.final_answer = fallback
    if hasattr(state, "steps"):
        state.steps = int(getattr(state, "steps", 0) or 0) + 1
    return state
