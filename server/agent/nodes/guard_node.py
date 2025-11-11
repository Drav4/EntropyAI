from __future__ import annotations
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from ..state import GraphState
from ...llm import make_llm

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
REFUSAL = (
    "I can only help with data-science / ML tasks "
    "(EDA, modeling, evaluation, theory, algorithms, etc.). "
    "Please rephrase your request in that scope."
)

# ---------------------------------------------------------------------
# LLM-based router setup
# ---------------------------------------------------------------------
_GUARD_SYSTEM = """
You are a strict classifier for routing user requests to a data-science assistant.

Return ONLY one token: DS or OOS.

Rules:
- DS (data-science) includes: data analysis, statistics, machine learning, AI, deep learning,
  neural networks, algorithms, theory (e.g., gradient descent, backpropagation, bias-variance),
  data visualization, EDA, pipelines, model evaluation, MLOps, Python for data/ML, etc.
- OOS (out of scope) includes: personal questions, device issues, AppleCare, software installation,
  image editing, general tech support, entertainment, travel, finance unrelated to data science,
  coding not related to ML/data analysis, or anything non-technical.
- Be generous: if the question is even slightly about data/ML, output DS.
- Return exactly one token — DS or OOS — nothing else.
"""

_guard_llm = make_llm(system_prompt=_GUARD_SYSTEM, temperature=0.0)


# ---------------------------------------------------------------------
# Helper to extract last human message
# ---------------------------------------------------------------------
def _latest_user_text(messages: List) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


# ---------------------------------------------------------------------
# Guard Node
# ---------------------------------------------------------------------
def guard_node(state: GraphState) -> GraphState:
    """LLM-based router that decides whether to allow or block the message."""
    text = _latest_user_text(state.messages)
    if not text:
        return state  # Nothing to classify; let it pass

    # Ask the small LLM router
    verdict_msg = _guard_llm.invoke(text)
    verdict = (verdict_msg.content or "").strip().upper()

    # Accept if model says DS (or can't decide)
    if verdict.startswith("DS"):
        return state

    # Out-of-scope → refuse gracefully
    refusal = AIMessage(content=REFUSAL)
    state.messages.append(refusal)
    if hasattr(state, "final_answer"):
        state.final_answer = REFUSAL
    return state
