from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from ...llm import make_llm
from ..state import GraphState

# ---------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------
WRITER_SYSTEM = """
You are a senior data-science copilot writing concise, technical Markdown summaries.

Guidelines:
- Derive reasoning ONLY from structured facts and the user's intent.
- Keep tone analytical and practical.
- Respect explicit family requests (classification, clustering, regression, etc.).
- If explicit family exists, NEVER override it with automatic inference.
- Otherwise, infer likely families from dataset facts (target type, datetime columns, etc.).
- Include: Dataset Overview, Recommended Algorithms, Preprocessing, Evaluation Strategy, Next Steps.
"""

CONCEPT_SYSTEM = """
You are a concise data-science mentor. For conceptual DS/ML questions (without dataset),
answer with:
1. One-line definition
2. Practical intuition
3. 3–5 common examples or variants
4. When to use vs. when not to use
Keep it under 12 sentences and avoid filler.
"""

# ---------------------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------------------
_writer = make_llm(system_prompt=WRITER_SYSTEM, temperature=0.45)
_concept_llm = make_llm(system_prompt=CONCEPT_SYSTEM, temperature=0.2)

_CLASSIFIER_SYSTEM = """
You are a strict router for a data-science assistant.

Return EXACTLY one token: CONCEPT or DATA.

Rules:
- CONCEPT = the user asks about DS/ML theory, algorithm definitions, comparisons,
  intuition, formulas, or conceptual best practices (e.g., "What is gradient descent?").
- DATA = the user requests dataset analysis, model recommendation, EDA, etc.
- Return only CONCEPT or DATA, no punctuation.
"""

_classifier_llm = make_llm(system_prompt=_CLASSIFIER_SYSTEM, temperature=0.0)

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def _latest_user_text(messages: List[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""

def _len_or_zero(x: Optional[list]) -> int:
    return len(x or [])

def _fmt_pct(v: Any) -> str:
    try:
        f = float(v)
        if f > 1:  # already in %
            return f"{f:.1f}%"
        return f"{f * 100:.1f}%"
    except Exception:
        return "Not computed"

# ---- restored from your original family heuristics ----
def _detect_requested_family(state: GraphState) -> str:
    facts = getattr(state, "facts", {}) or {}
    explicit = (facts.get("requested_family") or "").strip().lower()
    if explicit:
        return explicit.capitalize()

    text = _latest_user_text(getattr(state, "messages", [])).lower()
    if any(k in text for k in ["cluster", "clustering", "unsupervised"]):
        return "Clustering"
    if any(k in text for k in ["regression", "regressor"]):
        return "Regression"
    if any(k in text for k in ["forecast", "time series", "timeseries"]):
        return "Forecasting"
    if any(k in text for k in ["anomaly", "outlier"]):
        return "Anomaly Detection"
    return ""

def _task_candidates_from_facts(facts: Dict[str, Any]) -> List[str]:
    tgt = facts.get("target") or {}
    cols = facts.get("columns") or {}
    has_target = bool(tgt.get("name"))
    classy = isinstance(tgt.get("classes"), int) and tgt["classes"] > 1
    likely_class = classy or (tgt.get("positive_rate") is not None)
    has_dt = _len_or_zero(cols.get("datetime")) > 0
    many_feats = (
        _len_or_zero(cols.get("numeric"))
        + _len_or_zero(cols.get("categorical"))
        + _len_or_zero(cols.get("datetime"))
    ) >= 20
    cands: List[str] = []
    if not has_target:
        cands.extend(["Clustering", "Anomaly Detection"])
    else:
        cands.append("Classification" if likely_class else "Regression")
    if has_dt:
        cands.append("Forecasting")
    if many_feats:
        cands.append("Dimensionality Reduction")
    seen = set()
    return [c for c in cands if not (c in seen or seen.add(c))]

# ---------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------
def writer_node(state: GraphState) -> GraphState:
    """
    Final writer stage.
      1. Conceptual Q&A → LLM explanation (no dataset)
      2. Dataset-driven summary → structured technical guidance
      3. Fallback → ask user to provide dataset or intent
    """
    messages = getattr(state, "messages", []) or []
    facts: Dict[str, Any] = getattr(state, "facts", {}) or {}
    tool_outs: List[Dict[str, Any]] = getattr(state, "tool_outputs", []) or []
    user_text = _latest_user_text(messages)
    has_any_facts = bool(facts) or bool(tool_outs)

    # ---------- Step 1: Concept-vs-Data classification (LLM) ----------
    verdict = "DATA"
    if user_text:
        try:
            res = _classifier_llm.invoke(user_text)
            verdict = (res.content or "").strip().upper()
        except Exception:
            verdict = "DATA"

    # ---------- MODE 1: Conceptual Q&A ----------
    if verdict.startswith("CONCEPT") and not has_any_facts:
        prompt = f"User question:\n{user_text}\n"
        msg = _concept_llm.invoke(prompt)
        answer = (msg.content or "").strip() or "Could you clarify what concept you want explained?"
        state.final_answer = answer
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # ---------- MODE 2: Dataset-driven summary ----------
    if facts:
        ds = facts.get("dataset", {})
        cols = facts.get("columns", {})
        tgt = facts.get("target", {})
        miss = facts.get("missing", {})
        corr = facts.get("correlation", {})

        requested_family = _detect_requested_family(state)
        auto_candidates = _task_candidates_from_facts(facts)
        intent_line = (
            requested_family
            if requested_family
            else ", ".join(auto_candidates) or "Unclear"
        )

        hard_rule = ""
        if requested_family:
            hard_rule = (
                f"Hard constraint: The user explicitly requested **{requested_family}**. "
                "Recommend algorithms ONLY for this family, even if a target exists."
            )

        # Construct the dataset-based prompt (your old structure)
        prompt = f"""
You will produce a structured technical Markdown answer.

Task family to consider: {intent_line}
{hard_rule}

## Dataset Overview
Provide cohesive bullet points combining metrics with interpretation.

Facts:
- Rows: {ds.get('rows','Not computed')}
- Columns: {ds.get('cols','Not computed')}
- Target: {tgt.get('name') or 'None'}
- Classes: {tgt.get('classes') if tgt.get('classes') is not None else 'Not computed'}
- Positive rate: {_fmt_pct(tgt.get('positive_rate'))}
- Imbalance ratio: {tgt.get('imbalance_ratio') if tgt.get('imbalance_ratio') is not None else 'Not computed'}
- Numeric features: {_len_or_zero(cols.get('numeric'))}
- Categorical features: {_len_or_zero(cols.get('categorical'))}
- Datetime features: {_len_or_zero(cols.get('datetime'))}
- Missing values (%): {_fmt_pct(miss.get('total_pct'))}
- Strong correlation pairs (|r|≥0.7): {len((corr or {}).get('pairs') or [])}

## Recommended Algorithms
List 2–4 algorithms within ONLY the relevant family. For clustering, consider
k-means, DBSCAN/HDBSCAN, Agglomerative, or GMM; for classification, pick
appropriate supervised models. Justify each briefly.

## Preprocessing & Feature Handling
Mention only what applies to THIS family (scaling, encoding, missing-value treatment, etc.).

## Evaluation Strategy
Describe proper validation:
- Classification → stratified k-fold, macro-F1/ROC-AUC
- Regression → k-fold, MAE/RMSE
- Clustering → silhouette, Davies–Bouldin
- Forecasting → rolling-window or walk-forward validation

## Next Steps
List 2–4 concrete actions to improve data quality or model selection.
"""

        msg = _writer.invoke(prompt)
        state.final_answer = (msg.content or "").strip()
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # ---------- MODE 3: Fallback ----------
    fallback = (
        "I couldn’t compute dataset facts yet. Please attach a CSV/XLSX "
        "and specify your task (classification, regression, clustering, etc.) "
        "and target column if applicable."
    )
    state.final_answer = fallback
    if hasattr(state, "steps"):
        state.steps = int(getattr(state, "steps", 0) or 0) + 1
    return state
