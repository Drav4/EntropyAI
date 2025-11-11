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
- Derive reasoning ONLY from structured facts and the current user intent.
- NEVER persist task families across turns; re-evaluate each query independently.
- Respect explicit family requests (classification, clustering, etc.) for this message only.
- If no explicit family is mentioned, infer from dataset facts (target type, datetime, etc.).
- Include: Dataset Overview, Recommended Algorithms, Preprocessing, Evaluation Strategy, Next Steps.
"""

CONCEPT_SYSTEM = """
You are a concise data-science mentor. For conceptual DS/ML questions (without dataset),
answer with:
1. One-line definition
2. Practical intuition
3. 3–5 common examples or variants
4. When to use vs. when not to use
Keep it under 12 sentences.
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
- CONCEPT = theoretical ML/DS questions (definitions, algorithm explanations, comparisons, formulas, etc.)
- DATA = dataset or experiment-based questions (EDA, model suggestions, correlations, etc.)
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
        if f > 1:
            return f"{f:.1f}%"
        return f"{f * 100:.1f}%"
    except Exception:
        return "Not computed"

# Explicit vs inferred task family (always recomputed per message)
def _detect_task_family(user_text: str, facts: Dict[str, Any]) -> (str, str):
    """Returns (explicit_family, inferred_family)"""
    text = user_text.lower()
    explicit = ""
    if any(k in text for k in ["cluster", "clustering", "unsupervised"]):
        explicit = "Clustering"
    elif any(k in text for k in ["classification", "classify"]):
        explicit = "Classification"
    elif any(k in text for k in ["regression", "regressor"]):
        explicit = "Regression"
    elif any(k in text for k in ["forecast", "time series", "timeseries"]):
        explicit = "Forecasting"
    elif any(k in text for k in ["anomaly", "outlier"]):
        explicit = "Anomaly Detection"

    # auto inference from facts
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
    inferred = []
    if not has_target:
        inferred.extend(["Clustering", "Anomaly Detection"])
    else:
        inferred.append("Classification" if likely_class else "Regression")
    if has_dt:
        inferred.append("Forecasting")
    if many_feats:
        inferred.append("Dimensionality Reduction")
    # dedup
    seen = set()
    inferred = [f for f in inferred if not (f in seen or seen.add(f))]
    return explicit, ", ".join(inferred) if inferred else "Unclear"

# ---------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------
def writer_node(state: GraphState) -> GraphState:
    """
    Final writer stage.
      1. Conceptual Q&A → LLM explanation (no dataset)
      2. Dataset-driven summary → structured guidance
      3. Fallback → request dataset/intent
    """
    messages = getattr(state, "messages", []) or []
    facts: Dict[str, Any] = getattr(state, "facts", {}) or {}
    tool_outs: List[Dict[str, Any]] = getattr(state, "tool_outputs", []) or []
    user_text = _latest_user_text(messages)
    has_any_facts = bool(facts) or bool(tool_outs)

    # Step 1: classify query intent (Concept vs Data)
    verdict = "DATA"
    if user_text:
        try:
            res = _classifier_llm.invoke(user_text)
            verdict = (res.content or "").strip().upper()
        except Exception:
            verdict = "DATA"

    # -----------------------------------------------------------------
    # MODE 1 — Conceptual Q&A
    # -----------------------------------------------------------------
    if verdict.startswith("CONCEPT") and not has_any_facts:
        msg = _concept_llm.invoke(f"User question:\n{user_text}")
        answer = (msg.content or "").strip() or "Could you clarify what concept you want explained?"
        state.final_answer = answer
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # -----------------------------------------------------------------
    # MODE 2 — Dataset-driven summary
    # -----------------------------------------------------------------
    if facts:
        ds = facts.get("dataset", {})
        cols = facts.get("columns", {})
        tgt = facts.get("target", {})
        miss = facts.get("missing", {})
        corr = facts.get("correlation", {})

        explicit_family, inferred_family = _detect_task_family(user_text, facts)
        active_family = explicit_family or inferred_family
        hard_rule = ""
        if explicit_family:
            hard_rule = (
                f"Hard constraint: The user explicitly requested **{explicit_family}**. "
                "Recommend algorithms ONLY for this family, even if a target exists."
            )

        prompt = f"""
You will produce a structured technical Markdown answer.

Task family to consider: {active_family}
{hard_rule}

## Dataset Overview
Provide concise bullets that merge metrics with interpretation.

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
List 2–4 algorithms ONLY within the relevant family.
If clustering: k-means, DBSCAN/HDBSCAN, Agglomerative, GMM, etc.
If classification: logistic regression, random forest, XGBoost, LightGBM, etc.
If regression: linear, ridge/lasso, XGBoost regressor, etc.
Justify briefly when each performs best or worst.

## Preprocessing & Feature Handling
Mention what applies for THIS family only (scaling, encoding, imputation, etc.).

## Evaluation Strategy
Describe validation per family (classification → stratified k-fold; clustering → silhouette, etc.)

## Next Steps
List 2–4 concrete actions for improving data quality or model selection.
"""

        msg = _writer.invoke(prompt)
        state.final_answer = (msg.content or "").strip()
        state.messages.append(msg)
        if hasattr(state, "steps"):
            state.steps = int(getattr(state, "steps", 0) or 0) + 1
        return state

    # -----------------------------------------------------------------
    # MODE 3 — Fallback
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
