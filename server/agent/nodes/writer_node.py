# server/agent/nodes/writer_node.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from ...llm import make_llm
from ..state import GraphState
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------
WRITER_SYSTEM = """
You are a senior data-science copilot writing concise, technical Markdown summaries.

Core principles:
- Derive all reasoning ONLY from the structured facts and the user's intent.
- Keep tone analytical but natural — avoid template repetition.
- When the user explicitly requests a task family (classification, clustering, etc.),
  lock your recommendations to that family only.
- Merge dataset facts and interpretations in one 'Dataset Overview' section.
- Each bullet: show the fact ➜ interpret what it implies for modeling.
- Always include: Recommended Algorithms, Preprocessing & Feature Handling,
  Evaluation Strategy, and Next Steps.
"""

_writer = make_llm(system_prompt=WRITER_SYSTEM, temperature=0.45)

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return "Not computed"

def _strong_corr_count(corr_pairs: Any) -> int:
    try:
        return sum(1 for p in (corr_pairs or [])
                   if isinstance(p, dict) and abs(float(p.get("pearson", 0))) >= 0.7)
    except Exception:
        return 0

def _len_or_zero(x: Optional[list]) -> int:
    return len(x or [])

def _detect_requested_family(state: GraphState) -> str:
    """Detect explicit or implicit user intent."""
    facts = state.facts or {}
    explicit = (facts.get("requested_family") or "").strip().lower()
    if explicit:
        return explicit.capitalize()

    # fall back to latest human message
    text = ""
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage):
            text = (msg.content or "").lower()
            break

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
    tgt = (facts.get("target") or {})
    cols = (facts.get("columns") or {})
    has_target = bool(tgt.get("name"))
    classy = isinstance(tgt.get("classes"), int) and tgt["classes"] > 1
    likely_class = classy or (tgt.get("positive_rate") is not None)
    has_dt = _len_or_zero(cols.get("datetime")) > 0
    many_feats = (_len_or_zero(cols.get("numeric")) +
                  _len_or_zero(cols.get("categorical")) +
                  _len_or_zero(cols.get("datetime"))) >= 20
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
    """Generate the final Markdown analysis using dataset facts + user intent."""
    if not state.facts:
        state.final_answer = (
            "I couldn’t compute dataset facts yet. Please attach a dataset "
            "and specify the target column or analysis intent."
        )
        return state

    facts = state.facts or {}
    ds, cols, tgt = facts.get("dataset", {}), facts.get("columns", {}), facts.get("target", {})
    miss, corr = facts.get("missing", {}), facts.get("correlation", {})

    # Determine task family
    requested_family = _detect_requested_family(state)
    auto_candidates = _task_candidates_from_facts(facts)
    intent_line = requested_family if requested_family else ", ".join(auto_candidates) or "Unclear"
    hard_rule = ""
    if requested_family:
        hard_rule = (
            f"Hard constraint: The user explicitly requested **{requested_family}**. "
            "Recommend algorithms ONLY for this family, even if a target exists."
        )

    # -----------------------------------------------------------------
    # PROMPT CONSTRUCTION
    # -----------------------------------------------------------------
    prompt = f"""
You will produce a structured technical Markdown answer.

Task family to consider: {intent_line}
{hard_rule}

## Dataset Overview
Provide one cohesive bullet list that merges dataset facts with interpretation.
Each bullet must both *state* the metric and *explain its implication* for modeling.
Example style:
- **Rows:** 5000 → medium-sized dataset; suitable for algorithms up to quadratic complexity.

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
- Strong correlation pairs (|r|≥0.7): {_strong_corr_count((corr or {}).get('pairs'))}

## Recommended Algorithms
List and rank 2–4 algorithms within ONLY the relevant family.  
Each bullet should include:
- short factual reason for suitability
- short note on when it may underperform (e.g., sensitive to scale, noise, imbalance).

If **Clustering**, consider options like k-means, k-prototypes, DBSCAN/HDBSCAN,
Agglomerative, or GMM, depending on data types and size.

## Preprocessing & Feature Handling
State only what applies to THIS family.
Mention scaling, encoding, missing-value treatment, dimensionality reduction, or
distance metrics as relevant.

## Evaluation Strategy
Describe proper validation for THIS family:
- Classification → stratified k-fold, macro-F1/ROC-AUC.
- Regression → k-fold, MAE/RMSE.
- Clustering → silhouette, Davies–Bouldin, Calinski–Harabasz, stability checks.
Include key hyperparameters to tune for each algorithm.

## Next Steps
List 2–4 concrete actions for improving model selection or performance.

Avoid templated phrasing; write as if you’re a senior data scientist summarizing findings.
"""

    msg = _writer.invoke(prompt)
    state.final_answer = msg.content.strip()
    state.messages.append(msg)
    state.steps += 1
    return state
