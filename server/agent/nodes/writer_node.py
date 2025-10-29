# server/agent/nodes/writer_node.py
from __future__ import annotations
from typing import Any, Dict, List
from ...llm import make_llm
from ..state import GraphState

WRITER_SYSTEM = """
You are a senior data scientist.
Write a concise, technical answer in Markdown using ONLY the structured facts you are given.
Rules:
- Do NOT mention internal tool or function names.
- Do NOT invent numbers or fields. If a value is missing, say "Not computed".
- Prefer bullet points and short sentences. Include key metrics and what they imply.
- Recommendations must be justified strictly by the facts (e.g., imbalance ratio, feature types, correlations, missingness).
- Never output JSON. Output Markdown only.
"""

_writer = make_llm(system_prompt=WRITER_SYSTEM, temperature=0.1)

def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return "Not computed"

def _strong_corr_count(corr_pairs: Any) -> int:
    try:
        return sum(1 for p in (corr_pairs or []) if isinstance(p, dict) and abs(float(p.get("pearson", 0))) >= 0.7)
    except Exception:
        return 0

def _derive_reco(facts: Dict[str, Any]) -> List[str]:
    """Small deterministic addendum derived from facts only."""
    recos: List[str] = []
    tgt = (facts.get("target") or {})
    cols = (facts.get("columns") or {})
    corr = (facts.get("correlation") or {})
    miss = (facts.get("missing") or {})

    imb = tgt.get("imbalance_ratio")
    pos_rate = tgt.get("positive_rate")
    if isinstance(imb, (int, float)) and imb > 1.5:
        recos.append("Use class weighting or resampling to address class imbalance.")
    elif isinstance(pos_rate, (int, float)) and (pos_rate < 0.3 or pos_rate > 0.7):
        recos.append("Target is skewed; consider class weighting or resampling.")

    if cols.get("numeric"):
        recos.append("Tree ensembles (RandomForest, Gradient Boosting/XGBoost) work well for numeric features and non-linearities.")
    if cols.get("categorical"):
        recos.append("Ensure proper categorical encoding (one-hot/target encoding) before modeling.")
    if _strong_corr_count((corr or {}).get("pairs")) > 0:
        recos.append("High inter-feature correlation detected; prefer regularization or tree ensembles to handle multicollinearity.")

    total_pct = (miss or {}).get("total_pct")
    if isinstance(total_pct, (int, float)) and total_pct > 0.01:
        recos.append("Impute missing values (median for numeric, most-frequent for categorical) or use models robust to missingness.")

    recos.append("Include a simple baseline (Logistic Regression for classification) for interpretability and calibration.")
    return recos

def writer_node(state: GraphState) -> GraphState:
    """Synthesize the only user-facing answer from structured facts."""
    # If no facts, avoid printing "Not computed" walls.
    if not state.facts:
        state.final_answer = (
            "I couldn’t compute dataset facts yet. Please ensure a dataset is attached and the target column "
            "is specified (e.g., `Target: label`)."
        )
        return state

    facts = state.facts or {}
    dataset = facts.get("dataset") or {}
    cols = facts.get("columns") or {}
    tgt = facts.get("target") or {}
    miss = facts.get("missing") or {}
    corr = facts.get("correlation") or {}

    prompt = f"""
Summarize model-selection-relevant characteristics and produce recommendations using ONLY these facts:

Dataset
- file: {dataset.get('file_id') or "Not provided"}
- rows: {dataset.get('rows') if dataset.get('rows') is not None else "Not computed"}
- cols: {dataset.get('cols') if dataset.get('cols') is not None else "Not computed"}

Target
- name: {tgt.get('name') or "Not provided"}
- classes: {tgt.get('classes') if tgt.get('classes') is not None else "Not computed"}
- positive_rate: {_fmt_pct(tgt.get('positive_rate'))}
- imbalance_ratio: {tgt.get('imbalance_ratio') if tgt.get('imbalance_ratio') is not None else "Not computed"}

Features
- numeric: {len(cols.get('numeric') or [])}
- categorical: {len(cols.get('categorical') or [])}
- datetime: {len(cols.get('datetime') or [])}

Missingness
- total_pct: {_fmt_pct(miss.get('total_pct'))}
- per_column: {"available" if isinstance(miss.get('per_column'), dict) else "Not computed"}

Correlations
- strong_pairs(|r|≥0.7): {_strong_corr_count((corr or {}).get('pairs'))}

Output Markdown with sections:
- **Data Characteristics**
- **Recommended Algorithms** (2–4 items; each with one-line justification tied to the facts)
- **Limitations / Next Steps**
"""

    msg = _writer.invoke(prompt)
    derived = _derive_reco(facts)
    final = msg.content.rstrip()
    if derived:
        final += "\n\n**Additional recommendations (derived from facts):**\n- " + "\n- ".join(derived)

    state.final_answer = final
    state.messages.append(msg)
    state.steps += 1
    return state
