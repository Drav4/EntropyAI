# server/agent/nodes/guard_node.py
from __future__ import annotations
import re
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from ..state import GraphState

REFUSAL = (
    "I can only help with data-science / ML tasks (EDA, feature engineering, modeling, "
    "evaluation, MLOps, plots, etc.). Please rephrase your request in that scope."
)

_DS_POS = re.compile(
    r"\b(eda|dataset|data\s*frame|feature\s*engineering|imput|scaler|normalize|"
    r"classif(y|ication)|regress(ion|or)|cluster(ing)?|forecast(ing)?|time\s*series|"
    r"cross[- ]?valid|roc|auc|f1|rmse|mae|silhouette|dbscan|k[- ]?means|hdbscan|"
    r"model(ing)?|train(ing)?|hyperparam|pipeline|mlflow|sklearn|xgboost|lightgbm|"
    r"bert|transformer|token(iz|is)er|embedding|pca|tsne|umap|confusion\s*matrix|"
    r"feature\s*importance|grid\s*search|random\s*search)\b",
    re.I,
)
_DS_NEG = re.compile(
    r"\b(wallpaper|photo\s*edit|image\s*edit|device\s*repair|applecare|subscription|"
    r"travel|politic|relationship|poe\s*2|game\s*engine|zbrush|houdini|unreal|"
    r"monitor\s*settings|bluedart|matrimony|bank|loan|billing|tax|shopping)\b",
    re.I,
)

def _latest_user_text(messages: List) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""

def guard_node(state: GraphState) -> GraphState:
    text = _latest_user_text(state.messages)
    is_ds = bool(_DS_POS.search(text))
    is_oos_hint = bool(_DS_NEG.search(text))

    # If OOS, write the refusal message and (optionally) set final_answer.
    if text and (not is_ds or is_oos_hint):
        msg = AIMessage(content=REFUSAL)
        state.messages.append(msg)
        # Only set if GraphState actually has this field
        if hasattr(state, "final_answer"):
            state.final_answer = REFUSAL
    # Otherwise do nothing; flow will continue to agent.
    return state
