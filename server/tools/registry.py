# server/tools/registry.py
from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.toolsim import ToolWrap
from .stats_tools import (
    compute_dataset_facts,
    compute_correlation,
    plot_histogram,
    profile_for_model_selection,
)

# ---------- Arg Schemas ----------

class FactsArgs(BaseModel):
    file_id_or_name: str = Field(..., description="CSV id or filename under uploads/")

class CorrArgs(BaseModel):
    file_id_or_name: str = Field(..., description="CSV id or filename under uploads/")
    x: str = Field(..., description="Name of numeric column X")
    y: str = Field(..., description="Name of numeric column Y")
    method: str = Field(default="pearson", description="Correlation method (only 'pearson' supported)")

class HistArgs(BaseModel):
    file_id_or_name: str = Field(..., description="CSV id or filename under uploads/")
    column: str = Field(..., description="Numeric column to histogram")
    bins: int = Field(default=30, ge=1, description="Number of histogram bins")

class ProfileArgs(BaseModel):
    file_id_or_name: str = Field(..., description="CSV id or filename under uploads/")
    target: str = Field(..., description="Target label column for classification")
    imbalance_threshold: float = Field(0.60, ge=0.5, le=0.99, description="Majority ratio threshold to flag imbalance")
    high_cardinality_threshold: int = Field(20, ge=5, description="Nunique threshold for high-cardinality categoricals")
    small_n_threshold: int = Field(1000, ge=10, description="Row-count threshold to warn about small-n")

# ---------- Tool Registry (facts only; non-prescriptive) ----------

TOOLS = [
    ToolWrap(
        name="compute_dataset_facts",
        description="Compute dataset summary: n_rows, n_cols, dtype counts, and describe(include='all').",
        args_schema=FactsArgs,
        fn=lambda **kw: compute_dataset_facts(**kw),
        how_computed="pandas.DataFrame.describe(include='all') with NA handling",
    ),
    ToolWrap(
        name="compute_correlation",
        description="Compute Pearson correlation r and p-value between two numeric columns.",
        args_schema=CorrArgs,
        fn=lambda **kw: compute_correlation(**kw),
        how_computed="scipy.stats.pearsonr with pairwise-complete observations",
    ),
    ToolWrap(
        name="plot_histogram",
        description="Return histogram counts and edges for a numeric column (no image).",
        args_schema=HistArgs,
        fn=lambda **kw: plot_histogram(**kw),
        how_computed="numpy.histogram on numeric column (dropna)",
    ),
    ToolWrap(
        name="profile_for_model_selection",
        description="Profile dataset for model selection (facts only): size, dtypes, missingness, class balance, high-cardinality categoricals, suggested metrics & preprocessing. NO recommendations.",
        args_schema=ProfileArgs,
        fn=lambda **kw: profile_for_model_selection(**kw),
        how_computed="Heuristic profiling only; non-prescriptive.",
    ),
]
