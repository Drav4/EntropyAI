import json, os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import HTTPException
from ..config import UPLOAD_DIR, PLOTS_DIR
from ..utils.paths import sha256_file, cache_summary_path

# ---------- Loaders ----------
def load_df_from_uploads(file_id_or_name: str) -> Tuple[str, pd.DataFrame, str]:
    direct = os.path.join(UPLOAD_DIR, file_id_or_name)
    if os.path.exists(direct):
        path = direct
    else:
        base = os.path.splitext(file_id_or_name)[0]
        path = None
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(base):
                path = os.path.join(UPLOAD_DIR, fname)
                break
        if not path:
            raise FileNotFoundError(file_id_or_name)

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type (use .csv or .xlsx)")
    return path, df, sha256_file(path)

# ---------- Summary ----------
def summarize_dataframe(df: pd.DataFrame, max_cats: int = 12) -> Dict[str, Any]:
    rows, _ = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum().to_dict()
    uniques = {}
    for c in df.columns:
        try: uniques[c] = int(df[c].nunique(dropna=True))
        except: uniques[c] = None

    top_values = {}
    for c in df.columns:
        try:
            if (df[c].dtype == "object") or (uniques.get(c, rows+1) <= min(50, max(rows // 20, 10))):
                vc = df[c].astype(str).value_counts(dropna=False).head(max_cats).to_dict()
                top_values[c] = {str(k): int(v) for k, v in vc.items()}
        except: pass

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_stats = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty: continue
        numeric_stats[c] = {
            "count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "p25": float(s.quantile(0.25)), "p50": float(s.median()),
            "p75": float(s.quantile(0.75)), "max": float(s.max()),
        }

    return {
        "rows": rows,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing": missing,
        "uniques": uniques,
        "numeric_stats": numeric_stats,
        "top_values": top_values,
        "sample_rows": df.head(5).to_dict(orient="records"),
    }

def get_or_build_summary(path: str, df: pd.DataFrame, file_hash: str) -> Dict[str, Any]:
    cp = cache_summary_path(file_hash)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    summary = summarize_dataframe(df)
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary

# ---------- Facts (descriptives + indicators) ----------
def compute_dataset_facts(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_like_cols = [c for c in df.columns
                     if (not pd.api.types.is_numeric_dtype(df[c])) or df[c].nunique(dropna=True) <= 20]

    miss_counts = df.isna().sum()
    miss_pct = (miss_counts / float(n_rows or 1))

    # target guess
    name_map = {c.lower(): c for c in df.columns}
    target_col: Optional[str] = None
    for p in ["target", "label", "class", "y"]:
        if p in name_map: target_col = name_map[p]; break
    if target_col is None and df.columns.any():
        last = df.columns[-1]
        uniq = int(df[last].nunique(dropna=True))
        if uniq <= 50 or not pd.api.types.is_numeric_dtype(df[last]):
            target_col = last

    target_type = None
    class_balance = None
    if target_col:
        if (not pd.api.types.is_numeric_dtype(df[target_col])) or df[target_col].nunique(dropna=True) <= 20:
            target_type = "categorical"
            vc = df[target_col].astype(str).value_counts(dropna=False)
            class_balance = {str(k): int(v) for k, v in vc.items()}
        else:
            target_type = "numeric"

    # Descriptive stats
    desc = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty: continue
        q1, q2, q3 = s.quantile(0.25), s.median(), s.quantile(0.75)
        iqr = q3 - q1
        out_lo, out_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_rate = float(((s < out_lo) | (s > out_hi)).mean())
        desc[c] = {
            "count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "q1": float(q1), "median": float(q2), "q3": float(q3),
            "max": float(s.max()), "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
            "outlier_rate_iqr": out_rate,
        }

    # Mean abs inter-feature corr
    mean_abs_feature_corr = None
    if len(numeric_cols) >= 2:
        cm = df[numeric_cols].corr(numeric_only=True).abs()
        if cm.shape[0] > 1:
            upper = np.triu(np.ones_like(cm, dtype=bool), k=1)
            vals = cm.where(upper).stack().values
            if len(vals) > 0:
                mean_abs_feature_corr = float(np.mean(vals))

    # Top corr with binary target
    corr_highlights = None
    if target_col and target_type == "categorical" and df[target_col].nunique(dropna=True) == 2:
        classes = sorted(df[target_col].dropna().unique(), key=lambda x: str(x))
        mapping = {classes[0]: 0, classes[1]: 1}
        y = df[target_col].map(mapping)
        num_corrs = {}
        for c in numeric_cols:
            try:
                r = y.corr(df[c], method="pearson")
                if pd.notna(r): num_corrs[c] = float(round(r, 6))
            except: pass
        if num_corrs:
            top = sorted(num_corrs.items(), key=lambda kv: -abs(kv[1]))[:5]
            corr_highlights = [{"feature": k, "corr_with_target": v} for k, v in top]

    return {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "n_numeric": int(len(numeric_cols)),
        "n_categorical_like": int(len(cat_like_cols)),
        "numeric_cols_sample": numeric_cols[:10],
        "categorical_like_sample": cat_like_cols[:10],
        "missing": {c: {"count": int(miss_counts[c]), "pct": float(round(miss_pct[c], 6))} for c in df.columns},
        "high_missing_cols_over_5pct": sorted(
            [c for c in df.columns if (miss_pct[c] >= 0.05)],
            key=lambda x: -miss_pct[x]
        )[:10],
        "target_guess": target_col,
        "target_type": target_type,
        "class_balance": class_balance,
        "descriptive_stats": desc,
        "mean_abs_feature_corr": mean_abs_feature_corr,
        "corr_highlights": corr_highlights,
    }

# ---------- Tool wrappers ----------
def tool_get_dataset_summary(file_id_or_name: str) -> Dict[str, Any]:
    path, df, file_hash = load_df_from_uploads(file_id_or_name)
    summary = get_or_build_summary(path, df, file_hash)
    return {"file_id": os.path.basename(path), "file_hash": file_hash, "summary": summary}

def tool_compute_dataset_facts(file_id_or_name: str) -> Dict[str, Any]:
    path, df, file_hash = load_df_from_uploads(file_id_or_name)
    facts = compute_dataset_facts(df)
    return {"file_id": os.path.basename(path), "file_hash": file_hash, "facts": facts}

def tool_run_pandas_op(file_id_or_name: str, op: str, column: Optional[str] = None, top_n: int = 10) -> Dict[str, Any]:
    path, df, file_hash = load_df_from_uploads(file_id_or_name)
    op = op.lower().strip()
    if op == "head":
        result = df.head(min(10, top_n)).to_dict(orient="records")
    elif op == "describe":
        result = df.describe(include="all").fillna("").to_dict()
    elif op == "value_counts":
        if not column or column not in df.columns:
            raise HTTPException(status_code=400, detail="value_counts requires 'column'")
        vc = df[column].astype(str).value_counts(dropna=False).head(top_n).to_dict()
        result = {str(k): int(v) for k, v in vc.items()}
    elif op == "corr":
        corr = df.select_dtypes(include="number").corr(numeric_only=True)
        result = corr.fillna(0).round(6).to_dict()
    elif op == "missing_report":
        miss = df.isna().sum().to_dict()
        total = len(df)
        result = {k: {"count": int(v), "pct": (float(v) / float(total or 1))} for k, v in miss.items()}
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported op '{op}'")
    return {"file_id": os.path.basename(path), "file_hash": file_hash, "op": op, "column": column, "result": result}

def tool_plot_histogram(file_id_or_name: str, column: str, bins: int = 30) -> Dict[str, Any]:
    path, df, file_hash = load_df_from_uploads(file_id_or_name)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise HTTPException(status_code=400, detail=f"Column '{column}' must be numeric for histogram")

    plt.clf()
    df[column].dropna().plot(kind="hist", bins=bins)
    plt.xlabel(column); plt.ylabel("count")
    out_name = f"{os.path.basename(path)}.{column}.{file_hash[:8]}.hist.png".replace(os.sep, "_")
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    return {"file_id": os.path.basename(path), "file_hash": file_hash, "plot_url": f"/files/plots/{out_name}"}
