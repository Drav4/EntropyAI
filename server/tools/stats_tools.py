# server/tools/stats_tools.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from server.config import UPLOAD_DIR

# -------- path & loading helpers --------

def _resolve_path(file_id_or_name: str) -> Tuple[Optional[str], Optional[str]]:
    file_id_or_name = (file_id_or_name or "").strip()
    if not file_id_or_name:
        return None, "file_id_or_name is empty"

    cand = os.path.join(UPLOAD_DIR, file_id_or_name)
    if os.path.isfile(cand):
        return cand, None

    base = os.path.basename(file_id_or_name)
    try:
        for fn in os.listdir(UPLOAD_DIR):
            if fn == base or base in fn:
                return os.path.join(UPLOAD_DIR, fn), None
    except Exception as e:
        return None, f"Failed to list {UPLOAD_DIR}: {e}"

    return None, f"File not found in uploads: {file_id_or_name}"

def _load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    # default try CSV
    return pd.read_csv(path)

def _classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    boolean = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    datetime = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in df.columns if (df[c].dtype == "object") or pd.api.types.is_categorical_dtype(df[c])]
    # remove overlaps
    categorical = [c for c in categorical if c not in numeric and c not in boolean and c not in datetime]
    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime,
    }

def _missing_report(df: pd.DataFrame) -> Dict[str, Any]:
    if df.size == 0:
        return {"total_pct": 0.0, "per_column": {}}
    per_col = {c: float(df[c].isna().mean()) for c in df.columns}
    total_pct = float(df.isna().mean().mean())
    return {"total_pct": _finite_or_none(total_pct), "per_column": {k: _finite_or_none(v) for k, v in per_col.items()}}

def _summary_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_column": {}}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.size == 0:
            out["per_column"][c] = {"mean": None, "std": None, "min": None, "max": None, "p50": None}
            continue
        out["per_column"][c] = {
            "mean": _finite_or_none(np.nanmean(s)),
            "std": _finite_or_none(np.nanstd(s)),
            "min": _finite_or_none(np.nanmin(s)),
            "max": _finite_or_none(np.nanmax(s)),
            "p50": _finite_or_none(np.nanpercentile(s, 50)),
        }
    return out

def _correlation_pairs(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    if len(cols) < 2:
        return pairs
    corr = df[cols].corr(method="pearson")
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            val = corr.at[a, b]
            if pd.isna(val):
                continue
            pairs.append({"x": a, "y": b, "pearson": _finite_or_none(float(val))})
    # strongest first
    pairs = [p for p in pairs if p["pearson"] is not None]
    pairs.sort(key=lambda p: abs(p["pearson"]), reverse=True)
    return pairs

def _target_stats(df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
    if not target or target not in df.columns:
        return {"name": target, "classes": None, "class_counts": None, "positive_rate": None, "imbalance_ratio": None}
    y = df[target]
    vc = y.value_counts(dropna=True)
    class_counts = {str(k): int(v) for k, v in vc.items()}
    classes = list(vc.index)
    imb = None
    if len(vc) >= 2:
        maj = int(vc.max()); minc = int(vc.min())
        imb = _finite_or_none(float(maj / max(minc, 1)))
    # positive rate heuristic
    try:
        if set(vc.index) <= {0, 1}:
            pos_rate = _finite_or_none(float((y == 1).mean()))
        else:
            pos_rate = _finite_or_none(float(vc.min() / vc.sum())) if vc.sum() > 0 else None
    except Exception:
        pos_rate = None
    return {
        "name": target,
        "classes": classes,
        "class_counts": class_counts,
        "positive_rate": pos_rate,
        "imbalance_ratio": imb,
    }

def _finite_or_none(x: Any) -> Optional[float]:
    try:
        xf = float(x)
        return None if (np.isnan(xf) or np.isinf(xf)) else xf
    except Exception:
        return None

# --------- public tools (function style) ---------

def compute_dataset_facts(*, file_id_or_name: str) -> Dict[str, Any]:
    path, err = _resolve_path(file_id_or_name)
    if err:
        return {"ok": False, "error": err, "arguments": {"file_id_or_name": file_id_or_name}}
    try:
        df = _load_df(path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load file: {e}", "arguments": {"file_id_or_name": file_id_or_name}}

    cols = _classify_columns(df)
    miss = _missing_report(df)
    summ = _summary_stats(df, cols["numeric"])
    return {
        "ok": True,
        "dataset": {"file_id": os.path.basename(path), "rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": cols,
        "missing": miss,
        "summary": summ,
        "notes": [],
        "limitations": [],
    }

def compute_correlation(*, file_id_or_name: str, x: str, y: str, method: str = "pearson") -> Dict[str, Any]:
    if method.lower() != "pearson":
        return {"ok": False, "error": f"Unsupported method '{method}'. Only 'pearson' supported.",
                "arguments": {"file_id_or_name": file_id_or_name, "x": x, "y": y, "method": method}}
    path, err = _resolve_path(file_id_or_name)
    if err:
        return {"ok": False, "error": err, "arguments": {"file_id_or_name": file_id_or_name, "x": x, "y": y, "method": method}}
    try:
        df = _load_df(path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load file: {e}",
                "arguments": {"file_id_or_name": file_id_or_name, "x": x, "y": y, "method": method}}

    if x not in df.columns or y not in df.columns:
        return {"ok": False, "error": f"Columns '{x}' or '{y}' not found",
                "arguments": {"file_id_or_name": file_id_or_name, "x": x, "y": y, "method": method}}

    sx = pd.to_numeric(df[x], errors="coerce")
    sy = pd.to_numeric(df[y], errors="coerce")
    mask = sx.notna() & sy.notna()
    if mask.sum() < 2:
        return {"ok": False, "error": "Not enough valid pairs for correlation.",
                "arguments": {"file_id_or_name": file_id_or_name, "x": x, "y": y, "method": method}}

    r = float(np.corrcoef(sx[mask], sy[mask])[0, 1])
    # p-value via t-stat approximation (no SciPy dependency)
    n = int(mask.sum())
    try:
        t = abs(r) * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
        # two-tailed p approx using survival function of Student's t; fallback to None if not available
        from mpmath import quad, power  # optional; if missing, skip
        # crude approx: not shipping full t-CDF; set None if mpmath not present
        p_value = None
    except Exception:
        p_value = None

    return {
        "ok": True,
        "dataset": {"file_id": os.path.basename(path)},
        "correlation": {"x": x, "y": y, "pearson": _finite_or_none(r), "n": n, "p_value": p_value},
        "notes": [],
        "limitations": [],
    }

def plot_histogram(*, file_id_or_name: str, column: str, bins: int = 30) -> Dict[str, Any]:
    path, err = _resolve_path(file_id_or_name)
    if err:
        return {"ok": False, "error": err, "arguments": {"file_id_or_name": file_id_or_name, "column": column, "bins": bins}}
    try:
        df = _load_df(path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load file: {e}",
                "arguments": {"file_id_or_name": file_id_or_name, "column": column, "bins": bins}}

    if column not in df.columns:
        return {"ok": False, "error": f"Column '{column}' not found",
                "arguments": {"file_id_or_name": file_id_or_name, "column": column, "bins": bins}}

    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return {"ok": False, "error": f"Column '{column}' has no numeric data",
                "arguments": {"file_id_or_name": file_id_or_name, "column": column, "bins": bins}}

    counts, edges = np.histogram(s.values, bins=bins)
    return {
        "ok": True,
        "dataset": {"file_id": os.path.basename(path)},
        "histogram": {"column": column, "bins": int(bins), "counts": counts.tolist(), "edges": [float(e) for e in edges.tolist()]},
        "notes": [],
        "limitations": [],
    }

def profile_for_model_selection(*,
                                file_id_or_name: str,
                                target: str,
                                imbalance_threshold: float = 0.60,
                                high_cardinality_threshold: int = 20,
                                small_n_threshold: int = 1000) -> Dict[str, Any]:
    path, err = _resolve_path(file_id_or_name)
    if err:
        return {"ok": False, "error": err, "arguments": locals()}
    try:
        df = _load_df(path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load file: {e}", "arguments": locals()}

    cols = _classify_columns(df)
    miss = _missing_report(df)
    summ = _summary_stats(df, cols["numeric"])
    tgt = _target_stats(df, target)
    corr_pairs = _correlation_pairs(df, cols["numeric"])
    # heuristics (facts only)
    warnings: List[str] = []
    if isinstance(tgt.get("imbalance_ratio"), (int, float)) and tgt["imbalance_ratio"] and tgt["imbalance_ratio"] > (1.0 / max(1e-6, 1 - imbalance_threshold)):
        warnings.append("Severe class imbalance detected.")
    # high-cardinality categoricals
    for c in cols["categorical"]:
        nunique = int(df[c].nunique(dropna=True))
        if nunique >= high_cardinality_threshold:
            warnings.append(f"High-cardinality categorical: '{c}' ({nunique} unique)")
    if df.shape[0] < small_n_threshold:
        warnings.append(f"Small-n warning: only {df.shape[0]} rows")

    return {
        "ok": True,
        "dataset": {"file_id": os.path.basename(path), "rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": cols,
        "missing": miss,
        "summary": summ,
        "target": tgt,
        "correlation": {"pairs": corr_pairs},
        "notes": [],
        "limitations": warnings,
    }
