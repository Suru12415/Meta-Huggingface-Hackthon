from __future__ import annotations
import numpy as np
import pandas as pd

def score(
    merged_df: pd.DataFrame | None,
    ground_truth_merged: pd.DataFrame | None,
) -> float:
    if merged_df is None or ground_truth_merged is None:
        return 0.0001
    if len(merged_df) == 0:
        return 0.0001
    numeric_truth_cols = ground_truth_merged.select_dtypes(include=[np.number]).columns
    numeric_agent_cols = merged_df.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_truth_cols if c in numeric_agent_cols]
    if not common_cols:
        return 0.0001
    min_len = min(len(merged_df), len(ground_truth_merged))
    col_scores: list[float] = []
    for col in common_cols:
        a = pd.to_numeric(merged_df[col].iloc[:min_len], errors="coerce").fillna(0).values
        r = pd.to_numeric(ground_truth_merged[col].iloc[:min_len], errors="coerce").fillna(0).values
        if r.std() < 1e-9 and a.std() < 1e-9:
            col_scores.append(1.0)
            continue
        if r.std() < 1e-9 or a.std() < 1e-9:
            col_scores.append(0.0)
            continue
        corr = float(np.corrcoef(a, r)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        col_scores.append((corr + 1.0) / 2.0)
    mean_col_score = float(np.mean(col_scores)) if col_scores else 0.0
    shape_factor = min(1.0, len(merged_df) / max(1, len(ground_truth_merged)))
    raw = round(mean_col_score * shape_factor, 4)
    return float(max(0.0001, min(0.9999, raw)))