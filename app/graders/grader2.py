from __future__ import annotations
import numpy as np
import pandas as pd

def outlier_pr(flagged: list[int], known: list[int]) -> tuple[float, float]:
    if not known:
        return (1.0, 1.0)
    if not flagged:
        return (0.0, 0.0)
    flagged_set = set(flagged)
    known_set = set(known)
    tp = len(flagged_set & known_set)
    precision = tp / len(flagged_set) if flagged_set else 0.0
    recall = tp / len(known_set) if known_set else 0.0
    return (precision, recall)

def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

NUMERIC_COLS = ["tensile_strength_mpa", "yield_point_mpa", "elongation_pct", "length_mm"]

def mae_ratio(agent_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    maes = []
    ref_len = min(len(agent_df), len(ground_truth_df))
    for col in NUMERIC_COLS:
        if col not in agent_df.columns or col not in ground_truth_df.columns:
            continue
        a = pd.to_numeric(agent_df[col].iloc[:ref_len], errors="coerce")
        r = pd.to_numeric(ground_truth_df[col].iloc[:ref_len], errors="coerce")
        mask = ~(a.isna() | r.isna())
        if mask.sum() == 0:
            continue
        mae = np.abs(a[mask].values - r[mask].values).mean()
        scale = np.abs(r[mask].values).mean()
        maes.append(mae / scale if scale > 1e-9 else 1.0)
    return float(np.mean(maes)) if maes else 1.0

def score(
    agent_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    flagged_outlier_ids: list[int],
    known_outlier_ids: list[int],
) -> float:
    precision, recall = outlier_pr(flagged_outlier_ids, known_outlier_ids)
    f1 = _f1(precision, recall)
    mae = mae_ratio(agent_df, ground_truth_df)
    mae_score = max(0.0, 1.0 - min(mae, 1.0))
    raw = round(0.5 * f1 + 0.5 * mae_score, 4)
    return float(max(1e-9, min(1 - 1e-9, float(raw))))