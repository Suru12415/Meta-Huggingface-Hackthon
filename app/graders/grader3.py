"""
app/graders/grader3.py — Task 3 grader: Cross-experiment merge quality.

Score = average column-wise Pearson correlation between the agent's merged
        dataset and the ground-truth merged dataset, with a shape penalty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(
    merged_df: pd.DataFrame | None,
    ground_truth_merged: pd.DataFrame | None,
) -> float:
    """
    Returns a float in [0.0, 1.0]:
        1.0 → merged_df perfectly matches ground_truth_merged (columns and values)
        0.0 → no overlap at all

    Steps:
      1. Find common numeric columns.
      2. Align lengths (min of the two).
      3. Compute Pearson r per column.
      4. Convert r → [0, 1] via (r + 1) / 2.
      5. Apply shape penalty: rows_agent / rows_truth, clipped to [0, 1].
      6. Final score = mean column score × shape factor.
    """
    if merged_df is None or ground_truth_merged is None:
        return 0.0
    if len(merged_df) == 0:
        return 0.0

    # Identify numeric columns common to both
    numeric_truth_cols = ground_truth_merged.select_dtypes(include=[np.number]).columns
    numeric_agent_cols = merged_df.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_truth_cols if c in numeric_agent_cols]

    if not common_cols:
        return 0.0

    min_len = min(len(merged_df), len(ground_truth_merged))
    col_scores: list[float] = []

    for col in common_cols:
        a = pd.to_numeric(merged_df[col].iloc[:min_len], errors="coerce").fillna(0).values
        r = pd.to_numeric(ground_truth_merged[col].iloc[:min_len], errors="coerce").fillna(0).values

        if r.std() < 1e-9 and a.std() < 1e-9:
            col_scores.append(1.0)  # both constant and equal
            continue
        if r.std() < 1e-9 or a.std() < 1e-9:
            col_scores.append(0.0)
            continue

        corr = float(np.corrcoef(a, r)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        col_scores.append((corr + 1.0) / 2.0)  # map [-1,1] → [0,1]

    mean_col_score = float(np.mean(col_scores)) if col_scores else 0.0

    # Shape penalty: penalise if agent produced far fewer rows than ground truth
    shape_factor = min(1.0, len(merged_df) / max(1, len(ground_truth_merged)))

    return float(max(0.0001, min(0.9999, round(mean_col_score * shape_factor, 4))))
