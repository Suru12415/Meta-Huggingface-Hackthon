"""
app/graders/grader1.py — Task 1 grader: Cell-level F1 vs ground truth.

score() returns a float in [0.0, 1.0]:
  1.0 → the agent's dataframe perfectly matches the ground truth
  0.0 → completely wrong
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise(df: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align agent df to reference shape/columns for fair comparison."""
    # Drop duplicate rows from agent df before scoring (same as drop_duplicates action)
    df = df.drop_duplicates().reset_index(drop=True)

    # Align columns
    common_cols = [c for c in ref.columns if c in df.columns]
    df = df[common_cols].copy()
    ref = ref[common_cols].copy()

    # Truncate/pad to same length
    min_len = min(len(df), len(ref))
    df = df.iloc[:min_len].reset_index(drop=True)
    ref = ref.iloc[:min_len].reset_index(drop=True)

    # Coerce types
    for col in common_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        ref[col] = pd.to_numeric(ref[col], errors="coerce")

    return df, ref


def score(agent_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Computes cell-level match score between agent_df and ground_truth_df.

    Cell is a "true positive" if:
      - both are NaN (agent preserved the null for a drop strategy), OR
      - numeric values are within 1 % relative tolerance.

    Returns float in [0.0, 1.0].
    """
    if agent_df is None or len(agent_df) == 0:
        return 0.0

    df, ref = _normalise(agent_df, ground_truth_df)

    if df.empty or ref.empty:
        return 0.0

    total_cells = df.size
    if total_cells == 0:
        return 0.0

    matching = 0
    for col in df.columns:
        a_col = df[col].values
        r_col = ref[col].values

        both_nan = np.isnan(a_col.astype(float)) & np.isnan(r_col.astype(float))
        both_valid = ~np.isnan(a_col.astype(float)) & ~np.isnan(r_col.astype(float))

        # within 1 % relative tolerance for valid cells
        within_tol = np.zeros(len(a_col), dtype=bool)
        within_tol[both_valid] = np.isclose(
            a_col[both_valid].astype(float),
            r_col[both_valid].astype(float),
            rtol=0.01,
            atol=1e-6,
        )

        matching += int(both_nan.sum()) + int(within_tol.sum())

    return float(np.clip(round(matching / total_cells, 4), 0.0001, 0.9999))

