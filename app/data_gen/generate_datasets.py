"""
app/data_gen/generate_datasets.py — Pre-generate and save all task datasets.

Can be run as a module:
    python -m app.data_gen.generate_datasets

Also imported at server startup to ensure CSVs exist before the first request.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from app.tasks import task1_hygiene, task2_outliers, task3_crossvalidate

DATA_DIR = pathlib.Path(__file__).parent / "data"


def _save(df: pd.DataFrame, name: str) -> pathlib.Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    df.to_csv(path, index=False)
    return path


def generate_all(seed: int = 42) -> None:
    """Generate and persist ground-truth + dirty CSVs for all tasks."""

    # ── Task 1 ─────────────────────────────────────────────────────────────
    t1 = task1_hygiene.generate(seed=seed)
    _save(t1["clean"], "task1_clean.csv")
    _save(t1["dirty"], "task1_dirty.csv")
    print(f"[task1] clean={t1['clean'].shape}, dirty={t1['dirty'].shape}")

    # ── Task 2 ─────────────────────────────────────────────────────────────
    t2 = task2_outliers.generate(seed=seed)
    _save(t2["clean"], "task2_clean.csv")
    _save(t2["dirty"], "task2_dirty.csv")
    import json
    (DATA_DIR / "task2_outlier_ids.json").write_text(
        json.dumps(t2["outlier_ids"])
    )
    print(
        f"[task2] clean={t2['clean'].shape}, dirty={t2['dirty'].shape}, "
        f"outlier_ids={t2['outlier_ids']}"
    )

    # ── Task 3 ─────────────────────────────────────────────────────────────
    t3 = task3_crossvalidate.generate(seed=seed)
    _save(t3["clean"], "task3_A_clean.csv")
    _save(t3["dirty"], "task3_A_dirty.csv")
    _save(t3["dirty_B"], "task3_B_dirty.csv")
    _save(t3["clean_merged"], "task3_merged_clean.csv")
    print(
        f"[task3] A_clean={t3['clean'].shape}, B_dirty={t3['dirty_B'].shape}, "
        f"merged_clean={t3['clean_merged'].shape}"
    )

    print(f"\n[OK] All datasets written to: {DATA_DIR.resolve()}")


# Auto-run when imported for the first time (idempotent - skip if files exist)
_sentinel = DATA_DIR / "task1_clean.csv"
if not _sentinel.exists():
    generate_all()


if __name__ == "__main__":
    generate_all()
