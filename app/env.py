"""
app/env.py — SciCleanEnv: the core episode state machine.

Holds one active episode at a time. Thread-safety is not required since
the environment is designed for a single agent.
"""
from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from app.models import Observation, StepResponse, StateResponse
from app.tasks import task1_hygiene, task2_outliers, task3_crossvalidate
from app.graders import grader1, grader2, grader3


# Maps task_id → (task module, grader module, max_steps)
TASK_REGISTRY: dict[int, tuple[Any, Any, int]] = {
    1: (task1_hygiene, grader1, 20),
    2: (task2_outliers, grader2, 30),
    3: (task3_crossvalidate, grader3, 40),
}


class SciCleanEnv:
    """Gym-style environment for scientific data cleaning."""

    def __init__(self) -> None:
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.episode_id: str = str(uuid.uuid4())
        self.task_id: int | None = None
        self.step_count: int = 0
        self.max_steps: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0

        # DataFrames
        self.current_df: pd.DataFrame | None = None
        self.ground_truth_df: pd.DataFrame | None = None

        # Task-3 extras
        self.current_df_B: pd.DataFrame | None = None
        self.ground_truth_merged: pd.DataFrame | None = None

        # Task-2 extras (flagged outlier ids)
        self.flagged_outlier_ids: list[int] = []
        self.known_outlier_ids: list[int] = []

        # Active task module + grader
        self._task_mod: Any = None
        self._grader_mod: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: int, seed: int | None = None) -> Observation:
        """Start a new episode."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id={task_id}. Must be 1, 2, or 3.")

        task_mod, grader_mod, max_steps = TASK_REGISTRY[task_id]

        self._reset_state()
        self.task_id = task_id
        self.max_steps = max_steps
        self._task_mod = task_mod
        self._grader_mod = grader_mod

        # Each task module exposes a `generate(seed)` function that returns
        # (dirty_df, ground_truth_df, **extras)
        result = task_mod.generate(seed=seed)

        self.current_df = result["dirty"].copy()
        self.ground_truth_df = result["clean"]

        aux: dict[str, Any] = {}

        if task_id == 2:
            self.known_outlier_ids = result.get("outlier_ids", [])
            self.flagged_outlier_ids = []
            aux["known_columns"] = list(self.current_df.columns)

        if task_id == 3:
            self.current_df_B = result.get("dirty_B", pd.DataFrame()).copy()
            self.ground_truth_merged = result.get("clean_merged")
            aux["dataset_B"] = self.current_df_B.to_dict(orient="records")
            aux["columns_A"] = list(self.current_df.columns)
            aux["columns_B"] = list(self.current_df_B.columns)

        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            max_steps=self.max_steps,
            dataframe=self.current_df.to_dict(orient="records"),
            aux=aux,
        )

    def step(self, action: dict[str, Any]) -> StepResponse:
        """Apply one action and return (observation, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is done. Call /reset to start a new one.")
        if self.task_id is None:
            raise RuntimeError("No active episode. Call /reset first.")

        self.step_count += 1
        info: dict[str, Any] = {}

        action_name = action.get("action", "")

        # ── Finalise submission ──────────────────────────────────────────
        if action_name == "submit":
            reward = self._grade()
            self.cumulative_reward += reward
            self.done = True
            info["message"] = "Episode finalised via submit action."
            info["final_score"] = reward
        # ── Step limit reached ───────────────────────────────────────────
        elif self.step_count >= self.max_steps:
            reward = self._grade()
            self.cumulative_reward += reward
            self.done = True
            info["message"] = "Max steps reached."
        else:
            # Apply action, get incremental reward
            reward = self._apply_action(action, info)
            self.cumulative_reward += reward

        obs = self._build_obs()
        return StepResponse(
            observation=obs,
            reward=round(reward, 4),
            done=self.done,
            info=info,
        )

    def get_state(self) -> StateResponse:
        return StateResponse(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step=self.step_count,
            done=self.done,
            cumulative_reward=round(self.cumulative_reward, 4),
        )

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: dict[str, Any], info: dict[str, Any]) -> float:
        """Route action to task-specific handler. Returns per-step reward."""
        name = action.get("action", "")
        reward = 0.0

        try:
            if self.task_id == 1:
                reward = self._handle_task1(name, action, info)
            elif self.task_id == 2:
                reward = self._handle_task2(name, action, info)
            elif self.task_id == 3:
                reward = self._handle_task3(name, action, info)
            else:
                info["error"] = "Unknown task."
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
            reward = -0.01  # small penalty for invalid actions

        return reward

    # ── Task 1 handlers ─────────────────────────────────────────────────

    def _handle_task1(
        self, name: str, action: dict[str, Any], info: dict[str, Any]
    ) -> float:
        df = self.current_df
        before_score = grader1.score(df, self.ground_truth_df)

        if name == "drop_duplicates":
            self.current_df = df.drop_duplicates().reset_index(drop=True)
        elif name == "fill_null":
            col = action["column"]
            strategy = action.get("strategy", "mean")
            self.current_df = task1_hygiene.fill_null(df, col, strategy)
        elif name == "cast_column":
            col = action["column"]
            dtype = action["dtype"]
            self.current_df = task1_hygiene.cast_column(df, col, dtype)
        else:
            info["error"] = f"Unknown task-1 action: '{name}'"
            return -0.01

        after_score = grader1.score(self.current_df, self.ground_truth_df)
        delta = after_score - before_score
        info["score_delta"] = round(delta, 4)
        info["current_score"] = round(after_score, 4)
        return max(delta, 0.0)  # only reward improvements

    # ── Task 2 handlers ─────────────────────────────────────────────────

    def _handle_task2(
        self, name: str, action: dict[str, Any], info: dict[str, Any]
    ) -> float:
        df = self.current_df

        if name == "flag_outlier":
            row_id = int(action["row_id"])
            if row_id not in self.flagged_outlier_ids:
                self.flagged_outlier_ids.append(row_id)
            # Reward: fraction of known outliers now flagged
            precision, recall = grader2.outlier_pr(
                self.flagged_outlier_ids, self.known_outlier_ids
            )
            info["precision"] = round(precision, 4)
            info["recall"] = round(recall, 4)
            return round(recall * 0.05, 4)  # small incremental reward per flag

        elif name == "drop_row":
            row_id = int(action["row_id"])
            if row_id in self.current_df.index:
                self.current_df = df.drop(index=row_id).reset_index(drop=True)
                # Also remove from flagged list if present
                self.flagged_outlier_ids = [
                    r for r in self.flagged_outlier_ids if r != row_id
                ]
            before_mae = grader2.mae_ratio(df, self.ground_truth_df)
            after_mae = grader2.mae_ratio(self.current_df, self.ground_truth_df)
            delta = before_mae - after_mae  # lower MAE is better
            info["mae_improvement"] = round(delta, 4)
            return max(delta * 0.5, 0.0)

        elif name == "rescale_column":
            col = action["column"]
            factor = float(action["factor"])
            before_score = grader2.score(
                self.current_df, self.ground_truth_df,
                self.flagged_outlier_ids, self.known_outlier_ids,
            )
            self.current_df = task2_outliers.rescale_column(df, col, factor)
            after_score = grader2.score(
                self.current_df, self.ground_truth_df,
                self.flagged_outlier_ids, self.known_outlier_ids,
            )
            delta = after_score - before_score
            info["score_delta"] = round(delta, 4)
            return max(delta, 0.0)

        else:
            info["error"] = f"Unknown task-2 action: '{name}'"
            return -0.01

    # ── Task 3 handlers ─────────────────────────────────────────────────

    def _handle_task3(
        self, name: str, action: dict[str, Any], info: dict[str, Any]
    ) -> float:
        if name == "rename_column":
            dataset = action.get("dataset", "B")
            old, new = action["old"], action["new"]
            if dataset == "B" and self.current_df_B is not None:
                if old in self.current_df_B.columns:
                    self.current_df_B = self.current_df_B.rename(columns={old: new})
                    info["renamed"] = f"B: {old} → {new}"
                    return 0.05
                else:
                    info["error"] = f"Column '{old}' not found in dataset B."
                    return -0.01
            elif dataset == "A":
                if old in self.current_df.columns:
                    self.current_df = self.current_df.rename(columns={old: new})
                    info["renamed"] = f"A: {old} → {new}"
                    return 0.05
                else:
                    info["error"] = f"Column '{old}' not found in dataset A."
                    return -0.01
            return -0.01

        elif name == "drop_row":
            dataset = action.get("dataset", "B")
            row_id = int(action["row_id"])
            if dataset == "B" and self.current_df_B is not None:
                if row_id in self.current_df_B.index:
                    self.current_df_B = self.current_df_B.drop(index=row_id).reset_index(drop=True)
                    return 0.02
            elif dataset == "A":
                if row_id in self.current_df.index:
                    self.current_df = self.current_df.drop(index=row_id).reset_index(drop=True)
                    return 0.02
            info["error"] = f"Row {row_id} not found or invalid dataset '{dataset}'."
            return -0.01

        elif name == "flag_contradiction":
            # Informational action — small fixed reward for using it
            col = action.get("column", "")
            row_id = action.get("row_id", -1)
            info["flagged_contradiction"] = {"column": col, "row_id": row_id}
            return 0.02

        elif name == "merge_datasets":
            if self.current_df_B is None:
                info["error"] = "No dataset B to merge."
                return -0.05
            try:
                merged = task3_crossvalidate.merge_datasets(
                    self.current_df, self.current_df_B
                )
                # Score the merge quality
                score = grader3.score(merged, self.ground_truth_merged)
                self.current_df = merged
                self.current_df_B = None  # consumed
                info["merge_score"] = round(score, 4)
                info["merged_shape"] = list(merged.shape)
                return score * 0.5  # partial reward; rest comes at submit
            except Exception as exc:  # noqa: BLE001
                info["error"] = f"Merge failed: {exc}"
                return -0.05

        else:
            info["error"] = f"Unknown task-3 action: '{name}'"
            return -0.01

    # ------------------------------------------------------------------
    # Grading helpers
    # ------------------------------------------------------------------

    def _grade(self) -> float:
        """Final score for the current episode state."""
        if self.task_id == 1:
            s = grader1.score(self.current_df, self.ground_truth_df)
        elif self.task_id == 2:
            s = grader2.score(
                self.current_df, self.ground_truth_df,
                self.flagged_outlier_ids, self.known_outlier_ids,
            )
        elif self.task_id == 3:
            if self.current_df_B is not None:
                try:
                    merged = task3_crossvalidate.merge_datasets(
                        self.current_df, self.current_df_B
                    )
                except Exception:
                    merged = self.current_df
            else:
                merged = self.current_df
            s = grader3.score(merged, self.ground_truth_merged)
        else:
            s = 0.0
        return float(max(0.0001, min(0.9999, s)))

    def _build_obs(self) -> Observation:
        aux: dict[str, Any] = {}
        if self.task_id == 3 and self.current_df_B is not None:
            aux["dataset_B"] = self.current_df_B.to_dict(orient="records")
            aux["columns_A"] = list(self.current_df.columns)
            aux["columns_B"] = list(self.current_df_B.columns)
        if self.task_id == 2:
            aux["flagged_outlier_ids"] = self.flagged_outlier_ids
            aux["num_known_outliers"] = len(self.known_outlier_ids)
        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            max_steps=self.max_steps,
            dataframe=self.current_df.to_dict(orient="records"),
            aux=aux,
        )
