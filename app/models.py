"""
app/models.py — Pydantic v2 schemas for SciClean-Env API.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=3, description="Task to load (1=easy, 2=medium, 3=hard)")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(
        ...,
        description="Action dict with at minimum an 'action' key",
        examples=[
            {"action": "drop_duplicates"},
            {"action": "fill_null", "column": "temperature", "strategy": "mean"},
            {"action": "cast_column", "column": "sample_id", "dtype": "int"},
            {"action": "flag_outlier", "row_id": 42},
            {"action": "rescale_column", "column": "length_mm", "factor": 0.1},
            {"action": "submit"},
        ],
    )


# ---------------------------------------------------------------------------
# Response/observation bodies
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: int
    step: int
    max_steps: int
    dataframe: list[dict[str, Any]] = Field(
        description="Current working dataset as list of row dicts"
    )
    aux: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific auxiliary data (e.g. dataset_B for task 3)",
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: float = Field(description="Reward for this action, 0.0–1.0")
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    episode_id: str
    task_id: int | None
    step: int
    done: bool
    cumulative_reward: float
