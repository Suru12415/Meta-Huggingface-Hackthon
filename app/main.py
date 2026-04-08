"""
app/main.py — FastAPI server exposing the SciCleanEnv as HTTP endpoints.
"""

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.env import SciCleanEnv
from app.models import (
    Observation,
    ResetRequest,
    StateResponse,
    StepRequest,
    StepResponse,
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
env = SciCleanEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Pre-warm: ensure datasets exist before first request."""
    import app.data_gen.generate_datasets as _gen  # noqa: F401
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SciClean-Env",
    description=(
        "A reinforcement-learning-style environment for scientific data cleaning. "
        "An AI agent receives a messy CSV dataset and must issue structured cleaning "
        "actions to produce a valid, analysis-ready dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
async def root() -> dict:
    return {"status": "ok", "name": "SciClean-Env", "version": "1.0.0"}


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset", response_model=Observation, tags=["environment"])
async def reset(body: Request) -> Observation:
    try:
        data = await body.json()
    except Exception:
        data = {}
    task_id = data.get("task_id", 1)
    seed = data.get("seed", 42)
    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return obs 
    """
    Start a new episode.

    - **task_id**: 1 = Basic Data Hygiene, 2 = Outlier Detection, 3 = Cross-Experiment
    - **seed**: optional integer for reproducible dataset generation
    """
    try:
        obs = env.reset(task_id=body.task_id, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return obs


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(body: StepRequest) -> StepResponse:
    """
    Apply one action to the current episode.

    Returns the new observation, per-step reward, done flag, and debug info.

    ### Action catalogue

    **Task 1 — Basic Data Hygiene**
    ```json
    {"action": "drop_duplicates"}
    {"action": "fill_null", "column": "col_name", "strategy": "mean|median|mode|drop"}
    {"action": "cast_column", "column": "col_name", "dtype": "float|int|str"}
    {"action": "submit"}
    ```

    **Task 2 — Outlier Detection & Unit Errors**
    ```json
    {"action": "flag_outlier", "row_id": 42}
    {"action": "drop_row", "row_id": 42}
    {"action": "rescale_column", "column": "col_name", "factor": 0.1}
    {"action": "submit"}
    ```

    **Task 3 — Cross-Experiment Consistency**
    ```json
    {"action": "rename_column", "dataset": "B", "old": "old_name", "new": "new_name"}
    {"action": "drop_row", "dataset": "B", "row_id": 5}
    {"action": "flag_contradiction", "column": "col_name", "row_id": 5}
    {"action": "merge_datasets"}
    {"action": "submit"}
    ```
    """
    try:
        result = env.step(body.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/state", response_model=StateResponse, tags=["environment"])
async def state() -> StateResponse:
    """Return current episode metadata without advancing the environment."""
    return env.get_state()
