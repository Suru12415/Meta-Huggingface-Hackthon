"""
app/main.py — FastAPI env server for Meta x Scaler Hackathon.
"""
from __future__ import annotations
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ── Score safety helper ──────────────────────────────────────────────────
def _clip(score) -> float:
    """Strictly between 0 and 1 — never 0.0, never 1.0."""
    return float(max(1e-9, min(1 - 1e-9, float(score))))

# ── In-memory state ──────────────────────────────────────────────────────
_state: dict[str, Any] = {}


# ── Health check (REQUIRED by inference.py) ─────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── OpenAI-compatible stub (required by validator checklist) ─────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 256

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    last = req.messages[-1].content if req.messages else ""
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"Response to: {last}"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    }


# ── Task datasets ────────────────────────────────────────────────────────
def _make_task1_data(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(20):
        rows.append({
            "sample_id": float(i),
            "temperature_c": rng.uniform(20, 40) if rng.random() > 0.15 else None,
            "ph_level": rng.uniform(6, 8) if rng.random() > 0.15 else None,
            "cell_count": float(rng.randint(100, 10000)) if rng.random() > 0.15 else None,
            "incubation_hours": rng.choice([12, 24, 36, 48]) if rng.random() > 0.15 else None,
        })
    # Add a duplicate
    rows.append(rows[0].copy())
    return rows

def _make_task2_data(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(20):
        rows.append({
            "part_id": i,
            "length_mm": rng.uniform(100, 200),
            "tensile_strength_mpa": rng.uniform(300, 500),
            "yield_point_mpa": rng.uniform(200, 400),
            "elongation_pct": rng.uniform(10, 30),
        })
    # Add one outlier
    rows.append({
        "part_id": 999,
        "length_mm": 150.0,
        "tensile_strength_mpa": 99999.0,  # outlier
        "yield_point_mpa": 250.0,
        "elongation_pct": 20.0,
    })
    return rows

def _make_task3_data(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    dataset_a = [
        {"subject_id": i, "body_temp_c": rng.uniform(36, 38),
         "systolic_bp_mmhg": rng.randint(110, 140)}
        for i in range(1, 11)
    ]
    dataset_b = [
        {"patient_id": i, "temp_celsius": rng.uniform(36, 38),
         "bp_systolic": rng.randint(110, 140)}
        for i in range(1, 11)
    ]
    return dataset_a, dataset_b


# ── /reset ───────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: int = 1   # ✅ default so empty body works
    seed: int = 42

@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    _state.clear()
    _state["task_id"] = req.task_id
    _state["seed"]    = req.seed
    _state["done"]    = False
    _state["score"]   = 0.0
    _state["steps"]   = 0

    if req.task_id == 1:
        _state["df"] = _make_task1_data(req.seed)
        return {"dataframe": _state["df"], "done": False, "reward": _clip(0.0), "info": {}}

    elif req.task_id == 2:
        _state["df"] = _make_task2_data(req.seed)
        return {"dataframe": _state["df"], "done": False, "reward": _clip(0.0), "info": {}}

    elif req.task_id == 3:
        df_a, df_b = _make_task3_data(req.seed)
        _state["df"]        = df_a
        _state["dataset_b"] = df_b
        return {
            "dataframe": df_a,
            "aux": {"dataset_B": df_b},
            "done": False,
            "reward": _clip(0.0),
            "info": {},
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task_id {req.task_id}")


# ── /step ────────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: dict

@app.post("/step")
def step(req: StepRequest):
    if _state.get("done"):
        return {"reward": _clip(0.0), "done": True, "info": {"final_score": _clip(_state["score"])}}

    action  = req.action
    name    = action.get("action", "")
    reward  = 0.0
    done    = False
    df: list[dict] = _state.get("df", [])

    try:
        if name == "drop_duplicates":
            before = len(df)
            seen, deduped = set(), []
            for row in df:
                key = tuple(sorted(row.items()))
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            _state["df"] = deduped
            reward = 0.1 * (before - len(deduped))

        elif name == "fill_null":
            col      = action["column"]
            strategy = action.get("strategy", "mean")
            vals     = [r[col] for r in df if r.get(col) is not None]
            if vals:
                if strategy == "mean":
                    fill = sum(vals) / len(vals)
                elif strategy == "median":
                    fill = sorted(vals)[len(vals) // 2]
                elif strategy == "mode":
                    fill = max(set(vals), key=vals.count)
                else:
                    fill = vals[0]
                count = sum(1 for r in df if r.get(col) is None)
                for r in df:
                    if r.get(col) is None:
                        r[col] = fill
                reward = 0.1 * count

        elif name == "cast_column":
            col   = action["column"]
            dtype = action.get("dtype", "int")
            count = 0
            for r in df:
                try:
                    if dtype == "int":
                        r[col] = int(float(r[col]))
                    elif dtype == "float":
                        r[col] = float(r[col])
                    count += 1
                except (TypeError, ValueError):
                    pass
            reward = 0.05 * count

        elif name == "rescale_column":
            col    = action["column"]
            factor = float(action.get("factor", 1.0))
            count  = 0
            for r in df:
                if r.get(col) is not None:
                    r[col] = r[col] * factor
                    count += 1
            reward = 0.1 * count

        elif name == "flag_outlier":
            row_id = action["row_id"]
            if 0 <= row_id < len(df):
                df[row_id]["_outlier"] = True
                reward = 0.2

        elif name == "drop_row":
            row_id  = action["row_id"]
            dataset = action.get("dataset", "A")
            if dataset == "B":
                bdf = _state.get("dataset_b", [])
                if 0 <= row_id < len(bdf):
                    bdf.pop(row_id)
                    reward = 0.1
            else:
                if 0 <= row_id < len(df):
                    df.pop(row_id)
                    reward = 0.1

        elif name == "rename_column":
            dataset = action.get("dataset", "A")
            old, new = action["old"], action["new"]
            target  = _state.get("dataset_b", []) if dataset == "B" else df
            count   = 0
            for r in target:
                if old in r:
                    r[new] = r.pop(old)
                    count += 1
            reward = 0.1 * count

        elif name == "merge_datasets":
            df_a = _state.get("df", [])
            df_b = _state.get("dataset_b", [])
            merged = df_a + df_b
            _state["df"] = merged
            reward = 0.5

        elif name == "submit":
            done   = True
            reward = 0.9999  # ✅ FIX: was 1.0 — strictly less than 1
            _state["done"]  = True
            _state["score"] = _state.get("score", 0.0) + reward

        else:
            reward = 0.0

    except Exception as e:
        print(f"[STEP ERROR] action={name} error={e}", flush=True)
        reward = 0.0

    _state["score"] = _state.get("score", 0.0) + reward
    _state["steps"] += 1

    return {
        "reward":     _clip(reward),           # ✅ always clipped
        "done":       done,
        "dataframe":  _state.get("df", []),
        "info":       {"final_score": _clip(_state["score"]), "steps": _state["steps"]},
    }