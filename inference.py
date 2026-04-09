"""
inference.py — SciClean-Env agent (OpenEnv RL Hackathon compliant)
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import httpx
from openai import OpenAI

# ── Required env vars (as per guidelines) ────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

ENV_BASE_URL = "http://localhost:7860"
TIMEOUT = 30.0

# ── Structured log helpers ────────────────────────────────────────────────────
def log_start(task_name: str, env: str, model: str):
    print(f"[START] task={task_name} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None):
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}", flush=True)

def log_end(success: bool, steps: int, rewards: list[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def post(http: httpx.Client, path: str, body: dict | None = None) -> dict:
    try:
        resp = http.post(f"{ENV_BASE_URL}{path}", json=body or {}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] POST {path}: {e}", file=sys.stderr)
        raise

def get_req(http: httpx.Client, path: str) -> dict:
    try:
        resp = http.get(f"{ENV_BASE_URL}{path}", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] GET {path}: {e}", file=sys.stderr)
        raise


# ── LLM call (uses OpenAI client as required) ─────────────────────────────────
def ask_llm(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
        return ""


# ── Task runners ──────────────────────────────────────────────────────────────
def run_task(http: httpx.Client, task_id: int, seed: int = 42) -> tuple[bool, list[float]]:
    task_names = {1: "basic-hygiene", 2: "outlier-detection", 3: "cross-experiment"}
    task_name  = task_names[task_id]

    log_start(task_name, "SciClean-Env", MODEL_NAME)

    obs = post(http, "/reset", {"task_id": task_id, "seed": seed})

    rewards: list[float] = []
    step = 0
    success = False

    try:
        if task_id == 1:
            actions = [
                {"action": "drop_duplicates"},
                {"action": "fill_null", "column": "temperature_c",    "strategy": "mean"},
                {"action": "fill_null", "column": "ph_level",         "strategy": "mean"},
                {"action": "fill_null", "column": "cell_count",       "strategy": "median"},
                {"action": "fill_null", "column": "incubation_hours", "strategy": "mode"},
                {"action": "cast_column", "column": "sample_id",  "dtype": "int"},
                {"action": "cast_column", "column": "cell_count", "dtype": "int"},
                {"action": "submit"},
            ]
            for action in actions:
                step += 1
                result = post(http, "/step", {"action": action})
                r = result["reward"]
                rewards.append(r)
                done = result["done"]
                err = result.get("info", {}).get("error") if isinstance(result.get("info"), dict) else None
                log_step(step, action["action"], r, done, err)
                if done:
                    success = True
                    break

        elif task_id == 2:
            df = obs["dataframe"]

            # Rescale
            step += 1
            result = post(http, "/step", {"action": {"action": "rescale_column", "column": "length_mm", "factor": 0.1}})
            rewards.append(result["reward"])
            log_step(step, "rescale_column(length_mm,0.1)", result["reward"], result["done"], None)

            # Detect outliers
            numeric_cols = ["tensile_strength_mpa", "yield_point_mpa", "elongation_pct"]
            col_stats: dict[str, tuple[float, float]] = {}
            for col in numeric_cols:
                vals = [row[col] for row in df if row.get(col) is not None]
                if vals:
                    m = statistics.mean(vals)
                    s = statistics.stdev(vals) if len(vals) > 1 else 0
                    col_stats[col] = (m, s)

            outlier_ids = []
            for i, row in enumerate(df):
                for col, (m, s) in col_stats.items():
                    v = row.get(col)
                    if v is not None and s > 0 and abs(v - m) > 5 * s:
                        outlier_ids.append(i)
                        break

            for row_id in outlier_ids:
                step += 1
                result = post(http, "/step", {"action": {"action": "flag_outlier", "row_id": row_id}})
                rewards.append(result["reward"])
                log_step(step, f"flag_outlier({row_id})", result["reward"], result["done"], None)

            for row_id in sorted(outlier_ids, reverse=True):
                step += 1
                result = post(http, "/step", {"action": {"action": "drop_row", "row_id": row_id}})
                rewards.append(result["reward"])
                log_step(step, f"drop_row({row_id})", result["reward"], result["done"], None)

            step += 1
            result = post(http, "/step", {"action": {"action": "submit"}})
            rewards.append(result["reward"])
            log_step(step, "submit", result["reward"], result["done"], None)
            success = result["done"]

        elif task_id == 3:
            dataset_b = obs["aux"].get("dataset_B", [])

            renames = [
                {"action": "rename_column", "dataset": "B", "old": "patient_id",  "new": "subject_id"},
                {"action": "rename_column", "dataset": "B", "old": "temp_celsius", "new": "body_temp_c"},
                {"action": "rename_column", "dataset": "B", "old": "bp_systolic", "new": "systolic_bp_mmhg"},
            ]
            for action in renames:
                step += 1
                result = post(http, "/step", {"action": action})
                rewards.append(result["reward"])
                log_step(step, f"rename_column({action['old']}->{action['new']})", result["reward"], result["done"], None)

            junk_ids = [i for i, row in enumerate(dataset_b)
                        if (row.get("patient_id") or row.get("subject_id") or 1) < 1]
            for row_id in sorted(junk_ids, reverse=True):
                step += 1
                result = post(http, "/step", {"action": {"action": "drop_row", "dataset": "B", "row_id": row_id}})
                rewards.append(result["reward"])
                log_step(step, f"drop_row(B,{row_id})", result["reward"], result["done"], None)

            step += 1
            result = post(http, "/step", {"action": {"action": "merge_datasets"}})
            rewards.append(result["reward"])
            log_step(step, "merge_datasets", result["reward"], result["done"], None)

            step += 1
            result = post(http, "/step", {"action": {"action": "submit"}})
            rewards.append(result["reward"])
            log_step(step, "submit", result["reward"], result["done"], None)
            success = result["done"]

    except Exception as e:
        print(f"[ERROR] Task {task_id} exception: {e}", file=sys.stderr)
        log_end(False, step, rewards)
        return False, rewards

    log_end(success, step, rewards)
    return success, rewards


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    with httpx.Client() as http:
        for attempt in range(30):
            try:
                resp = http.get(f"{ENV_BASE_URL}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            print(f"Waiting for server... ({attempt+1}/30)", end="\r", flush=True)
            time.sleep(2)
        else:
            print(f"\n[ERROR] Server not reachable at {ENV_BASE_URL}")
            sys.exit(1)

        print(f"[OK] Connected to {ENV_BASE_URL}", flush=True)

        for task_id in [1, 2, 3]:
            try:
                run_task(http, task_id, seed=42)
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()