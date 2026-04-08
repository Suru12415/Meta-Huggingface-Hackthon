"""
inference.py — Fixed agent loop for SciClean-Env.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


def post(client: httpx.Client, path: str, body: dict | None = None) -> dict:
    try:
        resp = client.post(f"{BASE_URL}{path}", json=body or {}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP {e.response.status_code} on POST {path}: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"[ERROR] Request failed on POST {path}: {e}", file=sys.stderr)
        sys.exit(1)


def get(client: httpx.Client, path: str) -> dict:
    try:
        resp = client.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP {e.response.status_code} on GET {path}: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"[ERROR] Request failed on GET {path}: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-task heuristic strategies
# ---------------------------------------------------------------------------

def run_task1(client: httpx.Client, seed: int) -> float:
    obs = post(client, "/reset", {"task_id": 1, "seed": seed})
    print(f"\n[Task 1] Reset. Dirty rows: {len(obs['dataframe'])}")

    actions = [
        {"action": "drop_duplicates"},
        {"action": "fill_null", "column": "temperature_c", "strategy": "mean"},
        {"action": "fill_null", "column": "ph_level", "strategy": "mean"},
        {"action": "fill_null", "column": "cell_count", "strategy": "median"},
        {"action": "fill_null", "column": "incubation_hours", "strategy": "mode"},
        {"action": "cast_column", "column": "sample_id", "dtype": "int"},
        {"action": "cast_column", "column": "cell_count", "dtype": "int"},
        {"action": "submit"},
    ]

    total_reward = 0.0
    for action in actions:
        result = post(client, "/step", {"action": action})
        total_reward += result["reward"]
        status = "[done]" if result["done"] else f"step {result['observation']['step']}"
        print(
            f"  {action['action']:20s}  reward={result['reward']:.4f}  "
            f"cumulative={total_reward:.4f}  [{status}]"
            + (f"  info={result['info']}" if result["info"] else "")
        )
        if result["done"]:
            break

    return total_reward


def run_task2(client: httpx.Client, seed: int) -> float:
    obs = post(client, "/reset", {"task_id": 2, "seed": seed})
    df = obs["dataframe"]
    print(f"\n[Task 2] Reset. Rows: {len(df)}")

    total_reward = 0.0

    # Step 1: Rescale length_mm
    r = post(client, "/step", {"action": {"action": "rescale_column", "column": "length_mm", "factor": 0.1}})
    total_reward += r["reward"]
    print(f"  rescale_column(length_mm, 0.1)  reward={r['reward']:.4f}")

    # ✅ FIX: Use original df from reset — DO NOT call /reset again after steps!
    numeric_cols = ["tensile_strength_mpa", "yield_point_mpa", "elongation_pct"]

    col_stats: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        vals = [row[col] for row in df if row.get(col) is not None]
        if vals:
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0
            col_stats[col] = (mean, stdev)

    outlier_row_ids = []
    for i, row in enumerate(df):
        for col, (mean, stdev) in col_stats.items():
            val = row.get(col)
            if val is not None and stdev > 0 and abs(val - mean) > 5 * stdev:
                outlier_row_ids.append(i)
                break

    print(f"  Detected {len(outlier_row_ids)} outlier rows: {outlier_row_ids[:10]}")

    for row_id in outlier_row_ids:
        r = post(client, "/step", {"action": {"action": "flag_outlier", "row_id": row_id}})
        total_reward += r["reward"]

    for row_id in sorted(outlier_row_ids, reverse=True):
        r = post(client, "/step", {"action": {"action": "drop_row", "row_id": row_id}})
        total_reward += r["reward"]

    r = post(client, "/step", {"action": {"action": "submit"}})
    total_reward += r["reward"]
    print(f"  submit  reward={r['reward']:.4f}  final_score={r['info'].get('final_score', '?')}")

    return total_reward


def run_task3(client: httpx.Client, seed: int) -> float:
    obs = post(client, "/reset", {"task_id": 3, "seed": seed})
    dataset_b = obs['aux'].get('dataset_B', [])
    print(f"\n[Task 3] Reset. Dataset A rows: {len(obs['dataframe'])}, "
          f"Dataset B rows: {len(dataset_b)}")

    total_reward = 0.0
    rename_actions = [
        {"action": "rename_column", "dataset": "B", "old": "patient_id",   "new": "subject_id"},
        {"action": "rename_column", "dataset": "B", "old": "temp_celsius",  "new": "body_temp_c"},
        {"action": "rename_column", "dataset": "B", "old": "bp_systolic",  "new": "systolic_bp_mmhg"},
    ]

    for action in rename_actions:
        r = post(client, "/step", {"action": action})
        total_reward += r["reward"]
        print(f"  {action['action']}  {action.get('old','')} -> {action.get('new','')}  reward={r['reward']:.4f}")

    # ✅ FIX: Detect junk rows dynamically instead of hardcoding [80-84]
    junk_row_ids = []
    for i, row in enumerate(dataset_b):
        pid = row.get("patient_id") or row.get("subject_id")
        if pid is not None and pid < 1:
            junk_row_ids.append(i)

    print(f"  Dropping {len(junk_row_ids)} junk rows: {junk_row_ids}")
    for row_id in sorted(junk_row_ids, reverse=True):
        r = post(client, "/step", {"action": {"action": "drop_row", "dataset": "B", "row_id": row_id}})
        total_reward += r["reward"]

    r = post(client, "/step", {"action": {"action": "merge_datasets"}})
    total_reward += r["reward"]
    print(f"  merge_datasets  reward={r['reward']:.4f}  info={r['info']}")

    r = post(client, "/step", {"action": {"action": "submit"}})
    total_reward += r["reward"]
    print(f"  submit  reward={r['reward']:.4f}  final_score={r['info'].get('final_score', '?')}")

    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_URL

    parser = argparse.ArgumentParser(description="SciClean-Env sample inference agent")
    parser.add_argument("--host", default=BASE_URL, help="Base URL of the environment server")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="Run only this task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    BASE_URL = args.host.rstrip("/")
    tasks_to_run = [args.task] if args.task else [1, 2, 3]
    task_runners = {1: run_task1, 2: run_task2, 3: run_task3}

    with httpx.Client() as client:
        for attempt in range(30):
            try:
                resp = client.get(f"{BASE_URL}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            print(f"Waiting for server... ({attempt + 1}/30)", end="\r", flush=True)
            time.sleep(2)
        else:
            print(f"\n[ERROR] Could not reach server at {BASE_URL}. Is it running?")
            sys.exit(1)

        print(f"[OK] Connected to {BASE_URL}")

        totals: dict[int, float] = {}
        for task_id in tasks_to_run:
            try:
                reward = task_runners[task_id](client, seed=args.seed)
                totals[task_id] = reward
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
                sys.exit(1)

            state = get(client, "/state")
            print(f"  -> Final state: {state}")

    print("\n" + "=" * 55)
    print("  INFERENCE SUMMARY")
    print("=" * 55)
    for tid, r in totals.items():
        print(f"  Task {tid}: cumulative reward = {r:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()