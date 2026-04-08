---
title: SciClean-Env
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---


# SciClean-Env 🧪

> **A Reinforcement-Learning Environment for Scientific Data Cleaning**
> An AI agent receives messy experimental CSV datasets and must issue structured
> cleaning actions to produce a valid, analysis-ready dataset.

---

## Why SciClean-Env?

| Property | Detail |
|---|---|
| **Real-world** | Scientists, pharma analysts, and lab technicians clean data daily |
| **Deterministic grading** | Synthetic ground truth → grader is 100 % objective |
| **Rich partial rewards** | Score improves incrementally at every step (0.0 → 1.0) |
| **No GPU required** | Pure CPU, pandas + numpy, starts in < 3 seconds |
| **Scales nicely** | Easy → Medium → Hard creates clear performance gradients |

---

## Quick Start

### Bare Python (development)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```bash
python inference.py          # run all 3 tasks with heuristic agent
python inference.py --task 2 # run only task 2
```

### Docker

```bash
docker build -t sciclean-env .
docker run -p 8000:8000 sciclean-env
```

### Interactive API docs

Open `http://localhost:8000/docs` in your browser.

---

## Three Tasks

| # | Name | Difficulty | Errors Injected | Grader |
|---|---|---|---|---|
| 1 | Basic Data Hygiene | Easy | Duplicates, nulls, wrong dtype | Cell-level match F1 |
| 2 | Outlier Detection & Unit Errors | Medium | 10× outliers, mm↔cm mismatch | Outlier F1 + MAE |
| 3 | Cross-Experiment Consistency | Hard | Column mismatches, contradictions, junk rows | Pearson correlation |

---

## HTTP API

### `POST /reset`

Start a new episode.

```json
{"task_id": 1, "seed": 42}
```

Returns an **Observation** with the dirty dataframe.

### `POST /step`

Apply one action. Returns `{observation, reward, done, info}`.

### Action Catalogue

#### Task 1 — Basic Data Hygiene

```json
{"action": "drop_duplicates"}
{"action": "fill_null", "column": "temperature_c", "strategy": "mean"}
{"action": "fill_null", "column": "ph_level", "strategy": "median"}
{"action": "fill_null", "column": "incubation_hours", "strategy": "mode"}
{"action": "fill_null", "column": "temperature_c", "strategy": "drop"}
{"action": "cast_column", "column": "sample_id", "dtype": "int"}
{"action": "cast_column", "column": "cell_count", "dtype": "float"}
{"action": "submit"}
```

#### Task 2 — Outlier Detection & Unit Errors

```json
{"action": "flag_outlier", "row_id": 42}
{"action": "drop_row", "row_id": 42}
{"action": "rescale_column", "column": "length_mm", "factor": 0.1}
{"action": "submit"}
```

#### Task 3 — Cross-Experiment Consistency

```json
{"action": "rename_column", "dataset": "B", "old": "patient_id", "new": "subject_id"}
{"action": "rename_column", "dataset": "B", "old": "temp_celsius", "new": "body_temp_c"}
{"action": "rename_column", "dataset": "B", "old": "bp_systolic", "new": "systolic_bp_mmhg"}
{"action": "drop_row", "dataset": "B", "row_id": 80}
{"action": "flag_contradiction", "column": "systolic_bp_mmhg", "row_id": 5}
{"action": "merge_datasets"}
{"action": "submit"}
```

### `GET /state`

Returns episode metadata without advancing the environment.

```json
{
  "episode_id": "...",
  "task_id": 1,
  "step": 3,
  "done": false,
  "cumulative_reward": 0.42
}
```

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## Observation Schema

```json
{
  "task_id": 1,
  "step": 0,
  "max_steps": 20,
  "dataframe": [{"sample_id": "1", "temperature_c": null, ...}],
  "aux": {
    "dataset_B": [...],      // task 3 only
    "columns_A": [...],      // task 3 only
    "flagged_outlier_ids": [], // task 2 only
    "num_known_outliers": 7   // task 2 only
  }
}
```

---

## Grading Details

### Task 1 — Cell-level match rate

Score = (cells within 1 % of ground truth) / (total cells).

### Task 2 — Composite

```
score = 0.5 × F1(detected outliers) + 0.5 × (1 − normalised MAE)
```

### Task 3 — Correlation quality

```
score = mean_column_pearson_r (mapped to [0,1]) × shape_factor
```
where `shape_factor = min(1.0, agent_rows / truth_rows)`.

---

## Project Layout

```
sciclean-env/
├── Dockerfile
├── openenv.yaml          # environment metadata
├── inference.py          # sample heuristic agent loop
├── README.md
├── requirements.txt
└── app/
    ├── main.py           # FastAPI server
    ├── env.py            # SciCleanEnv state machine
    ├── models.py         # Pydantic schemas
    ├── tasks/
    │   ├── task1_hygiene.py
    │   ├── task2_outliers.py
    │   └── task3_crossvalidate.py
    ├── graders/
    │   ├── grader1.py
    │   ├── grader2.py
    │   └── grader3.py
    └── data_gen/
        └── generate_datasets.py
```
