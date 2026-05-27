"""Compute held-out NLL/BIC for trivial reference baselines on the SAME
train/test split as ``analysis.fit_irt``. Appends rows to fit_table.csv,
augments fit_summary.json, and refreshes headline_findings.json's
model_fit_table.

Baselines reported (all probabilistic, calibrated by held-out NLL):
    constant            global mean correctness         (1 param)
    subject_mean        per-subject mean correctness    (J params)
    pathology_mean      per-pathology mean correctness  (14 params)
    subj_x_pathology    per (subject, pathology) cell   (J*14 params)
    item_mean_smoothed  per-item smoothed mean          (I params, alpha=2)

The first four mirror common stratification baselines; the last shows what
a per-item lookup with mild Laplace smoothing achieves. Comparing the
IRT/factor models against them shows whether psychometric latent structure
adds anything over simple averages.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import load_all  # noqa: E402
from analysis.fit_irt import (  # noqa: E402
    build_index,
    summarize_corpus,
    to_tensors,
    train_test_split,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
EPS = 1e-7


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    """Sum of -log p(y) across observations."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).sum())


def _bic(train_ll: float, n_train_obs: int, n_params: int) -> float:
    return math.log(n_train_obs) * n_params - 2.0 * train_ll


def _stats(
    name: str,
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    n_params: int,
) -> dict:
    train_ll = -_logloss(p_train, y_train)
    test_ll = -_logloss(p_test, y_test)
    return {
        "model": name,
        "n_params": int(n_params),
        "n_obs": int(len(y_train)),
        "log_likelihood": float(train_ll),
        "AIC": float(2 * n_params - 2 * train_ll),
        "BIC": float(_bic(train_ll, len(y_train), n_params)),
        "test_log_likelihood": float(test_ll),
        "test_nll_per_obs": float(-test_ll / max(len(y_test), 1)),
        "seconds": 0.0,
        "final_loss": float(-train_ll / max(len(y_train), 1)),
        "kind": "baseline",
    }


def _predict_constant(y_train, s_train, i_train, item_keys, s_test, i_test):
    p_global = float(y_train.mean())
    p_train = np.full(len(y_train), p_global)
    p_test = np.full(len(s_test), p_global)
    return p_train, p_test, 1


def _predict_by_subject(y_train, s_train, i_train, item_keys, s_test, i_test):
    n_subj = int(max(s_train.max(), s_test.max())) + 1
    sums = np.zeros(n_subj)
    counts = np.zeros(n_subj)
    np.add.at(sums, s_train, y_train)
    np.add.at(counts, s_train, 1.0)
    p_global = float(y_train.mean())
    p_subj = np.where(counts > 0, sums / np.maximum(counts, 1.0), p_global)
    p_train = p_subj[s_train]
    p_test = p_subj[s_test]
    n_params = int((counts > 0).sum())
    return p_train, p_test, n_params


def _predict_by_pathology(y_train, s_train, i_train, item_keys, s_test, i_test):
    item_to_path_id: dict[int, int] = {}
    path_to_id: dict[str, int] = {}
    for idx, (_img, path) in enumerate(item_keys):
        if path not in path_to_id:
            path_to_id[path] = len(path_to_id)
        item_to_path_id[idx] = path_to_id[path]
    n_paths = len(path_to_id)

    p_train_ids = np.array([item_to_path_id[int(i)] for i in i_train], dtype=np.int64)
    p_test_ids = np.array([item_to_path_id[int(i)] for i in i_test], dtype=np.int64)

    sums = np.zeros(n_paths)
    counts = np.zeros(n_paths)
    np.add.at(sums, p_train_ids, y_train)
    np.add.at(counts, p_train_ids, 1.0)
    p_global = float(y_train.mean())
    p_p = np.where(counts > 0, sums / np.maximum(counts, 1.0), p_global)

    return p_p[p_train_ids], p_p[p_test_ids], int((counts > 0).sum())


def _predict_by_subject_x_pathology(
    y_train, s_train, i_train, item_keys, s_test, i_test
):
    item_to_path_id: dict[int, int] = {}
    path_to_id: dict[str, int] = {}
    for idx, (_img, path) in enumerate(item_keys):
        if path not in path_to_id:
            path_to_id[path] = len(path_to_id)
        item_to_path_id[idx] = path_to_id[path]
    n_paths = len(path_to_id)
    n_subj = int(max(s_train.max(), s_test.max())) + 1

    p_train_pids = np.array([item_to_path_id[int(i)] for i in i_train], dtype=np.int64)
    p_test_pids = np.array([item_to_path_id[int(i)] for i in i_test], dtype=np.int64)

    flat_train = s_train * n_paths + p_train_pids
    flat_test = s_test * n_paths + p_test_pids
    sums = np.zeros(n_subj * n_paths)
    counts = np.zeros(n_subj * n_paths)
    np.add.at(sums, flat_train, y_train)
    np.add.at(counts, flat_train, 1.0)
    p_global = float(y_train.mean())
    p_cell = np.where(counts > 0, sums / np.maximum(counts, 1.0), p_global)
    return p_cell[flat_train], p_cell[flat_test], int((counts > 0).sum())


def _predict_item_mean_smoothed(
    y_train, s_train, i_train, item_keys, s_test, i_test, alpha: float = 2.0
):
    n_items = len(item_keys)
    sums = np.zeros(n_items)
    counts = np.zeros(n_items)
    np.add.at(sums, i_train, y_train)
    np.add.at(counts, i_train, 1.0)
    p_global = float(y_train.mean())
    p_item = (sums + alpha * p_global) / (counts + alpha)
    return p_item[i_train], p_item[i_test], int((counts > 0).sum())


BASELINES = [
    ("constant",            _predict_constant),
    ("subject_mean",        _predict_by_subject),
    ("pathology_mean",      _predict_by_pathology),
    ("subj_x_pathology",    _predict_by_subject_x_pathology),
    ("item_mean_smoothed",  _predict_item_mean_smoothed),
]


def compute_baselines():
    print("──  Loading observations  ──", flush=True)
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations available.")
    subject_to_id, item_to_id, subject_names, item_keys = build_index(obs)
    s_idx, i_idx, y = to_tensors(obs, subject_to_id, item_to_id)

    s_train_t, i_train_t, y_train_t, s_test_t, i_test_t, y_test_t = train_test_split(
        s_idx, i_idx, y, test_frac=0.10, seed=0,
    )

    s_train = s_train_t.numpy()
    i_train = i_train_t.numpy()
    y_train = y_train_t.numpy()
    s_test = s_test_t.numpy()
    i_test = i_test_t.numpy()
    y_test = y_test_t.numpy()

    results: list[dict] = []
    for name, fn in BASELINES:
        print(f"──  baseline: {name}  ──", flush=True)
        p_train, p_test, n_params = fn(
            y_train, s_train, i_train, item_keys, s_test, i_test
        )
        stats = _stats(name, p_train, y_train, p_test, y_test, n_params)
        print(
            f"  n_params={stats['n_params']:>8,}  "
            f"BIC={stats['BIC']:>14,.1f}  "
            f"test-NLL/obs={stats['test_nll_per_obs']:.6f}",
            flush=True,
        )
        results.append(stats)

    return results


def write_outputs(baseline_stats: list[dict]):
    os.makedirs(OUT_DIR, exist_ok=True)

    table_path = os.path.join(OUT_DIR, "fit_table.csv")
    existing_rows: list[dict] = []
    if os.path.exists(table_path):
        with open(table_path) as f:
            existing_rows = list(csv.DictReader(f))
    existing_models = {r["model"] for r in existing_rows}

    appended = []
    for st in baseline_stats:
        if st["model"] in existing_models:
            continue
        appended.append({
            "model":               st["model"],
            "n_params":            st["n_params"],
            "n_obs":               st["n_obs"],
            "log_likelihood":      f"{st['log_likelihood']:.4f}",
            "AIC":                 f"{st['AIC']:.4f}",
            "BIC":                 f"{st['BIC']:.4f}",
            "test_log_likelihood": f"{st['test_log_likelihood']:.4f}",
            "test_nll_per_obs":    f"{st['test_nll_per_obs']:.6f}",
            "seconds":             f"{st['seconds']:.2f}",
            "final_loss":          f"{st['final_loss']:.6f}",
        })

    if existing_rows or appended:
        fieldnames = (
            list(existing_rows[0].keys())
            if existing_rows
            else list(appended[0].keys())
        )
        with open(table_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in existing_rows + appended:
                w.writerow(r)
        print(f"\n✓ Appended {len(appended)} baselines to {table_path}", flush=True)

    summary_path = os.path.join(OUT_DIR, "fit_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        summary.setdefault("fits", {})
        for st in baseline_stats:
            summary["fits"][st["model"]] = {k: v for k, v in st.items() if k != "model"}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Updated {summary_path}", flush=True)

    headline_path = os.path.join(OUT_DIR, "headline_findings.json")
    if os.path.exists(headline_path):
        with open(headline_path) as f:
            headline = json.load(f)
        rows = headline.get("model_fit_table", [])
        existing = {r["model"] for r in rows}
        for st in baseline_stats:
            if st["model"] in existing:
                continue
            rows.append({
                "model": st["model"],
                "test_nll_per_obs": f"{st['test_nll_per_obs']:.6f}",
                "BIC": f"{st['BIC']:.4f}",
                "n_params": str(st["n_params"]),
            })
        headline["model_fit_table"] = rows
        with open(headline_path, "w") as f:
            json.dump(headline, f, indent=2)
        print(f"✓ Updated {headline_path}", flush=True)


def main():
    stats = compute_baselines()
    write_outputs(stats)

    print("\n──  Combined ranking (IRT + baselines, by held-out NLL/obs)  ──", flush=True)
    table_path = os.path.join(OUT_DIR, "fit_table.csv")
    if os.path.exists(table_path):
        rows = list(csv.DictReader(open(table_path)))
        rows = [r for r in rows if r.get("test_nll_per_obs")]
        rows.sort(key=lambda r: float(r["test_nll_per_obs"]))
        for r in rows:
            print(
                f"  {r['model']:<22s}  "
                f"test-NLL/obs={float(r['test_nll_per_obs']):.4f}  "
                f"BIC={float(r['BIC']):>14,.1f}  "
                f"params={int(r['n_params']):>8,}",
                flush=True,
            )


if __name__ == "__main__":
    main()
