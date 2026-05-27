"""Augment the model-fit comparison table with held-out F1 and ROC-AUC.

For each IRT/factor model and each reference baseline (see
``analysis/baseline_nll.py``), this script:
  1. Reproduces the same 90/10 train/test split (seed=0) used by
     ``analysis/fit_irt.py``.
  2. For IRT models, reloads abilities + item-params from the CSVs in
     ``outputs/irt/`` and reconstructs ``P(correct)`` on the test set
     using each model's forward equation. No refit required.
  3. For baselines, recomputes per-test-cell probabilities from the
     training-set statistics (same logic as ``baseline_nll.py``).
  4. Reports macro-F1 (threshold = 0.5) and ROC-AUC on the held-out
     observations, and patches ``fit_table.csv``, ``fit_summary.json``
     and ``headline_findings.json`` with the new columns.

F1 and AUC are the two scalar metrics most commonly cited for diagnostic
AI evaluation (see manuscript §Intro: "Aggregate accuracy---typically
reported as AUC or F1---is the dominant metric") and are reported here
on the same held-out split as the held-out NLL/obs so that the IRT
models and the trivial baselines compare apples-to-apples.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import load_all  # noqa: E402
from analysis.fit_irt import build_index, to_tensors, train_test_split  # noqa: E402
from analysis.baseline_nll import BASELINES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
EPS = 1e-7
IRT_MODELS = ["rasch", "twopl", "threepl", "fm1", "fm2", "fm3", "bifactor"]


def _sigmoid(x):
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _load_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_abilities_aligned(model_name, subject_names):
    """Returns abilities aligned to subject_to_id ordering.

    Shape: (n_subj, K). For Rasch/2PL/3PL/Bifactor K=1; for FM-K K=K.
    """
    path = os.path.join(OUT_DIR, f"abilities_{model_name}.csv")
    rows = _load_csv_dicts(path)
    if not rows:
        raise RuntimeError(f"{path} has no rows")
    ability_cols = sorted(c for c in rows[0] if c != "subject")
    name_to_sid = {s: i for i, s in enumerate(subject_names)}
    K = len(ability_cols)
    out = np.zeros((len(subject_names), K))
    for row in rows:
        sid = name_to_sid[row["subject"]]
        out[sid] = [float(row[c]) for c in ability_cols]
    return out, ability_cols


def _load_item_params_aligned(model_name, item_keys):
    """Returns dict col_name → np.ndarray aligned to item_to_id ordering."""
    path = os.path.join(OUT_DIR, f"item_params_{model_name}.csv")
    rows = _load_csv_dicts(path)
    key_to_iid = {k: i for i, k in enumerate(item_keys)}
    skip = {"item_idx", "image_path", "pathology"}
    cols = [c for c in rows[0] if c not in skip]
    out = {col: np.zeros(len(item_keys)) for col in cols}
    for row in rows:
        iid = key_to_iid[(row["image_path"], row["pathology"])]
        for col in cols:
            out[col][iid] = float(row[col])
    return out


def _load_group_abilities(subject_names):
    """Bifactor only. Returns (n_subj, n_groups)."""
    path = os.path.join(OUT_DIR, "group_abilities_bifactor.csv")
    rows = _load_csv_dicts(path)
    group_cols = sorted(c for c in rows[0] if c != "subject")
    name_to_sid = {s: i for i, s in enumerate(subject_names)}
    out = np.zeros((len(subject_names), len(group_cols)))
    for row in rows:
        sid = name_to_sid[row["subject"]]
        out[sid] = [float(row[c]) for c in group_cols]
    return out


def predict_rasch(s_test, i_test, *, subject_names, item_keys):
    ab, _ = _load_abilities_aligned("rasch", subject_names)
    ip = _load_item_params_aligned("rasch", item_keys)
    theta = ab[s_test, 0]
    beta = ip["difficulty"][i_test]
    return _sigmoid(theta - beta)


def predict_twopl(s_test, i_test, *, subject_names, item_keys):
    ab, _ = _load_abilities_aligned("twopl", subject_names)
    ip = _load_item_params_aligned("twopl", item_keys)
    theta = ab[s_test, 0]
    beta = ip["difficulty"][i_test]
    a = ip["discrimination"][i_test]
    return _sigmoid(a * (theta - beta))


def predict_threepl(s_test, i_test, *, subject_names, item_keys):
    ab, _ = _load_abilities_aligned("threepl", subject_names)
    ip = _load_item_params_aligned("threepl", item_keys)
    theta = ab[s_test, 0]
    beta = ip["difficulty"][i_test]
    a = ip["discrimination"][i_test]
    c = ip["guessing"][i_test]
    return c + (1.0 - c) * _sigmoid(a * (theta - beta))


def predict_fm(model_name, s_test, i_test, *, subject_names, item_keys):
    ab, _ = _load_abilities_aligned(model_name, subject_names)
    ip = _load_item_params_aligned(model_name, item_keys)
    K = ab.shape[1]
    V = np.column_stack([ip[f"loading_V_{k}"] for k in range(K)])
    Z = ip["intercept_Z"]
    U_test = ab[s_test]
    V_test = V[i_test]
    logit = (U_test * V_test).sum(axis=1) + Z[i_test]
    return _sigmoid(logit)


def predict_bifactor(s_test, i_test, *, subject_names, item_keys):
    ab_general, _ = _load_abilities_aligned("bifactor", subject_names)
    ab_group = _load_group_abilities(subject_names)
    ip = _load_item_params_aligned("bifactor", item_keys)
    general_loading = ip["general_loading"][i_test]
    group_loading = ip["group_loading"][i_test]
    group_idx = ip["group_idx"][i_test].astype(np.int64)
    Z = ip["intercept_Z"][i_test]
    general = ab_general[s_test, 0] * general_loading
    group = ab_group[s_test, group_idx] * group_loading
    return _sigmoid(general + group + Z)


IRT_PREDICTORS = {
    "rasch":    lambda s, i, **kw: predict_rasch(s, i, **kw),
    "twopl":    lambda s, i, **kw: predict_twopl(s, i, **kw),
    "threepl":  lambda s, i, **kw: predict_threepl(s, i, **kw),
    "fm1":      lambda s, i, **kw: predict_fm("fm1", s, i, **kw),
    "fm2":      lambda s, i, **kw: predict_fm("fm2", s, i, **kw),
    "fm3":      lambda s, i, **kw: predict_fm("fm3", s, i, **kw),
    "bifactor": lambda s, i, **kw: predict_bifactor(s, i, **kw),
}


def _metrics(name, p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    y = y.astype(np.int64)
    pred = (p >= 0.5).astype(np.int64)
    f1_macro = f1_score(y, pred, average="macro", zero_division=0)
    f1_micro = f1_score(y, pred, average="micro", zero_division=0)
    f1_positive = f1_score(y, pred, average="binary", pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    acc = float((pred == y).mean())
    return {
        "model": name,
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1_macro),
        "test_f1_micro": float(f1_micro),
        "test_f1_positive": float(f1_positive),
        "test_auc": float(auc),
    }


def compute_metrics():
    print("──  Loading observations  ──", flush=True)
    obs = load_all(REPO_ROOT)
    subject_to_id, item_to_id, subject_names, item_keys = build_index(obs)
    s_idx, i_idx, y = to_tensors(obs, subject_to_id, item_to_id)
    s_tr, i_tr, y_tr, s_te, i_te, y_te = train_test_split(
        s_idx, i_idx, y, test_frac=0.10, seed=0,
    )
    s_train = s_tr.numpy()
    i_train = i_tr.numpy()
    y_train = y_tr.numpy()
    s_test = s_te.numpy()
    i_test = i_te.numpy()
    y_test = y_te.numpy()

    all_metrics: list[dict] = []

    print("\n──  IRT/factor models  ──", flush=True)
    for name in IRT_MODELS:
        try:
            p = IRT_PREDICTORS[name](
                s_test, i_test,
                subject_names=subject_names, item_keys=item_keys,
            )
        except FileNotFoundError as e:
            print(f"  {name}: skipped ({e})", flush=True)
            continue
        m = _metrics(name, p, y_test)
        print(
            f"  {name:<10s}  acc={m['test_accuracy']:.4f}  "
            f"F1+={m['test_f1_positive']:.4f}  "
            f"F1-macro={m['test_f1_macro']:.4f}  "
            f"AUC={m['test_auc']:.4f}",
            flush=True,
        )
        all_metrics.append(m)

    print("\n──  Reference baselines  ──", flush=True)
    for name, fn in BASELINES:
        _, p_test, _ = fn(
            y_train, s_train, i_train, item_keys, s_test, i_test
        )
        m = _metrics(name, p_test, y_test)
        print(
            f"  {name:<22s}  acc={m['test_accuracy']:.4f}  "
            f"F1+={m['test_f1_positive']:.4f}  "
            f"F1-macro={m['test_f1_macro']:.4f}  "
            f"AUC={m['test_auc']:.4f}",
            flush=True,
        )
        all_metrics.append(m)

    return all_metrics


METRIC_COLS = ("test_accuracy", "test_f1_macro", "test_f1_micro",
               "test_f1_positive", "test_auc")


def write_outputs(metrics: list[dict]):
    table_path = os.path.join(OUT_DIR, "fit_table.csv")
    rows = _load_csv_dicts(table_path)
    by_model = {r["model"]: r for r in rows}
    by_metric = {m["model"]: m for m in metrics}

    for r in rows:
        m = by_metric.get(r["model"])
        if not m:
            continue
        for col in METRIC_COLS:
            r[col] = f"{m[col]:.6f}"

    fieldnames = list(rows[0].keys())
    for col in METRIC_COLS:
        if col not in fieldnames:
            fieldnames.append(col)

    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for c in fieldnames:
                r.setdefault(c, "")
            w.writerow(r)
    print(f"\n✓ Patched {table_path}", flush=True)

    summary_path = os.path.join(OUT_DIR, "fit_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        summary.setdefault("fits", {})
        for m in metrics:
            entry = summary["fits"].setdefault(m["model"], {})
            for c in METRIC_COLS:
                entry[c] = m[c]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Updated {summary_path}", flush=True)

    headline_path = os.path.join(OUT_DIR, "headline_findings.json")
    if os.path.exists(headline_path):
        with open(headline_path) as f:
            headline = json.load(f)
        rows_h = headline.get("model_fit_table", [])
        by_m = {r["model"]: r for r in rows_h}
        for m in metrics:
            r = by_m.get(m["model"])
            if r is None:
                continue
            for c in METRIC_COLS:
                r[c] = f"{m[c]:.6f}"
        headline["model_fit_table"] = rows_h
        with open(headline_path, "w") as f:
            json.dump(headline, f, indent=2)
        print(f"✓ Updated {headline_path}", flush=True)


def main():
    metrics = compute_metrics()
    write_outputs(metrics)

    print(
        "\n──  Combined ranking by held-out F1 (positive class) and AUC  ──",
        flush=True,
    )
    by_model = {m["model"]: m for m in metrics}
    ordered = sorted(
        metrics, key=lambda m: -m["test_f1_positive"]
    )
    for m in ordered:
        print(
            f"  {m['model']:<22s}  "
            f"F1+={m['test_f1_positive']:.4f}  "
            f"F1-macro={m['test_f1_macro']:.4f}  "
            f"AUC={m['test_auc']:.4f}  "
            f"acc={m['test_accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
