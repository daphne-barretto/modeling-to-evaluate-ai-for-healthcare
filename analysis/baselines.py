"""Non-IRT baselines for comparison with measurement models.

Addresses peer-review feedback: "Move beyond mean aggregate accuracy as
the baseline and also compare metadata stratification and other more
advanced baselines."

Produces:
    outputs/baselines/aggregate_accuracy.csv
    outputs/baselines/per_pathology_accuracy.csv
    outputs/baselines/per_view_accuracy.csv
    outputs/baselines/per_anatomical_group_accuracy.csv
    outputs/baselines/per_prevalence_tier_accuracy.csv
    outputs/baselines/per_sex_accuracy.csv
    outputs/baselines/per_age_bin_accuracy.csv
    outputs/baselines/per_pathology_precision_recall_f1.csv
    outputs/baselines/baseline_summary.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import PATHOLOGIES, load_all  # noqa: E402
from analysis.item_metadata import (  # noqa: E402
    ANATOMICAL_GROUP, PREVALENCE_TIER, get_image_meta,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "baselines")


def _safe_div(num, denom):
    return float(num) / denom if denom else float("nan")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def aggregate_accuracy(observations):
    correct = defaultdict(int)
    total = defaultdict(int)
    for o in observations:
        total[o.subject] += 1
        correct[o.subject] += o.response
    rows = [(s, total[s], correct[s], _safe_div(correct[s], total[s])) for s in sorted(total)]
    write_csv(
        os.path.join(OUT_DIR, "aggregate_accuracy.csv"),
        ["subject", "n_observations", "n_correct", "accuracy"],
        [(s, n, c, f"{a:.6f}") for (s, n, c, a) in rows],
    )
    return {s: a for (s, _, _, a) in rows}


def per_pathology(observations):
    cells = defaultdict(lambda: [0, 0])  # (subject, pathology) -> [n, correct]
    for o in observations:
        cells[(o.subject, o.pathology)][0] += 1
        cells[(o.subject, o.pathology)][1] += o.response
    subjects = sorted({s for (s, _) in cells})
    rows = []
    for s in subjects:
        for p in PATHOLOGIES:
            n, c = cells.get((s, p), [0, 0])
            rows.append((s, p, n, c, f"{_safe_div(c, n):.6f}"))
    write_csv(
        os.path.join(OUT_DIR, "per_pathology_accuracy.csv"),
        ["subject", "pathology", "n", "n_correct", "accuracy"],
        rows,
    )


def per_anatomy_and_tier(observations):
    by_anat = defaultdict(lambda: [0, 0])
    by_tier = defaultdict(lambda: [0, 0])
    for o in observations:
        a = ANATOMICAL_GROUP.get(o.pathology, "other")
        t = PREVALENCE_TIER.get(o.pathology, "na")
        by_anat[(o.subject, a)][0] += 1
        by_anat[(o.subject, a)][1] += o.response
        by_tier[(o.subject, t)][0] += 1
        by_tier[(o.subject, t)][1] += o.response
    write_csv(
        os.path.join(OUT_DIR, "per_anatomical_group_accuracy.csv"),
        ["subject", "anatomical_group", "n", "n_correct", "accuracy"],
        [(s, a, n, c, f"{_safe_div(c, n):.6f}") for (s, a), (n, c) in sorted(by_anat.items())],
    )
    write_csv(
        os.path.join(OUT_DIR, "per_prevalence_tier_accuracy.csv"),
        ["subject", "prevalence_tier", "n", "n_correct", "accuracy"],
        [(s, t, n, c, f"{_safe_div(c, n):.6f}") for (s, t), (n, c) in sorted(by_tier.items())],
    )


def per_metadata_axis(observations, axis_name, key_fn, out_basename):
    bucket = defaultdict(lambda: [0, 0])
    unresolved = 0
    for o in observations:
        meta = get_image_meta(o.image_path, REPO_ROOT)
        if meta is None:
            unresolved += 1
            continue
        k = key_fn(meta)
        bucket[(o.subject, k)][0] += 1
        bucket[(o.subject, k)][1] += o.response
    rows = [
        (s, k, n, c, f"{_safe_div(c, n):.6f}")
        for (s, k), (n, c) in sorted(bucket.items())
    ]
    write_csv(
        os.path.join(OUT_DIR, out_basename),
        ["subject", axis_name, "n", "n_correct", "accuracy"],
        rows,
    )
    return unresolved


def per_pathology_prec_rec_f1(observations_raw):
    """Compute precision / recall / F1 per (subject, pathology) using the
    raw per-pathology answer fields.

    ``observations_raw`` is the unfiltered loader output; we re-read the
    raw JSON files because per-pathology answer/gt is needed (not just the
    binary correct field).
    """
    # Re-read sources at the raw JSON level to retrieve per-pathology answer + gt.
    import glob
    rows = []
    summary = {}
    daphne_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", "daphne_*.json")))
    izhan_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "qwen*_results_*.json")))

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for path in daphne_paths:
        with open(path) as f:
            data = json.load(f)
        for image_path, entry in data.items():
            if entry is None or entry.get("error") or entry.get("missing_reason"):
                continue
            subject = entry.get("deployment", "unknown")
            preds = entry.get("predictions") or {}
            labels = entry.get("binarized_labels") or {}
            for p in PATHOLOGIES:
                y = labels.get(p)
                yhat = preds.get(p)
                if y is None or yhat is None:
                    continue
                key = (subject, p)
                if y == 1 and yhat == 1:  counts[key]["tp"] += 1
                elif y == 0 and yhat == 1: counts[key]["fp"] += 1
                elif y == 1 and yhat == 0: counts[key]["fn"] += 1
                else:                       counts[key]["tn"] += 1

    for path in izhan_paths:
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            subject = entry.get("subject", "unknown")
            for p in PATHOLOGIES:
                y_raw = entry.get(f"{p}__gt")
                ans = entry.get(f"{p}__answer")
                if y_raw is None or ans is None or ans == "unclear":
                    continue
                try:
                    y = int(y_raw)
                except (TypeError, ValueError):
                    continue
                yhat = 1 if ans == "yes" else 0
                key = (subject, p)
                if y == 1 and yhat == 1:  counts[key]["tp"] += 1
                elif y == 0 and yhat == 1: counts[key]["fp"] += 1
                elif y == 1 and yhat == 0: counts[key]["fn"] += 1
                else:                       counts[key]["tn"] += 1

    for (s, p), c in sorted(counts.items()):
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if prec + rec else float("nan")
        rows.append((s, p, tp, fp, fn, tn,
                     f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}"))
    write_csv(
        os.path.join(OUT_DIR, "per_pathology_precision_recall_f1.csv"),
        ["subject", "pathology", "tp", "fp", "fn", "tn", "precision", "recall", "f1"],
        rows,
    )
    summary["n_subject_pathology_pairs"] = len(rows)
    return summary


def main():
    print("──  Loading observations  ──")
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations available.")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n──  Baseline 1: aggregate accuracy  ──")
    agg = aggregate_accuracy(obs)
    for s, a in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {s:<28s} {a:.4f}")

    print("\n──  Baseline 2: per-pathology accuracy  ──")
    per_pathology(obs)
    print(f"  → outputs/baselines/per_pathology_accuracy.csv")

    print("\n──  Baseline 3: per-anatomy + per-prevalence-tier  ──")
    per_anatomy_and_tier(obs)
    print(f"  → per_anatomical_group_accuracy.csv, per_prevalence_tier_accuracy.csv")

    print("\n──  Baseline 4: per-view (Frontal/Lateral)  ──")
    miss_v = per_metadata_axis(obs, "view", lambda m: m.view, "per_view_accuracy.csv")
    print(f"  Frontal/Lateral; unresolved images (not in train CSV): {miss_v:,}")

    print("\n──  Baseline 5: per-AP/PA  ──")
    miss_ap = per_metadata_axis(obs, "ap_pa", lambda m: m.ap_pa, "per_ap_pa_accuracy.csv")
    print(f"  AP/PA; unresolved: {miss_ap:,}")

    print("\n──  Baseline 6: per-sex (DIF candidate)  ──")
    miss_s = per_metadata_axis(obs, "sex", lambda m: m.sex, "per_sex_accuracy.csv")
    print(f"  unresolved: {miss_s:,}")

    print("\n──  Baseline 7: per-age-bin (DIF candidate)  ──")
    miss_a = per_metadata_axis(obs, "age_bin", lambda m: m.age_bin, "per_age_bin_accuracy.csv")
    print(f"  unresolved: {miss_a:,}")

    print("\n──  Baseline 8: per-pathology precision/recall/F1  ──")
    pf_summary = per_pathology_prec_rec_f1(obs)
    print(f"  rows written: {pf_summary['n_subject_pathology_pairs']}")

    summary = {
        "aggregate_accuracy": agg,
        "files": sorted(os.listdir(OUT_DIR)),
        "n_observations": len(obs),
        "n_unresolved_train_csv": {
            "view": miss_v, "ap_pa": miss_ap, "sex": miss_s, "age_bin": miss_a,
        },
    }
    with open(os.path.join(OUT_DIR, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Wrote summary: outputs/baselines/baseline_summary.json")


if __name__ == "__main__":
    main()
