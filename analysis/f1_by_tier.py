"""Per-tier precision / recall / F1 / accuracy and NLL-vs-F1 figures.

Why this exists
---------------
The stratified IRT summary (``outputs/irt/by_tier/summary.csv``) reports
held-out ``test_nll_per_obs`` per (tier, model).  NLL is dominated by the
majority class when prevalence is skewed -- a "predict absent" model gets
flatteringly low NLL on the ``low`` and ``na`` tiers despite never
detecting a positive.  F1 zooms in on the positive class and exposes
that failure mode.  This script computes per-(subject, tier)
TP/FP/FN/TN, derives precision/recall/F1 (micro and macro), accuracy,
and emits a comparative NLL-vs-F1 figure.

Outputs
-------
``outputs/baselines/per_prevalence_tier_prec_rec_f1.csv``
    One row per (subject, tier) with confusion-matrix counts and
    derived metrics (precision, recall, F1-micro, F1-macro, accuracy).

``outputs/irt/by_tier/metrics_by_tier.csv``
    Join of the IRT ``test_nll_per_obs`` (across all fitted models) with
    the new F1 metrics for direct side-by-side analysis.

``outputs/irt/by_tier/figures/fig_nll_vs_f1_by_tier.{pdf,png}``
    Two-panel figure: (left) NLL vs F1-macro scatter coloured by tier
    using the Rasch fit, (right) grouped bar chart of per-tier F1-macro
    for every model.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.data_loader import (  # noqa: E402
    PATHOLOGIES,
    _normalize_subject,
    discover_sources,
)
from analysis.item_metadata import PREVALENCE_TIER  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(REPO_ROOT, "outputs", "baselines")
BY_TIER_DIR = os.path.join(REPO_ROOT, "outputs", "irt", "by_tier")
FIG_DIR = os.path.join(BY_TIER_DIR, "figures")
TIERS = ["high", "mid", "low", "na"]


def _safe_div(num, denom):
    return float(num) / denom if denom else float("nan")


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 with the standard convention that an all-zero prediction
    column yields F1 = 0 (not NaN). Useful here because several VL
    models predict the negative class on every item."""
    if tp == 0 and fp == 0 and fn == 0:
        return float("nan")
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    if (p != p) or (r != r):
        return 0.0
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def collect_tier_counts() -> dict:
    """Sweep the same JSON sources baselines.py uses; bucket TP/FP/FN/TN
    into (subject, tier, pathology). Tier-level totals derive from the
    pathology-level cells (so we can compute both micro and macro F1)."""
    # cells[(subject, tier, pathology)] = {tp, fp, fn, tn}
    cells: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    def _bump(subject: str, pathology: str, y: int, yhat: int) -> None:
        tier = PREVALENCE_TIER.get(pathology, "na")
        key = (subject, tier, pathology)
        if y == 1 and yhat == 1:
            cells[key]["tp"] += 1
        elif y == 0 and yhat == 1:
            cells[key]["fp"] += 1
        elif y == 1 and yhat == 0:
            cells[key]["fn"] += 1
        else:
            cells[key]["tn"] += 1

    def _bump_dict_format(data: dict) -> None:
        for _, entry in data.items():
            if entry is None or entry.get("error") or entry.get("missing_reason"):
                continue
            subject = _normalize_subject(entry.get("deployment", "unknown"))
            preds = entry.get("predictions") or {}
            labels = entry.get("binarized_labels") or {}
            for p in PATHOLOGIES:
                y = labels.get(p)
                yhat = preds.get(p)
                if y is None or yhat is None:
                    continue
                _bump(subject, p, int(y), int(yhat))

    def _bump_list_format(data: list) -> None:
        for entry in data:
            subject = _normalize_subject(entry.get("subject", "unknown"))
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
                _bump(subject, p, y, yhat)

    for path in discover_sources(REPO_ROOT):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _bump_dict_format(data)
        else:
            _bump_list_format(data)
    return cells


def write_per_tier_metrics(cells: dict) -> str:
    """Aggregate cells to per (subject, tier) and write a CSV.

    F1-micro pools confusion-matrix counts across pathologies in the tier
    *before* the F1 calc, so it is dominated by majority-positive
    pathologies. F1-macro averages per-pathology F1 within the tier,
    giving equal weight to each constituent pathology -- the more honest
    summary when prevalence varies across the tier members.
    """
    # Aggregate
    by_st: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                       "path_f1s": []})
    # First compute per-pathology F1 cells then accumulate into tier.
    grouped: dict = defaultdict(list)  # (subject, tier) -> list of cell dicts
    for (s, t, _p), c in cells.items():
        grouped[(s, t)].append(c)
    for (s, t), cell_list in grouped.items():
        for c in cell_list:
            by_st[(s, t)]["tp"] += c["tp"]
            by_st[(s, t)]["fp"] += c["fp"]
            by_st[(s, t)]["fn"] += c["fn"]
            by_st[(s, t)]["tn"] += c["tn"]
            by_st[(s, t)]["path_f1s"].append(
                _f1_from_counts(c["tp"], c["fp"], c["fn"])
            )

    os.makedirs(BASELINES_DIR, exist_ok=True)
    out_csv = os.path.join(BASELINES_DIR, "per_prevalence_tier_prec_rec_f1.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "tier", "tp", "fp", "fn", "tn",
            "precision_micro", "recall_micro",
            "f1_micro", "f1_macro", "accuracy",
        ])
        for (s, t), agg in sorted(by_st.items()):
            tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
            prec = _safe_div(tp, tp + fp)
            rec = _safe_div(tp, tp + fn)
            f1_micro = _f1_from_counts(tp, fp, fn)
            f1s = [x for x in agg["path_f1s"] if x == x]
            f1_macro = float(np.mean(f1s)) if f1s else float("nan")
            acc = _safe_div(tp + tn, tp + fp + fn + tn)
            w.writerow([s, t, tp, fp, fn, tn,
                        f"{prec:.6f}", f"{rec:.6f}",
                        f"{f1_micro:.6f}", f"{f1_macro:.6f}", f"{acc:.6f}"])
    print(f"  ok  {os.path.relpath(out_csv, REPO_ROOT)}")
    return out_csv


def _load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _normalise_subject_for_irt(name: str) -> str:
    """The IRT summary stores model = 'rasch', 'twopl', ... so this is
    actually only needed to harmonise subjects from the F1 CSV with the
    abilities_rasch.csv subject column. We use the same passthrough as
    data_loader._normalize_subject (which the F1 CSV already used)."""
    return _normalize_subject(name)


def write_joined_metrics_csv(f1_csv: str) -> str | None:
    """Join the IRT summary (NLL/obs per (tier, model)) with per-tier F1.

    Because F1 is a property of the subject's predictions and NLL is a
    property of a fitted IRT model on those predictions, the join is
    long-format: one row per (tier, irt_model, subject) where the F1
    columns are the *subject*'s tier F1 (constant across irt_model) and
    the NLL column is the (tier, irt_model) value.
    """
    summary_path = os.path.join(BY_TIER_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print("  skip metrics_by_tier.csv: summary.csv missing")
        return None

    f1_rows = _load_csv(f1_csv)
    # (tier, subject) -> dict of F1 metrics
    f1_lookup: dict = {}
    subjects_in_f1: set = set()
    for r in f1_rows:
        key = (r["tier"], r["subject"])
        f1_lookup[key] = r
        subjects_in_f1.add(r["subject"])

    summary = _load_csv(summary_path)
    out_path = os.path.join(BY_TIER_DIR, "metrics_by_tier.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tier", "irt_model", "subject",
            "test_nll_per_obs",
            "f1_micro", "f1_macro", "precision_micro", "recall_micro",
            "accuracy",
        ])
        for row in summary:
            tier = row["tier"]
            irt_model = row["model"]
            try:
                nll = float(row["test_nll_per_obs"])
            except (KeyError, ValueError):
                continue
            for s in sorted(subjects_in_f1):
                meta = f1_lookup.get((tier, s))
                if meta is None:
                    continue
                w.writerow([
                    tier, irt_model, s, f"{nll:.6f}",
                    meta["f1_micro"], meta["f1_macro"],
                    meta["precision_micro"], meta["recall_micro"],
                    meta["accuracy"],
                ])
    print(f"  ok  {os.path.relpath(out_path, REPO_ROOT)}")
    return out_path


def _savefig(path_no_ext: str) -> None:
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig = plt.gcf()
    le = fig.get_layout_engine()
    if le is None or le.__class__.__name__ != "ConstrainedLayoutEngine":
        plt.tight_layout()
    plt.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ok  {os.path.relpath(path_no_ext, REPO_ROOT)}.{{pdf,png}}")


def figure_nll_vs_f1(f1_csv: str) -> None:
    """Three-panel comparative figure.

    Top-left: per-tier mean F1-macro (across subjects) plotted against
        the tier's Rasch NLL/obs -- exposes the inverse relationship
        (low NLL but low F1 in the rare-pathology regime).
    Top-right: per-(tier, subject) F1-macro grouped bars -- the F1
        analogue of the existing model-NLL bar chart.
    Bottom: per-(tier, subject) accuracy minus F1-macro, sorted by
        tier; large gaps signal base-rate-driven accuracy.
    """
    summary_path = os.path.join(BY_TIER_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print("  skip fig_nll_vs_f1_by_tier: summary.csv missing")
        return

    f1_rows = _load_csv(f1_csv)
    f1_lookup = {(r["tier"], r["subject"]): r for r in f1_rows}
    nll_rows = _load_csv(summary_path)
    nll_lookup: dict = {}
    for r in nll_rows:
        try:
            nll_lookup[(r["tier"], r["model"])] = float(r["test_nll_per_obs"])
        except (KeyError, ValueError):
            continue

    tiers_present = [t for t in TIERS if any(k[0] == t for k in f1_lookup)]
    subjects = sorted({k[1] for k in f1_lookup})
    tier_colours = {"high": "#1f77b4", "mid": "#ff7f0e",
                    "low": "#2ca02c", "na": "#d62728"}

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # ---- Panel A: mean F1 vs Rasch NLL across tiers (one big point per tier
    #               with per-subject scatter as small dots).
    axA = fig.add_subplot(gs[0, 0])
    for t in tiers_present:
        nll = nll_lookup.get((t, "rasch"))
        if nll is None:
            continue
        f1s = []
        for s in subjects:
            meta = f1_lookup.get((t, s))
            if meta is None:
                continue
            try:
                f1 = float(meta["f1_macro"])
            except ValueError:
                continue
            if f1 == f1:
                f1s.append(f1)
        if not f1s:
            continue
        axA.scatter(f1s, [nll] * len(f1s), s=40, color=tier_colours[t],
                    alpha=0.45, edgecolor="none", zorder=2)
        mean_f1 = float(np.mean(f1s))
        axA.scatter([mean_f1], [nll], s=320, color=tier_colours[t],
                    edgecolor="black", linewidth=1.5, marker="D",
                    label=f"{t}: mean F1={mean_f1:.2f}, NLL={nll:.3f}",
                    zorder=4)
        # iqr bars
        q1, q3 = np.percentile(f1s, [25, 75])
        axA.plot([q1, q3], [nll, nll], color=tier_colours[t],
                 linewidth=2.5, alpha=0.85, zorder=3)
    axA.set_xlabel("Per-subject F1-macro (positive-class)")
    axA.set_ylabel("Per-tier Rasch held-out NLL / obs")
    axA.set_title("Rasch NLL vs F1-macro by tier\n"
                  "(diamond = tier mean; bar = IQR; dots = subjects)")
    axA.legend(fontsize=8, loc="best")
    axA.grid(True, alpha=0.3)
    axA.invert_yaxis()  # low NLL is "better", put it at top

    # ---- Panel B: grouped bars of F1-macro per (subject, tier).
    axB = fig.add_subplot(gs[0, 1])
    n_models = len(subjects)
    x = np.arange(len(tiers_present))
    bar_width = 0.8 / max(1, n_models)
    cmap = plt.get_cmap("tab20")
    for k, s in enumerate(subjects):
        ys = []
        for t in tiers_present:
            meta = f1_lookup.get((t, s))
            try:
                v = float(meta["f1_macro"]) if meta else float("nan")
            except ValueError:
                v = float("nan")
            ys.append(v)
        offsets = x + (k - (n_models - 1) / 2) * bar_width
        axB.bar(offsets, ys, width=bar_width, label=s, color=cmap(k % 20),
                edgecolor="black", linewidth=0.3)
    axB.set_xticks(x)
    axB.set_xticklabels(tiers_present)
    axB.set_xlabel("Prevalence tier")
    axB.set_ylabel("F1-macro")
    axB.set_title("F1-macro per (subject, tier)")
    axB.legend(fontsize=7, ncols=2, loc="upper right")
    axB.grid(True, alpha=0.3, axis="y")

    # ---- Panel C: accuracy - F1-macro gap heatmap (subject x tier).
    axC = fig.add_subplot(gs[1, :])
    gap = np.full((len(subjects), len(tiers_present)), np.nan)
    for i, s in enumerate(subjects):
        for j, t in enumerate(tiers_present):
            meta = f1_lookup.get((t, s))
            if not meta:
                continue
            try:
                acc = float(meta["accuracy"])
                f1m = float(meta["f1_macro"])
            except ValueError:
                continue
            if f1m != f1m:
                f1m = 0.0
            gap[i, j] = acc - f1m
    vmax = float(np.nanmax(np.abs(gap))) if np.isfinite(gap).any() else 1.0
    im = axC.imshow(gap, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
    axC.set_xticks(range(len(tiers_present)))
    axC.set_xticklabels(tiers_present)
    axC.set_yticks(range(len(subjects)))
    axC.set_yticklabels(subjects, fontsize=8)
    for i in range(len(subjects)):
        for j in range(len(tiers_present)):
            v = gap[i, j]
            if v == v:
                axC.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="black" if v < 0.6 * vmax else "white",
                         fontsize=8)
    fig.colorbar(im, ax=axC, label="accuracy - F1-macro",
                 shrink=0.7)
    axC.set_title("Base-rate gap: accuracy minus F1-macro per (subject, tier)\n"
                  "(large red cells = high accuracy without detecting positives)")

    fig.suptitle("Comparative analysis: NLL vs F1 by prevalence tier",
                 fontsize=14, fontweight="bold")
    _savefig(os.path.join(FIG_DIR, "fig_nll_vs_f1_by_tier"))


def figure_f1_components_by_tier(f1_csv: str) -> None:
    """Per-tier (precision, recall) scatter -- additional comparative view
    that directly exposes whether low-tier "high accuracy" is
    everyone-predicts-absent (high recall on negative ~ low recall on
    positive). One panel per tier; diagonals show iso-F1 contours."""
    f1_rows = _load_csv(f1_csv)
    tiers_present = [t for t in TIERS if any(r["tier"] == t for r in f1_rows)]
    if not tiers_present:
        return

    fig, axes = plt.subplots(1, len(tiers_present),
                             figsize=(4 * len(tiers_present), 4.5),
                             sharex=True, sharey=True)
    if len(tiers_present) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    subjects = sorted({r["subject"] for r in f1_rows})
    s2colour = {s: cmap(i % 20) for i, s in enumerate(subjects)}

    pr_grid = np.linspace(0.01, 1.0, 100)
    P, R = np.meshgrid(pr_grid, pr_grid)
    F1 = 2 * P * R / (P + R)

    for ax, t in zip(axes, tiers_present):
        # Iso-F1 contours
        cs = ax.contour(P, R, F1, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                        colors="gray", linewidths=0.6, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt="F1=%.1f")
        for r in f1_rows:
            if r["tier"] != t:
                continue
            try:
                pp = float(r["precision_micro"])
                rr = float(r["recall_micro"])
            except ValueError:
                continue
            if pp != pp or rr != rr:
                continue
            ax.scatter(pp, rr, s=70, color=s2colour[r["subject"]],
                       edgecolor="black", linewidth=0.4, zorder=3)
            ax.annotate(r["subject"], (pp, rr), xytext=(4, 4),
                        textcoords="offset points", fontsize=6)
        ax.set_xlabel("Precision (micro)")
        ax.set_title(f"{t} tier")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Recall (micro)")
    fig.suptitle("Precision-recall trade-off per prevalence tier "
                 "(positive-class detection)", fontsize=12, y=1.02)
    _savefig(os.path.join(FIG_DIR, "fig_precision_recall_by_tier"))


def main() -> None:
    print("──  Counting confusion-matrix cells per (subject, tier, pathology)  ──")
    cells = collect_tier_counts()
    f1_csv = write_per_tier_metrics(cells)
    print("\n──  Joining with IRT summary  ──")
    write_joined_metrics_csv(f1_csv)
    print("\n──  Rendering comparative figures  ──")
    figure_nll_vs_f1(f1_csv)
    figure_f1_components_by_tier(f1_csv)
    print("\n+ Done.")


if __name__ == "__main__":
    main()
