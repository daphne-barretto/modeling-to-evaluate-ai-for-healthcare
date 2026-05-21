"""Generate manuscript figures.

All figures are written to ``outputs/figures/`` as PDF + PNG. Inputs are
the CSVs/JSON produced by ``analysis.fit_irt``, ``analysis.baselines``,
and ``analysis.irt_vs_baseline``.

Figures:
    fig_response_heatmap.{pdf,png}        — subject × pathology heatmap of mean accuracy
    fig_caterpillar.{pdf,png}             — IRT ability θ̂ per subject (Rasch)
    fig_icc_examples.{pdf,png}            — ICCs for a few selected items (2PL)
    fig_item_information.{pdf,png}        — test information I(θ) for Rasch and 2PL
    fig_difficulty_by_pathology.{pdf,png} — boxplot of β per pathology
    fig_discrim_by_tier.{pdf,png}         — boxplot of 2PL a per prevalence tier
    fig_factor_loading_heatmap.{pdf,png}  — 2-factor model loadings × pathology
    fig_rho_vs_accuracy.{pdf,png}         — IRT θ vs aggregate accuracy scatter
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import PATHOLOGIES, load_all  # noqa: E402
from analysis.item_metadata import PREVALENCE_TIER, ANATOMICAL_GROUP  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")


def savefig(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ outputs/figures/{name}.{{pdf,png}}")


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_response_heatmap():
    obs = load_all(REPO_ROOT)
    subjects = sorted({o.subject for o in obs})
    n_corr = defaultdict(lambda: [0, 0])
    for o in obs:
        n_corr[(o.subject, o.pathology)][0] += 1
        n_corr[(o.subject, o.pathology)][1] += o.response
    mat = np.full((len(subjects), len(PATHOLOGIES)), np.nan)
    for i, s in enumerate(subjects):
        for j, p in enumerate(PATHOLOGIES):
            n, c = n_corr.get((s, p), [0, 0])
            if n > 0:
                mat[i, j] = c / n

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(subjects) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(PATHOLOGIES)))
    ax.set_xticklabels(PATHOLOGIES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)
    for i in range(len(subjects)):
        for j in range(len(PATHOLOGIES)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.35 < v < 0.85 else "white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    ax.set_title("Per-subject × per-pathology mean accuracy")
    savefig("fig_response_heatmap")


def fig_caterpillar():
    path = os.path.join(IRT_DIR, "abilities_rasch.csv")
    if not os.path.exists(path):
        print("  skip caterpillar — rasch not fit")
        return
    rows = load_csv(path)
    rows.sort(key=lambda r: float(r["ability"]))
    names = [r["subject"] for r in rows]
    abilities = [float(r["ability"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.0))
    ax.errorbar(abilities, range(len(names)), xerr=0.2, fmt="o", color="steelblue", capsize=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Rasch ability θ̂")
    ax.set_title("Per-subject ability (Rasch)")
    savefig("fig_caterpillar")


def fig_icc_examples():
    path = os.path.join(IRT_DIR, "item_params_twopl.csv")
    if not os.path.exists(path):
        print("  skip ICC — twopl not fit")
        return
    rows = load_csv(path)
    # Pick: highest a (most discriminating) per pathology + lowest a (least)
    per_p = defaultdict(list)
    for r in rows:
        try:
            per_p[r["pathology"]].append(
                (float(r["difficulty"]), float(r["discrimination"]), r["image_path"])
            )
        except (KeyError, ValueError):
            continue

    fig, ax = plt.subplots(figsize=(9, 5))
    thetas = np.linspace(-4, 4, 200)
    # Pick 6 example items: top-discrimination items from the 6 most common pathologies
    selected = []
    for p in PATHOLOGIES[:6]:
        items = per_p.get(p, [])
        if not items:
            continue
        items.sort(key=lambda x: -x[1])
        if items:
            selected.append((p, items[0][0], items[0][1]))

    cmap = plt.get_cmap("tab10")
    for idx, (p, b, a) in enumerate(selected):
        p_correct = 1 / (1 + np.exp(-a * (thetas - b)))
        ax.plot(thetas, p_correct, label=f"{p[:18]}…  (β={b:.2f}, a={a:.2f})"
                if len(p) > 18 else f"{p}  (β={b:.2f}, a={a:.2f})",
                color=cmap(idx))
    ax.set_xlabel("θ")
    ax.set_ylabel("P(correct)")
    ax.set_title("Item characteristic curves (2PL, one high-discrimination item per pathology)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig("fig_icc_examples")


def fig_item_information():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    thetas = np.linspace(-4, 4, 400)
    # Rasch
    path_r = os.path.join(IRT_DIR, "item_params_rasch.csv")
    if os.path.exists(path_r):
        rows = load_csv(path_r)
        diffs = np.array([float(r["difficulty"]) for r in rows])
        I = np.zeros_like(thetas)
        for theta in thetas:
            p = 1 / (1 + np.exp(-(theta - diffs)))
            I[np.where(thetas == theta)[0][0]] = (p * (1 - p)).sum()
        ax.plot(thetas, I, label="Rasch (1PL)", color="C0", linewidth=2)
    # 2PL
    path_2 = os.path.join(IRT_DIR, "item_params_twopl.csv")
    if os.path.exists(path_2):
        rows = load_csv(path_2)
        diffs = np.array([float(r["difficulty"]) for r in rows])
        discrim = np.array([float(r["discrimination"]) for r in rows])
        I = np.zeros_like(thetas)
        for theta in thetas:
            p = 1 / (1 + np.exp(-discrim * (theta - diffs)))
            I[np.where(thetas == theta)[0][0]] = ((discrim ** 2) * p * (1 - p)).sum()
        ax.plot(thetas, I, label="2PL", color="C1", linewidth=2)
    ax.set_xlabel("θ")
    ax.set_ylabel("I(θ)")
    ax.set_title("Test information function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("fig_item_information")


def fig_difficulty_by_pathology():
    path = os.path.join(IRT_DIR, "item_params_rasch.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    by_p = defaultdict(list)
    for r in rows:
        try:
            by_p[r["pathology"]].append(float(r["difficulty"]))
        except (KeyError, ValueError):
            continue
    labels = [p for p in PATHOLOGIES if p in by_p]
    data = [by_p[p] for p in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Rasch difficulty β")
    ax.set_title("Item difficulty distribution per pathology (Rasch)")
    ax.grid(True, alpha=0.3)
    savefig("fig_difficulty_by_pathology")


def fig_discrim_by_tier():
    path = os.path.join(IRT_DIR, "item_params_twopl.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    by_t = defaultdict(list)
    for r in rows:
        t = PREVALENCE_TIER.get(r["pathology"], "na")
        try:
            by_t[t].append(float(r["discrimination"]))
        except (KeyError, ValueError):
            continue
    tier_order = ["high", "mid", "low", "na"]
    tier_labels = ["High prevalence\n(Easy tier)", "Mid", "Low prevalence\n(Hard tier)", "No Finding"]
    data = [by_t.get(t, []) for t in tier_order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=tier_labels, showfliers=False)
    ax.set_ylabel("2PL discrimination a")
    ax.set_title("Discrimination spread within prevalence tier\n(stratification cannot capture this within-tier variation)")
    ax.grid(True, alpha=0.3)
    savefig("fig_discrim_by_tier")


def fig_factor_loading_heatmap():
    """For the 2-factor LogisticFM: average loading by pathology and visualize."""
    path = os.path.join(IRT_DIR, "item_params_fm2.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    by_p = defaultdict(lambda: [np.zeros(2), 0])
    for r in rows:
        try:
            v = np.array([float(r["loading_V_0"]), float(r["loading_V_1"])])
            by_p[r["pathology"]][0] += v
            by_p[r["pathology"]][1] += 1
        except (KeyError, ValueError):
            continue
    mat = np.full((len(PATHOLOGIES), 2), np.nan)
    for i, p in enumerate(PATHOLOGIES):
        s, n = by_p.get(p, [np.zeros(2), 0])
        if n > 0:
            mat[i, :] = s / n
    fig, ax = plt.subplots(figsize=(5, 0.45 * len(PATHOLOGIES) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
    ax.set_yticks(range(len(PATHOLOGIES)))
    ax.set_yticklabels(
        [f"{p} [{ANATOMICAL_GROUP.get(p, 'other')[:4]}]" for p in PATHOLOGIES],
        fontsize=9,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Factor 1", "Factor 2"])
    for i in range(len(PATHOLOGIES)):
        for j in range(2):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 1.0 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean loading")
    ax.set_title("2-factor logistic FM: mean loading per pathology")
    savefig("fig_factor_loading_heatmap")


def fig_rho_vs_accuracy():
    """Scatter of subject IRT ability (Rasch) vs aggregate accuracy."""
    obs = load_all(REPO_ROOT)
    agg = defaultdict(lambda: [0, 0])
    for o in obs:
        agg[o.subject][0] += 1
        agg[o.subject][1] += o.response
    path = os.path.join(IRT_DIR, "abilities_rasch.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    pts = []
    for r in rows:
        s = r["subject"]
        n, c = agg[s]
        if n == 0:
            continue
        pts.append((s, float(r["ability"]), c / n))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.scatter(xs, ys, s=120, color="steelblue", edgecolor="black", zorder=3)
    for s, x, y in pts:
        ax.annotate(s, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Rasch ability θ̂")
    ax.set_ylabel("Aggregate accuracy")
    ax.set_title("Subject ability (IRT) vs aggregate accuracy")
    ax.grid(True, alpha=0.3)
    savefig("fig_rho_vs_accuracy")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("──  Generating figures  ──")
    fig_response_heatmap()
    fig_caterpillar()
    fig_icc_examples()
    fig_item_information()
    fig_difficulty_by_pathology()
    fig_discrim_by_tier()
    fig_factor_loading_heatmap()
    fig_rho_vs_accuracy()
    print("\n✓ All figures written to outputs/figures/")


if __name__ == "__main__":
    main()
