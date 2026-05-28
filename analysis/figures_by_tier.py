"""Generate per-tier and cross-tier figures for the stratified IRT fits.

Mirror of ``analysis/figures.py`` but operating on the stratified outputs in
``outputs/irt/by_tier/{tier}/``. Per-tier figures land in
``outputs/irt/by_tier/{tier}/figures/`` (PDF + PNG); cross-tier figures land
in ``outputs/irt/by_tier/figures/``.

Per-tier figures (when applicable for that tier):
    fig_response_heatmap                 subject x pathology mean accuracy
    fig_caterpillar                      Rasch ability per subject
    fig_icc_examples                     2PL ICCs (one high-discrim item per
                                         tier-pathology, up to 6)
    fig_item_information                 test information I(theta) for Rasch + 2PL
    fig_difficulty_by_pathology          Rasch beta boxplot (or histogram for na)
    fig_factor_loading_heatmap           fm2 mean loadings per tier-pathology
    fig_rho_vs_accuracy                  Rasch theta vs tier aggregate accuracy

Cross-tier figures:
    fig_ability_across_tiers             subject x tier Rasch ability heatmap
    fig_model_comparison_by_tier         held-out NLL/obs grouped bar chart
    fig_discrim_by_tier                  2PL discrimination boxplot per tier

Combined-panel figures (cross-tier composites):
    fig_panel_caterpillar_by_tier        1x4 Rasch caterpillars, shared subjects
    fig_panel_difficulty_by_tier         4 Rasch-beta violins, shared y-axis
    fig_panel_response_heatmaps          2x2 per-tier accuracy heatmaps
    fig_panel_summary                    2x2 manuscript-candidate composite

Additional comparative-analysis figures:
    fig_ability_profile                  parallel-coords subject x tier theta_hat
    fig_difficulty_kde_overlay           4 KDE curves of Rasch beta on one axes
    fig_info_curves_overlay              4 Rasch test-information curves overlaid
    fig_pairwise_tier_ability            2x3 grid of pairwise (theta_A, theta_B)
    fig_pooled_vs_stratified_caterpillar overlays pooled Rasch theta on per-tier

Inputs are the CSVs/JSON already produced by ``analysis.fit_irt_by_tier``
and the partitioned observations in ``data/observations_by_tier/``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.item_metadata import PREVALENCE_TIER  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BY_TIER_DIR = os.path.join(REPO_ROOT, "outputs", "irt", "by_tier")
OBS_DIR = os.path.join(REPO_ROOT, "data", "observations_by_tier")
TIERS = ["high", "mid", "low", "na"]

# Pathologies per tier, derived from PREVALENCE_TIER (pooled item_metadata).
TIER_PATHOLOGIES: dict[str, list[str]] = {t: [] for t in TIERS}
for p, t in PREVALENCE_TIER.items():
    if t in TIER_PATHOLOGIES:
        TIER_PATHOLOGIES[t].append(p)
for t in TIERS:
    TIER_PATHOLOGIES[t].sort()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def savefig(out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=180)
    plt.close()
    rel = os.path.relpath(os.path.join(out_dir, name), REPO_ROOT).replace("\\", "/")
    print(f"  ok  {rel}.{{pdf,png}}")


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_tier_observations(tier: str) -> list[dict]:
    """Stream the tier-partitioned JSONL into memory as a list of dicts."""
    path = os.path.join(OBS_DIR, f"{tier}.jsonl")
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tier_paths(tier: str) -> dict[str, str]:
    base = os.path.join(BY_TIER_DIR, tier)
    return {
        "base": base,
        "fig_dir": os.path.join(base, "figures"),
        "abilities_rasch": os.path.join(base, "abilities_rasch.csv"),
        "item_params_rasch": os.path.join(base, "item_params_rasch.csv"),
        "item_params_twopl": os.path.join(base, "item_params_twopl.csv"),
        "item_params_fm2": os.path.join(base, "item_params_fm2.csv"),
    }


# ---------------------------------------------------------------------------
# Per-tier figures
# ---------------------------------------------------------------------------


def fig_response_heatmap(tier: str, paths: dict[str, str]) -> None:
    obs = load_tier_observations(tier)
    if not obs:
        print(f"  skip {tier}/fig_response_heatmap: no observations")
        return
    pathologies = TIER_PATHOLOGIES[tier]
    subjects = sorted({o["subject"] for o in obs})
    n_corr: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for o in obs:
        key = (o["subject"], o["pathology"])
        n_corr[key][0] += 1
        n_corr[key][1] += int(o["response"])

    mat = np.full((len(subjects), len(pathologies)), np.nan)
    for i, s in enumerate(subjects):
        for j, p in enumerate(pathologies):
            n, c = n_corr.get((s, p), [0, 0])
            if n > 0:
                mat[i, j] = c / n

    width = max(4.0, 1.6 * len(pathologies) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 0.55 * len(subjects) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pathologies)))
    ax.set_xticklabels(pathologies, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)
    for i in range(len(subjects)):
        for j in range(len(pathologies)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.35 < v < 0.85 else "white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    ax.set_title(f"Per-subject x per-pathology mean accuracy ({tier} tier)")
    savefig(paths["fig_dir"], "fig_response_heatmap")


def fig_caterpillar(tier: str, paths: dict[str, str]) -> None:
    if not os.path.exists(paths["abilities_rasch"]):
        print(f"  skip {tier}/fig_caterpillar: rasch abilities missing")
        return
    rows = load_csv(paths["abilities_rasch"])
    rows.sort(key=lambda r: float(r["ability"]))
    names = [r["subject"] for r in rows]
    abilities = [float(r["ability"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.0))
    ax.errorbar(abilities, range(len(names)), xerr=0.2, fmt="o",
                color="steelblue", capsize=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Rasch ability $\\hat{\\theta}$")
    ax.set_title(f"Per-subject ability (Rasch, {tier} tier)")
    savefig(paths["fig_dir"], "fig_caterpillar")


def fig_icc_examples(tier: str, paths: dict[str, str]) -> None:
    if not os.path.exists(paths["item_params_twopl"]):
        print(f"  skip {tier}/fig_icc_examples: twopl item params missing")
        return
    rows = load_csv(paths["item_params_twopl"])
    per_p: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for r in rows:
        try:
            per_p[r["pathology"]].append(
                (float(r["difficulty"]), float(r["discrimination"]), r["image_path"])
            )
        except (KeyError, ValueError):
            continue

    pathologies = [p for p in TIER_PATHOLOGIES[tier] if p in per_p]
    if not pathologies:
        print(f"  skip {tier}/fig_icc_examples: no twopl items match tier pathologies")
        return
    pick = pathologies[: min(6, len(pathologies))]

    fig, ax = plt.subplots(figsize=(9, 5))
    thetas = np.linspace(-4, 4, 200)
    cmap = plt.get_cmap("tab10")
    for idx, p in enumerate(pick):
        items = per_p[p]
        items.sort(key=lambda x: -x[1])  # most discriminating first
        b, a, _ = items[0]
        p_correct = 1 / (1 + np.exp(-a * (thetas - b)))
        label = (f"{p[:18]}...  (b={b:.2f}, a={a:.2f})"
                 if len(p) > 18 else f"{p}  (b={b:.2f}, a={a:.2f})")
        ax.plot(thetas, p_correct, label=label, color=cmap(idx))
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Item characteristic curves (2PL, top-discrim item per "
                 f"pathology, {tier} tier)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(paths["fig_dir"], "fig_icc_examples")


def fig_item_information(tier: str, paths: dict[str, str]) -> None:
    have_r = os.path.exists(paths["item_params_rasch"])
    have_2 = os.path.exists(paths["item_params_twopl"])
    if not (have_r or have_2):
        print(f"  skip {tier}/fig_item_information: no item params")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    thetas = np.linspace(-4, 4, 400)

    if have_r:
        rows = load_csv(paths["item_params_rasch"])
        diffs = np.array([float(r["difficulty"]) for r in rows])
        # vectorise: I(theta) = sum_i p_i(theta) * (1 - p_i(theta))
        # P matrix: (n_theta, n_items)
        P = 1.0 / (1.0 + np.exp(-(thetas[:, None] - diffs[None, :])))
        I = (P * (1.0 - P)).sum(axis=1)
        ax.plot(thetas, I, label="Rasch (1PL)", color="C0", linewidth=2)

    if have_2:
        rows = load_csv(paths["item_params_twopl"])
        diffs = np.array([float(r["difficulty"]) for r in rows])
        discrim = np.array([float(r["discrimination"]) for r in rows])
        P = 1.0 / (1.0 + np.exp(-discrim[None, :] * (thetas[:, None] - diffs[None, :])))
        I = ((discrim[None, :] ** 2) * P * (1.0 - P)).sum(axis=1)
        ax.plot(thetas, I, label="2PL", color="C1", linewidth=2)

    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$I(\\theta)$")
    ax.set_title(f"Test information function ({tier} tier)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(paths["fig_dir"], "fig_item_information")


def fig_difficulty_by_pathology(tier: str, paths: dict[str, str]) -> None:
    if not os.path.exists(paths["item_params_rasch"]):
        print(f"  skip {tier}/fig_difficulty_by_pathology: rasch items missing")
        return
    rows = load_csv(paths["item_params_rasch"])
    by_p: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            by_p[r["pathology"]].append(float(r["difficulty"]))
        except (KeyError, ValueError):
            continue
    labels = [p for p in TIER_PATHOLOGIES[tier] if p in by_p]
    if not labels:
        print(f"  skip {tier}/fig_difficulty_by_pathology: no items for tier")
        return

    # For na (1 pathology) a single-box boxplot is uninformative -> histogram.
    if len(labels) == 1:
        vals = by_p[labels[0]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(vals, bins=40, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(float(np.median(vals)), color="firebrick", linestyle="--",
                   linewidth=1.2, label=f"median = {float(np.median(vals)):.2f}")
        ax.set_xlabel(r"Rasch difficulty $\beta$")
        ax.set_ylabel("Item count")
        ax.set_title(f"Item difficulty distribution ({tier} tier, {labels[0]})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(paths["fig_dir"], "fig_difficulty_by_pathology")
        return

    data = [by_p[p] for p in labels]
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels) + 2), 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"Rasch difficulty $\beta$")
    ax.set_title(f"Item difficulty distribution per pathology "
                 f"(Rasch, {tier} tier)")
    ax.grid(True, alpha=0.3)
    savefig(paths["fig_dir"], "fig_difficulty_by_pathology")


def fig_factor_loading_heatmap(tier: str, paths: dict[str, str]) -> None:
    if not os.path.exists(paths["item_params_fm2"]):
        print(f"  skip {tier}/fig_factor_loading_heatmap: fm2 items missing")
        return
    rows = load_csv(paths["item_params_fm2"])
    by_p: dict[str, tuple[np.ndarray, int]] = {}
    for r in rows:
        try:
            v = np.array([float(r["loading_V_0"]), float(r["loading_V_1"])])
        except (KeyError, ValueError):
            continue
        s, n = by_p.get(r["pathology"], (np.zeros(2), 0))
        by_p[r["pathology"]] = (s + v, n + 1)
    pathologies = [p for p in TIER_PATHOLOGIES[tier] if p in by_p]
    if not pathologies:
        print(f"  skip {tier}/fig_factor_loading_heatmap: no fm2 items match tier")
        return
    mat = np.full((len(pathologies), 2), np.nan)
    for i, p in enumerate(pathologies):
        s, n = by_p[p]
        if n > 0:
            mat[i, :] = s / n

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
    fig, ax = plt.subplots(figsize=(5, 0.55 * len(pathologies) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pathologies)))
    ax.set_yticklabels(pathologies, fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Factor 1", "Factor 2"])
    for i in range(len(pathologies)):
        for j in range(2):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.7 * vmax else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean loading")
    ax.set_title(f"2-factor logistic FM: mean loading per pathology "
                 f"({tier} tier)")
    savefig(paths["fig_dir"], "fig_factor_loading_heatmap")


def fig_rho_vs_accuracy(tier: str, paths: dict[str, str]) -> None:
    if not os.path.exists(paths["abilities_rasch"]):
        print(f"  skip {tier}/fig_rho_vs_accuracy: rasch abilities missing")
        return
    obs = load_tier_observations(tier)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for o in obs:
        agg[o["subject"]][0] += 1
        agg[o["subject"]][1] += int(o["response"])
    rows = load_csv(paths["abilities_rasch"])
    pts: list[tuple[str, float, float]] = []
    for r in rows:
        n, c = agg.get(r["subject"], [0, 0])
        if n == 0:
            continue
        pts.append((r["subject"], float(r["ability"]), c / n))
    if not pts:
        print(f"  skip {tier}/fig_rho_vs_accuracy: no matching subjects")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.scatter(xs, ys, s=120, color="steelblue", edgecolor="black", zorder=3)
    for s, x, y in pts:
        ax.annotate(s, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"Rasch ability $\hat{\theta}$")
    ax.set_ylabel("Aggregate accuracy")
    ax.set_title(f"Subject ability (IRT) vs aggregate accuracy ({tier} tier)")
    ax.grid(True, alpha=0.3)
    savefig(paths["fig_dir"], "fig_rho_vs_accuracy")


# ---------------------------------------------------------------------------
# Cross-tier figures
# ---------------------------------------------------------------------------


def _cross_tier_dir() -> str:
    return os.path.join(BY_TIER_DIR, "figures")


def fig_ability_across_tiers() -> None:
    """Subject x tier heatmap of Rasch theta_hat."""
    tier_to_abilities: dict[str, dict[str, float]] = {}
    subjects: set[str] = set()
    for t in TIERS:
        path = os.path.join(BY_TIER_DIR, t, "abilities_rasch.csv")
        if not os.path.exists(path):
            continue
        a = {r["subject"]: float(r["ability"]) for r in load_csv(path)}
        tier_to_abilities[t] = a
        subjects.update(a.keys())
    if not tier_to_abilities:
        print("  skip fig_ability_across_tiers: no per-tier Rasch fits")
        return

    tiers_present = [t for t in TIERS if t in tier_to_abilities]
    # Sort subjects by mean ability across the tiers we have.
    def mean_for(s: str) -> float:
        vals = [tier_to_abilities[t][s] for t in tiers_present
                if s in tier_to_abilities[t]]
        return float(np.mean(vals)) if vals else 0.0

    subj_sorted = sorted(subjects, key=mean_for)
    mat = np.full((len(subj_sorted), len(tiers_present)), np.nan)
    for i, s in enumerate(subj_sorted):
        for j, t in enumerate(tiers_present):
            v = tier_to_abilities[t].get(s)
            if v is not None:
                mat[i, j] = v

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
    fig, ax = plt.subplots(figsize=(2.0 + 1.2 * len(tiers_present),
                                     0.45 * len(subj_sorted) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tiers_present)))
    ax.set_xticklabels([t for t in tiers_present], fontsize=10)
    ax.set_yticks(range(len(subj_sorted)))
    ax.set_yticklabels(subj_sorted, fontsize=9)
    for i in range(len(subj_sorted)):
        for j in range(len(tiers_present)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.7 * vmax else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label=r"Rasch $\hat{\theta}$")
    ax.set_xlabel("Prevalence tier")
    ax.set_title("Subject ability across prevalence tiers (Rasch per tier)")
    savefig(_cross_tier_dir(), "fig_ability_across_tiers")


def fig_model_comparison_by_tier() -> None:
    """Grouped bar chart of held-out NLL/obs per (tier, model)."""
    summary_path = os.path.join(BY_TIER_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print("  skip fig_model_comparison_by_tier: summary.csv missing")
        return
    rows = load_csv(summary_path)
    # Preserve model order from summary, deduplicated.
    models: list[str] = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    by_tm: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            by_tm[(r["tier"], r["model"])] = float(r["test_nll_per_obs"])
        except (KeyError, ValueError):
            continue
    tiers_present = [t for t in TIERS if any((t, m) in by_tm for m in models)]
    if not tiers_present:
        print("  skip fig_model_comparison_by_tier: no NLL values")
        return

    n_models = len(models)
    x = np.arange(len(tiers_present))
    width = 0.8 / n_models
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(tiers_present) * n_models / 3), 5))
    for k, m in enumerate(models):
        ys = [by_tm.get((t, m), np.nan) for t in tiers_present]
        offsets = x + (k - (n_models - 1) / 2) * width
        ax.bar(offsets, ys, width=width, label=m, color=cmap(k % 10),
               edgecolor="black", linewidth=0.4)
    # Mark the winner per tier.
    for j, t in enumerate(tiers_present):
        best_m = min(models, key=lambda m: by_tm.get((t, m), float("inf")))
        if (t, best_m) in by_tm:
            best_k = models.index(best_m)
            best_off = j + (best_k - (n_models - 1) / 2) * width
            ax.annotate("best", (best_off, by_tm[(t, best_m)]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_present)
    ax.set_ylabel("Held-out NLL / obs (lower is better)")
    ax.set_xlabel("Prevalence tier")
    ax.set_title("Model comparison by prevalence tier")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.grid(True, alpha=0.3, axis="y")
    savefig(_cross_tier_dir(), "fig_model_comparison_by_tier")


def fig_discrim_by_tier_cross() -> None:
    """Boxplot of 2PL discrimination a per tier (across tier-local item params)."""
    by_t: dict[str, list[float]] = defaultdict(list)
    for t in TIERS:
        path = os.path.join(BY_TIER_DIR, t, "item_params_twopl.csv")
        if not os.path.exists(path):
            continue
        for r in load_csv(path):
            try:
                by_t[t].append(float(r["discrimination"]))
            except (KeyError, ValueError):
                continue
    tiers_present = [t for t in TIERS if by_t.get(t)]
    if not tiers_present:
        print("  skip fig_discrim_by_tier: no twopl items in any tier")
        return
    labels_map = {
        "high": "High prevalence\n(Easy tier)",
        "mid": "Mid",
        "low": "Low prevalence\n(Hard tier)",
        "na": "No Finding",
    }
    data = [by_t[t] for t in tiers_present]
    labels = [labels_map.get(t, t) for t in tiers_present]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("2PL discrimination $a$")
    ax.set_title("2PL discrimination per prevalence tier\n"
                 "(stratified item-pool variation)")
    ax.grid(True, alpha=0.3)
    savefig(_cross_tier_dir(), "fig_discrim_by_tier")


# ---------------------------------------------------------------------------
# Combined-panel and comparative figures
# ---------------------------------------------------------------------------


def _present_tiers_with_rasch() -> list[str]:
    return [
        t for t in TIERS
        if os.path.exists(os.path.join(BY_TIER_DIR, t, "abilities_rasch.csv"))
    ]


def _ability_table() -> tuple[list[str], list[str], np.ndarray]:
    """Build (tiers, subjects, mat) with mat[i, j] = theta_hat for
    subject i in tier j. NaN where missing. Subjects sorted by across-tier
    mean (ascending)."""
    tiers_present = _present_tiers_with_rasch()
    if not tiers_present:
        return [], [], np.zeros((0, 0))
    per_tier: dict[str, dict[str, float]] = {}
    subjects: set[str] = set()
    for t in tiers_present:
        a = {r["subject"]: float(r["ability"])
             for r in load_csv(os.path.join(BY_TIER_DIR, t, "abilities_rasch.csv"))}
        per_tier[t] = a
        subjects.update(a.keys())

    def mean_for(s: str) -> float:
        vals = [per_tier[t][s] for t in tiers_present if s in per_tier[t]]
        return float(np.mean(vals)) if vals else 0.0

    subj_sorted = sorted(subjects, key=mean_for)
    mat = np.full((len(subj_sorted), len(tiers_present)), np.nan)
    for i, s in enumerate(subj_sorted):
        for j, t in enumerate(tiers_present):
            v = per_tier[t].get(s)
            if v is not None:
                mat[i, j] = v
    return tiers_present, subj_sorted, mat


def fig_panel_caterpillar_by_tier() -> None:
    """1x4 grid of Rasch caterpillars with shared subject ordering + x-axis."""
    tiers_present, subjects, mat = _ability_table()
    if not tiers_present:
        print("  skip fig_panel_caterpillar_by_tier: no per-tier Rasch fits")
        return

    fig, axes = plt.subplots(
        1, len(tiers_present),
        figsize=(2.6 * len(tiers_present) + 1.5, 0.4 * len(subjects) + 1.5),
        sharey=True,
    )
    if len(tiers_present) == 1:
        axes = [axes]
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    pad = 0.1 * max(1e-6, vmax - vmin)
    for j, (ax, t) in enumerate(zip(axes, tiers_present)):
        ys = np.arange(len(subjects))
        xs = mat[:, j]
        mask = ~np.isnan(xs)
        ax.errorbar(xs[mask], ys[mask], xerr=0.2, fmt="o",
                    color="steelblue", capsize=3, markersize=5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(f"{t}")
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.grid(True, alpha=0.3, axis="x")
        if j == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels(subjects, fontsize=9)
        ax.set_xlabel(r"Rasch $\hat{\theta}$")
    fig.suptitle("Per-subject Rasch ability across prevalence tiers "
                 "(shared subject ordering)", y=1.02)
    fig.tight_layout()
    savefig(_cross_tier_dir(), "fig_panel_caterpillar_by_tier")


def fig_panel_difficulty_by_tier() -> None:
    """4 violins of Rasch beta, one per tier, shared y-axis."""
    by_t: dict[str, np.ndarray] = {}
    for t in TIERS:
        path = os.path.join(BY_TIER_DIR, t, "item_params_rasch.csv")
        if not os.path.exists(path):
            continue
        vals: list[float] = []
        for r in load_csv(path):
            try:
                vals.append(float(r["difficulty"]))
            except (KeyError, ValueError):
                continue
        if vals:
            by_t[t] = np.asarray(vals, dtype=float)
    tiers_present = [t for t in TIERS if t in by_t]
    if not tiers_present:
        print("  skip fig_panel_difficulty_by_tier: no Rasch items")
        return

    data = [by_t[t] for t in tiers_present]
    fig, ax = plt.subplots(figsize=(2.0 * len(tiers_present) + 2.0, 5.0))
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(tiers_present) + 1))
    ax.set_xticklabels(tiers_present)
    ax.set_xlabel("Prevalence tier")
    ax.set_ylabel(r"Rasch difficulty $\beta$")
    ax.set_title("Item difficulty distribution per prevalence tier (Rasch)")
    # Annotate median + N
    for i, t in enumerate(tiers_present, start=1):
        med = float(np.median(by_t[t]))
        ax.annotate(f"n={len(by_t[t]):,}\nmed={med:+.2f}", (i, med),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.7, edgecolor="gray"))
    ax.grid(True, alpha=0.3, axis="y")
    savefig(_cross_tier_dir(), "fig_panel_difficulty_by_tier")


def _accuracy_matrix(tier: str, subjects_order: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (mat, pathologies) of mean accuracy for the given subject ordering."""
    obs = load_tier_observations(tier)
    pathologies = TIER_PATHOLOGIES[tier]
    n_corr: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for o in obs:
        n_corr[(o["subject"], o["pathology"])][0] += 1
        n_corr[(o["subject"], o["pathology"])][1] += int(o["response"])
    mat = np.full((len(subjects_order), len(pathologies)), np.nan)
    for i, s in enumerate(subjects_order):
        for j, p in enumerate(pathologies):
            n, c = n_corr.get((s, p), [0, 0])
            if n > 0:
                mat[i, j] = c / n
    return mat, pathologies


def fig_panel_response_heatmaps() -> None:
    """2x2 grid of per-tier accuracy heatmaps with shared subject ordering."""
    tiers_present, subjects, _ = _ability_table()
    if not tiers_present:
        print("  skip fig_panel_response_heatmaps: no per-tier abilities")
        return

    n = len(tiers_present)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    # Figure width scales with the widest tier's pathology count.
    widths = [max(2, len(TIER_PATHOLOGIES[t])) for t in tiers_present]
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.2 * cols + max(widths) * 0.4, 0.4 * len(subjects) * rows + 1.5),
        squeeze=False,
    )
    im = None
    for k, t in enumerate(tiers_present):
        ax = axes[k // cols][k % cols]
        mat, pathologies = _accuracy_matrix(t, subjects)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pathologies)))
        ax.set_xticklabels(pathologies, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects if (k % cols) == 0 else [""] * len(subjects),
                           fontsize=8)
        for i in range(len(subjects)):
            for j in range(len(pathologies)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="black" if 0.35 < v < 0.85 else "white",
                            fontsize=6)
        ax.set_title(f"{t} tier")
    # Hide any unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Accuracy",
                     shrink=0.7)
    fig.suptitle("Subject x pathology mean accuracy by tier "
                 "(shared subject ordering)", y=1.02)
    savefig(_cross_tier_dir(), "fig_panel_response_heatmaps")


def fig_panel_summary() -> None:
    """2x2 manuscript-candidate composite.

    Inlines simplified versions of: (TL) ability-across-tiers heatmap,
    (TR) model-comparison bar chart, (BL) difficulty violins per tier,
    (BR) 2PL discrimination boxplot per tier.
    """
    tiers_present, subjects, ability_mat = _ability_table()
    summary_path = os.path.join(BY_TIER_DIR, "summary.csv")
    have_summary = os.path.exists(summary_path)
    if not (tiers_present and have_summary):
        print("  skip fig_panel_summary: missing inputs")
        return

    # Pre-compute the right-column data.
    rows = load_csv(summary_path)
    models: list[str] = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    by_tm: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            by_tm[(r["tier"], r["model"])] = float(r["test_nll_per_obs"])
        except (KeyError, ValueError):
            continue

    diff_by_t: dict[str, np.ndarray] = {}
    for t in tiers_present:
        path = os.path.join(BY_TIER_DIR, t, "item_params_rasch.csv")
        if os.path.exists(path):
            v = [float(r["difficulty"]) for r in load_csv(path)
                 if r.get("difficulty") not in (None, "")]
            if v:
                diff_by_t[t] = np.asarray(v, dtype=float)
    discrim_by_t: dict[str, list[float]] = defaultdict(list)
    for t in tiers_present:
        path = os.path.join(BY_TIER_DIR, t, "item_params_twopl.csv")
        if not os.path.exists(path):
            continue
        for r in load_csv(path):
            try:
                discrim_by_t[t].append(float(r["discrimination"]))
            except (KeyError, ValueError):
                continue

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # TL: ability heatmap
    ax_tl = fig.add_subplot(gs[0, 0])
    vmax = float(np.nanmax(np.abs(ability_mat))) if ability_mat.size else 1.0
    im = ax_tl.imshow(ability_mat, aspect="auto", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax)
    ax_tl.set_xticks(range(len(tiers_present)))
    ax_tl.set_xticklabels(tiers_present)
    ax_tl.set_yticks(range(len(subjects)))
    ax_tl.set_yticklabels(subjects, fontsize=8)
    for i in range(len(subjects)):
        for j in range(len(tiers_present)):
            v = ability_mat[i, j]
            if not np.isnan(v):
                ax_tl.text(j, i, f"{v:+.2f}", ha="center", va="center",
                           color="white" if abs(v) > 0.7 * vmax else "black",
                           fontsize=7)
    fig.colorbar(im, ax=ax_tl, label=r"Rasch $\hat{\theta}$", shrink=0.8)
    ax_tl.set_title("Subject ability across tiers")

    # TR: model comparison bar chart
    ax_tr = fig.add_subplot(gs[0, 1])
    n_models = len(models)
    x = np.arange(len(tiers_present))
    width = 0.8 / n_models
    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(models):
        ys = [by_tm.get((t, m), np.nan) for t in tiers_present]
        offsets = x + (k - (n_models - 1) / 2) * width
        ax_tr.bar(offsets, ys, width=width, label=m, color=cmap(k % 10),
                  edgecolor="black", linewidth=0.4)
    ax_tr.set_xticks(x)
    ax_tr.set_xticklabels(tiers_present)
    ax_tr.set_ylabel("Held-out NLL / obs")
    ax_tr.set_title("Model comparison by tier")
    ax_tr.legend(loc="upper left", fontsize=7, ncols=2)
    ax_tr.grid(True, alpha=0.3, axis="y")

    # BL: difficulty violins
    ax_bl = fig.add_subplot(gs[1, 0])
    diff_tiers = [t for t in tiers_present if t in diff_by_t]
    if diff_tiers:
        parts = ax_bl.violinplot([diff_by_t[t] for t in diff_tiers],
                                 showmedians=True, widths=0.85)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        ax_bl.set_xticks(range(1, len(diff_tiers) + 1))
        ax_bl.set_xticklabels(diff_tiers)
    ax_bl.set_ylabel(r"Rasch $\beta$")
    ax_bl.set_title("Item difficulty per tier")
    ax_bl.grid(True, alpha=0.3, axis="y")

    # BR: 2PL discrimination boxplot
    ax_br = fig.add_subplot(gs[1, 1])
    discrim_tiers = [t for t in tiers_present if discrim_by_t.get(t)]
    if discrim_tiers:
        ax_br.boxplot([discrim_by_t[t] for t in discrim_tiers],
                      labels=discrim_tiers, showfliers=False)
    ax_br.set_ylabel("2PL discrimination $a$")
    ax_br.set_title("2PL discrimination per tier")
    ax_br.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Stratified IRT: cross-tier comparative summary",
                 fontsize=14, fontweight="bold")
    savefig(_cross_tier_dir(), "fig_panel_summary")


def fig_ability_profile() -> None:
    """Parallel-coordinates: one polyline per subject across tier positions."""
    tiers_present, subjects, mat = _ability_table()
    if not tiers_present:
        print("  skip fig_ability_profile: no per-tier abilities")
        return
    x = np.arange(len(tiers_present))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(tiers_present), 6))
    for i, s in enumerate(subjects):
        ys = mat[i]
        mask = ~np.isnan(ys)
        if mask.sum() < 1:
            continue
        ax.plot(x[mask], ys[mask], marker="o", color=cmap(i % 20),
                linewidth=1.4, alpha=0.85, label=s)
        # Right-side label at the last available tier.
        last_j = int(np.where(mask)[0][-1])
        ax.annotate(s, (x[last_j], ys[last_j]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=cmap(i % 20))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_present)
    ax.set_xlabel("Prevalence tier")
    ax.set_ylabel(r"Rasch $\hat{\theta}$")
    ax.set_title("Subject ability profile across prevalence tiers")
    ax.grid(True, alpha=0.3, axis="y")
    savefig(_cross_tier_dir(), "fig_ability_profile")


def fig_difficulty_kde_overlay() -> None:
    """4 KDE curves of Rasch difficulty beta on a single axes."""
    by_t: dict[str, np.ndarray] = {}
    for t in TIERS:
        path = os.path.join(BY_TIER_DIR, t, "item_params_rasch.csv")
        if not os.path.exists(path):
            continue
        v = [float(r["difficulty"]) for r in load_csv(path)
             if r.get("difficulty") not in (None, "")]
        if v:
            by_t[t] = np.asarray(v, dtype=float)
    tiers_present = [t for t in TIERS if t in by_t]
    if not tiers_present:
        print("  skip fig_difficulty_kde_overlay: no Rasch items")
        return

    lo = min(float(np.percentile(by_t[t], 0.5)) for t in tiers_present)
    hi = max(float(np.percentile(by_t[t], 99.5)) for t in tiers_present)
    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(tiers_present):
        x = by_t[t]
        # Silverman bandwidth
        std = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
        h = 1.06 * std * (len(x) ** (-1 / 5)) if std > 0 else 1.0
        # KDE on the grid (use a subsample for speed if very large)
        if len(x) > 20000:
            rng = np.random.default_rng(0)
            x = rng.choice(x, size=20000, replace=False)
        diffs = (grid[:, None] - x[None, :]) / h
        density = np.exp(-0.5 * diffs * diffs).sum(axis=1) / (len(x) * h * np.sqrt(2 * np.pi))
        ax.plot(grid, density, label=f"{t} (n={len(by_t[t]):,})",
                color=cmap(k % 10), linewidth=2)
        med = float(np.median(by_t[t]))
        ax.axvline(med, color=cmap(k % 10), linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r"Rasch difficulty $\beta$")
    ax.set_ylabel("density")
    ax.set_title("Rasch difficulty distribution per tier (KDE overlay; "
                 "dashed = median)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(_cross_tier_dir(), "fig_difficulty_kde_overlay")


def fig_info_curves_overlay() -> None:
    """4 Rasch test-information curves overlaid."""
    tiers_present = [
        t for t in TIERS
        if os.path.exists(os.path.join(BY_TIER_DIR, t, "item_params_rasch.csv"))
    ]
    if not tiers_present:
        print("  skip fig_info_curves_overlay: no Rasch items")
        return
    thetas = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(tiers_present):
        rows = load_csv(os.path.join(BY_TIER_DIR, t, "item_params_rasch.csv"))
        diffs = np.array([float(r["difficulty"]) for r in rows])
        P = 1.0 / (1.0 + np.exp(-(thetas[:, None] - diffs[None, :])))
        I = (P * (1.0 - P)).sum(axis=1)
        ax.plot(thetas, I, label=f"{t} (n_items={len(diffs):,})",
                color=cmap(k % 10), linewidth=2)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$I(\theta)$")
    ax.set_title("Test information per prevalence tier (Rasch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(_cross_tier_dir(), "fig_info_curves_overlay")


def fig_pairwise_tier_ability() -> None:
    """2x3 grid of (theta_tierA vs theta_tierB) scatter plots with y=x."""
    tiers_present, subjects, mat = _ability_table()
    if len(tiers_present) < 2:
        print("  skip fig_pairwise_tier_ability: need at least 2 tiers")
        return
    pairs = [(i, j) for i in range(len(tiers_present))
             for j in range(i + 1, len(tiers_present))]
    n = len(pairs)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 3.8 * rows),
                             squeeze=False)
    lo = float(np.nanmin(mat))
    hi = float(np.nanmax(mat))
    pad = 0.1 * max(1e-6, hi - lo)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k // cols][k % cols]
        xs = mat[:, i]
        ys = mat[:, j]
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="gray", linestyle="--", linewidth=0.8)
        ax.scatter(xs[mask], ys[mask], s=70, color="steelblue",
                   edgecolor="black", zorder=3)
        for idx in np.where(mask)[0]:
            ax.annotate(subjects[idx], (xs[idx], ys[idx]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7)
        # Pearson correlation as a label
        if mask.sum() >= 2:
            r = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
            ax.annotate(f"r = {r:+.2f}", (0.05, 0.92), xycoords="axes fraction",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.25",
                                              facecolor="white", alpha=0.8,
                                              edgecolor="gray"))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(rf"$\hat{{\theta}}_{{{tiers_present[i]}}}$")
        ax.set_ylabel(rf"$\hat{{\theta}}_{{{tiers_present[j]}}}$")
        ax.grid(True, alpha=0.3)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Pairwise subject abilities across tiers "
                 "(dashed line: y=x)", y=1.01)
    fig.tight_layout()
    savefig(_cross_tier_dir(), "fig_pairwise_tier_ability")


def fig_pooled_vs_stratified_caterpillar() -> None:
    """Caterpillar overlay: per-tier theta_hat (small markers) + pooled
    Rasch theta_hat (large marker) per subject."""
    pooled_path = os.path.join(REPO_ROOT, "outputs", "irt", "abilities_rasch.csv")
    if not os.path.exists(pooled_path):
        print("  skip fig_pooled_vs_stratified_caterpillar: pooled "
              "abilities_rasch.csv missing")
        return
    tiers_present, subjects, mat = _ability_table()
    if not tiers_present:
        print("  skip fig_pooled_vs_stratified_caterpillar: no per-tier abilities")
        return
    pooled = {r["subject"]: float(r["ability"]) for r in load_csv(pooled_path)}

    # Sort by pooled ability for a clean reading order.
    sorter = [(s, pooled.get(s, float(np.nanmean(mat[idx]))))
              for idx, s in enumerate(subjects)]
    order = [s for s, _ in sorted(sorter, key=lambda x: x[1])]
    s_to_idx = {s: i for i, s in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(order) + 1.5))
    cmap = plt.get_cmap("tab10")
    for y, s in enumerate(order):
        idx = s_to_idx[s]
        for j, t in enumerate(tiers_present):
            v = mat[idx, j]
            if np.isnan(v):
                continue
            ax.scatter(v, y, s=55, color=cmap(j % 10), edgecolor="black",
                       linewidth=0.4, alpha=0.95, zorder=3,
                       label=t if y == 0 else None)
        if s in pooled:
            ax.scatter(pooled[s], y, s=180, marker="X", color="black",
                       zorder=5, label="pooled" if y == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel(r"Rasch $\hat{\theta}$")
    ax.set_title("Pooled vs stratified Rasch ability per subject")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    savefig(_cross_tier_dir(), "fig_pooled_vs_stratified_caterpillar")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    print("--  Generating stratified IRT figures  --")
    for tier in TIERS:
        paths = tier_paths(tier)
        if not os.path.isdir(paths["base"]):
            print(f"  skip tier '{tier}': folder missing")
            continue
        print(f"\n[{tier}]")
        fig_response_heatmap(tier, paths)
        fig_caterpillar(tier, paths)
        fig_icc_examples(tier, paths)
        fig_item_information(tier, paths)
        fig_difficulty_by_pathology(tier, paths)
        fig_factor_loading_heatmap(tier, paths)
        fig_rho_vs_accuracy(tier, paths)

    print("\n[cross-tier]")
    fig_ability_across_tiers()
    fig_model_comparison_by_tier()
    fig_discrim_by_tier_cross()

    print("\n[combined panels]")
    fig_panel_caterpillar_by_tier()
    fig_panel_difficulty_by_tier()
    fig_panel_response_heatmaps()
    fig_panel_summary()

    print("\n[comparative]")
    fig_ability_profile()
    fig_difficulty_kde_overlay()
    fig_info_curves_overlay()
    fig_pairwise_tier_ability()
    fig_pooled_vs_stratified_caterpillar()

    print("\n+ All stratified figures written under outputs/irt/by_tier/")


if __name__ == "__main__":
    main()
