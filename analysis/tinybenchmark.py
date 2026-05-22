"""Vision analogue of Polo et al. 2024 ``tinyBenchmarks``.

Given a fitted IRT panel, rank items via several item-selection strategies
and ask how well a *K-item* subset preserves the full-benchmark ranking
of test-takers compared with a uniformly random K-item subset.

Strategies evaluated:
  - ``top_discrimination``    rank by |2PL discrim| (Polo-2024 analogue)
  - ``fisher_info_at_mean``   rank by Fisher info I_j(θ̄) at the
                              population-mean Rasch ability θ̄
  - ``random``                random K subset (N_RANDOM_REPS reps)

For each strategy + K we report Spearman ρ between the full-corpus
ranking of subjects (by raw accuracy on all items) and the ranking
induced by accuracy on just K items.

Outputs
-------
outputs/tinybenchmark.csv   long-format ``(K, strategy, rep, spearman)``
outputs/figures/fig_tinybenchmark.{pdf,png}
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
OUT_CSV = os.path.join(REPO_ROOT, "outputs", "tinybenchmark.csv")
FIG_PDF = os.path.join(REPO_ROOT, "outputs", "figures", "fig_tinybenchmark.pdf")
FIG_PNG = os.path.join(REPO_ROOT, "outputs", "figures", "fig_tinybenchmark.png")

K_GRID = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
N_RANDOM_REPS = 100
SEED = 0


def load_item_disc(path: str) -> np.ndarray:
    """Return |a_j| from a 2PL/3PL item_params CSV in item_idx order."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    a = np.zeros(len(rows), dtype=np.float64)
    for r in rows:
        a[int(r["item_idx"])] = abs(float(r["discrimination"]))
    return a


def load_item_params_twopl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_j, b_j) from a 2PL CSV in item_idx order."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    a = np.zeros(len(rows), dtype=np.float64)
    b = np.zeros(len(rows), dtype=np.float64)
    for r in rows:
        idx = int(r["item_idx"])
        a[idx] = float(r["discrimination"])
        b[idx] = float(r["difficulty"])
    return a, b


def fisher_info_2pl(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
    """Per-item Fisher information I_j(θ) under 2PL.

    I_j(θ) = a_j² · p(θ) · (1 − p(θ))   where  p(θ) = σ(a_j(θ − b_j))
    """
    z = a * (theta - b)
    z = np.clip(z, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-z))
    return (a ** 2) * p * (1.0 - p)


def load_population_theta_mean() -> float:
    """Mean of fitted 2PL θ̂ across subjects."""
    vals = []
    with open(os.path.join(IRT_DIR, "abilities_twopl.csv")) as f:
        next(f)
        for ln in f:
            _, v = ln.strip().split(",")
            vals.append(float(v))
    return float(np.mean(vals))


def load_responses() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Reconstruct (subject_idx, item_idx, response) from any predictions file
    (they all share the same observation order). Subject names from abilities."""
    data = np.load(os.path.join(IRT_DIR, "predictions_rasch.npz"))
    s = data["subject_idx"].astype(np.int32)
    i = data["item_idx"].astype(np.int32)
    y = data["response"].astype(np.int8)
    subjects: list[str] = []
    with open(os.path.join(IRT_DIR, "abilities_rasch.csv")) as f:
        next(f)
        for ln in f:
            subjects.append(ln.split(",")[0])
    return s, i, y, subjects


def subject_acc_on_items(
    s_idx: np.ndarray,
    i_idx: np.ndarray,
    y: np.ndarray,
    n_subjects: int,
    item_set: np.ndarray,
) -> np.ndarray:
    """Return per-subject mean(y) on observations whose item is in ``item_set``.

    Subjects with zero observations in the subset get NaN (so they're dropped
    from rank correlations).
    """
    mask = np.isin(i_idx, item_set, assume_unique=False)
    if not mask.any():
        return np.full(n_subjects, np.nan)
    s_sub = s_idx[mask]
    y_sub = y[mask].astype(np.float64)
    sums = np.bincount(s_sub, weights=y_sub, minlength=n_subjects)
    counts = np.bincount(s_sub, minlength=n_subjects)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return acc


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    ar = a[mask].argsort().argsort().astype(np.float64)
    br = b[mask].argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ar, br)[0, 1])


def main() -> None:
    print("[tinybench] loading 2PL item params...", flush=True)
    a, b = load_item_params_twopl(os.path.join(IRT_DIR, "item_params_twopl.csv"))
    n_items = len(a)
    print(f"[tinybench] {n_items:,} items", flush=True)

    theta_bar = load_population_theta_mean()
    print(f"[tinybench] population θ̄ = {theta_bar:+.3f}", flush=True)

    print("[tinybench] loading responses...", flush=True)
    s_idx, i_idx, y, subjects = load_responses()
    n_subj = len(subjects)
    print(
        f"[tinybench] {n_subj} subjects, "
        f"{len(y):,} observations",
        flush=True,
    )

    print("[tinybench] computing full-corpus ranking...", flush=True)
    full_acc = subject_acc_on_items(
        s_idx, i_idx, y, n_subj, np.arange(n_items)
    )
    full_rank = full_acc.argsort().argsort()
    print(
        "[tinybench] full ranking by accuracy: "
        + ", ".join(
            f"{s}({full_acc[k]:.3f})"
            for k, s in sorted(
                enumerate(subjects), key=lambda x: -full_acc[x[0]]
            )
        ),
        flush=True,
    )

    order_by_disc = np.argsort(-np.abs(a))
    fisher = fisher_info_2pl(a, b, theta_bar)
    order_by_fisher = np.argsort(-fisher)

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for K in K_GRID:
        if K > n_items:
            continue
        # Strategy 1: top |a|
        top_disc_set = order_by_disc[:K]
        top_acc = subject_acc_on_items(s_idx, i_idx, y, n_subj, top_disc_set)
        rho_top = spearman(top_acc, full_acc)
        rows.append({"K": K, "strategy": "top_discrimination", "rep": 0, "spearman": rho_top})

        # Strategy 2: top I_j(θ̄)
        fisher_set = order_by_fisher[:K]
        fisher_acc = subject_acc_on_items(s_idx, i_idx, y, n_subj, fisher_set)
        rho_fisher = spearman(fisher_acc, full_acc)
        rows.append({"K": K, "strategy": "fisher_info_at_mean", "rep": 0, "spearman": rho_fisher})

        # Strategy 3: random K (N_RANDOM_REPS reps)
        rhos_rand = []
        for r in range(N_RANDOM_REPS):
            sample = rng.choice(n_items, size=K, replace=False)
            rand_acc = subject_acc_on_items(s_idx, i_idx, y, n_subj, sample)
            rho = spearman(rand_acc, full_acc)
            rhos_rand.append(rho)
            rows.append({"K": K, "strategy": "random", "rep": r, "spearman": rho})
        print(
            f"  K={K:>5}: top-|a| ρ={rho_top:+.4f}  "
            f"fisher@θ̄ ρ={rho_fisher:+.4f}  "
            f"random ρ̄={np.nanmean(rhos_rand):+.4f} "
            f"(σ={np.nanstd(rhos_rand):.4f})",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=["K", "strategy", "rep", "spearman"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[tinybench] wrote {OUT_CSV}", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ks = sorted({r["K"] for r in rows})
        top = [next(r["spearman"] for r in rows if r["K"] == K and r["strategy"] == "top_discrimination") for K in Ks]
        fisher_line = [next(r["spearman"] for r in rows if r["K"] == K and r["strategy"] == "fisher_info_at_mean") for K in Ks]
        rand_mean = []
        rand_lo = []
        rand_hi = []
        for K in Ks:
            samples = np.array(
                [r["spearman"] for r in rows if r["K"] == K and r["strategy"] == "random"],
                dtype=np.float64,
            )
            samples = samples[~np.isnan(samples)]
            rand_mean.append(samples.mean())
            rand_lo.append(np.quantile(samples, 0.05))
            rand_hi.append(np.quantile(samples, 0.95))

        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.plot(Ks, top, "-o", color="#1f77b4", label="Top-K by |2PL discrim|")
        ax.plot(Ks, fisher_line, "-^", color="#d62728", label="Top-K by Fisher info at θ̄")
        ax.plot(Ks, rand_mean, "-s", color="#aaaaaa", label="Random K (mean)")
        ax.fill_between(Ks, rand_lo, rand_hi, color="#aaaaaa", alpha=0.25,
                        label="Random K 5–95%")
        ax.set_xscale("log")
        ax.set_xlabel("Subset size K")
        ax.set_ylabel("Spearman ρ vs. full-corpus ranking")
        ax.set_title(f"Tiny-benchmark recovery ({len(subjects)} subjects)")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(1.0, color="black", linestyle=":", lw=0.7)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout()
        os.makedirs(os.path.dirname(FIG_PDF), exist_ok=True)
        fig.savefig(FIG_PDF, bbox_inches="tight")
        fig.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
        print(f"[tinybench] wrote {FIG_PDF}", flush=True)
    except Exception as e:
        print(f"[tinybench] WARN: figure failed: {e}", flush=True)


if __name__ == "__main__":
    main()
