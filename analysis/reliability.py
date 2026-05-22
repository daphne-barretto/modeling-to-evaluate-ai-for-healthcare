"""Bootstrap confidence intervals on subject ability θ̂.

For each subject, resample item observations with replacement B times,
re-estimate Rasch θ̂ on the bootstrap sample, and report the 5th and 95th
percentiles. Produces:
  - outputs/irt/theta_bootstrap_ci.csv: subject, theta_hat, theta_lo, theta_hi, sd
  - outputs/figures/fig_caterpillar_ci.{pdf,png}: caterpillar with 90% CI bars
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
OUT_CSV = os.path.join(IRT_DIR, "theta_bootstrap_ci.csv")
FIG_PDF = os.path.join(REPO_ROOT, "outputs", "figures", "fig_caterpillar_ci.pdf")
FIG_PNG = os.path.join(REPO_ROOT, "outputs", "figures", "fig_caterpillar_ci.png")

B = int(os.environ.get("RELIAB_B", "500"))
ALPHA = 0.05
SEED = 0
N_NEWTON = 30


def load_rasch():
    """Return item_difficulties (n_items,) and obs arrays."""
    diff = np.zeros(0, dtype=np.float64)
    with open(os.path.join(IRT_DIR, "item_params_rasch.csv")) as f:
        r = csv.DictReader(f)
        rows = list(r)
    diff = np.zeros(len(rows), dtype=np.float64)
    for row in rows:
        diff[int(row["item_idx"])] = float(row["difficulty"])
    data = np.load(os.path.join(IRT_DIR, "predictions_rasch.npz"))
    return diff, data["subject_idx"].astype(np.int32), data["item_idx"].astype(np.int32), data["response"].astype(np.int8)


def load_subjects() -> list[str]:
    subs = []
    with open(os.path.join(IRT_DIR, "abilities_rasch.csv")) as f:
        next(f)
        for ln in f:
            subs.append(ln.split(",")[0])
    return subs


def load_theta_hat() -> dict[str, float]:
    out: dict[str, float] = {}
    with open(os.path.join(IRT_DIR, "abilities_rasch.csv")) as f:
        next(f)
        for ln in f:
            name, val = ln.strip().split(",")
            out[name] = float(val)
    return out


def estimate_theta_rasch(b: np.ndarray, y: np.ndarray, init: float = 0.0) -> float:
    """Newton-Raphson MLE of θ in Rasch with fixed difficulties b for response y.

    Score:  s(θ) = Σ_j (y_j - σ(θ - b_j))
    Info:   I(θ) = Σ_j σ(θ-b_j)(1-σ(θ-b_j))
    """
    theta = init
    for _ in range(N_NEWTON):
        z = theta - b
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (y - p).sum()
        info = (p * (1.0 - p)).sum()
        if info <= 1e-9:
            break
        step = grad / info
        theta += float(step)
        if abs(step) < 1e-6:
            break
    return float(theta)


def main() -> None:
    print("[reliab] loading Rasch params + responses...", flush=True)
    b, s_idx, i_idx, y = load_rasch()
    subjects = load_subjects()
    theta_hat = load_theta_hat()
    n_subjects = len(subjects)
    print(
        f"[reliab] {n_subjects} subjects, {len(y):,} observations, B={B}",
        flush=True,
    )

    rng = np.random.default_rng(SEED)
    boot_thetas = np.full((n_subjects, B), np.nan, dtype=np.float64)
    se_per_subject = np.full(n_subjects, np.nan, dtype=np.float64)

    for k, name in enumerate(subjects):
        mask = s_idx == k
        b_k = b[i_idx[mask]]
        y_k = y[mask].astype(np.float64)
        n_k = len(y_k)
        if n_k < 10:
            print(f"  {name}: <10 obs, skipping", flush=True)
            continue
        init = theta_hat[name]
        for rep in range(B):
            sample = rng.integers(0, n_k, size=n_k)
            t = estimate_theta_rasch(b_k[sample], y_k[sample], init=init)
            boot_thetas[k, rep] = t
        # Wald CI around the joint-MLE point estimate using bootstrap SE.
        # (Percentile CI is biased here because the marginal-MLE used in the
        # bootstrap differs slightly from the joint-MLE point estimate.)
        sd = float(np.std(boot_thetas[k, :], ddof=1))
        z = 1.959963984540054  # 97.5th percentile of N(0,1)
        lo = init - z * sd
        hi = init + z * sd
        se_per_subject[k] = sd
        print(
            f"  {name}: θ̂={init:+.3f}   "
            f"CI{int((1-ALPHA)*100)}=[{lo:+.3f}, {hi:+.3f}]   "
            f"SE={sd:.4f}   n={n_k:,}",
            flush=True,
        )

    # Marginal reliability = Var(θ̂_population) / (Var(θ̂_population) + mean Var(θ̂_subject))
    thetas_pop = np.array([theta_hat[s] for s in subjects], dtype=np.float64)
    pop_var = float(np.var(thetas_pop, ddof=1))
    mean_se2 = float(np.nanmean(se_per_subject**2))
    rel = pop_var / (pop_var + mean_se2) if (pop_var + mean_se2) > 0 else float("nan")
    print(
        f"\n[reliab] population θ̂ variance = {pop_var:.4f}   "
        f"mean SE² = {mean_se2:.4f}   "
        f"marginal reliability ≈ {rel:.4f}",
        flush=True,
    )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.writer(f)
        w.writerow(["subject", "theta_hat", "theta_lo", "theta_hi", "sd", "n_obs"])
        for k, name in enumerate(subjects):
            n_k = int((s_idx == k).sum())
            mask = ~np.isnan(boot_thetas[k, :])
            if not mask.any():
                w.writerow([name, theta_hat[name], "", "", "", n_k])
                continue
            init = theta_hat[name]
            sd = float(np.std(boot_thetas[k, mask], ddof=1))
            z = 1.959963984540054
            lo = init - z * sd
            hi = init + z * sd
            w.writerow([
                name,
                f"{init:+.6f}",
                f"{lo:+.6f}",
                f"{hi:+.6f}",
                f"{sd:.6f}",
                n_k,
            ])
    print(f"[reliab] wrote {OUT_CSV}", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort([theta_hat[s] for s in subjects])
        names = [subjects[i] for i in order]
        thetas = [theta_hat[s] for s in names]
        z = 1.959963984540054
        sds = [float(np.std(boot_thetas[i, :], ddof=1)) for i in order]
        err = [z * s for s in sds]
        fig, ax = plt.subplots(figsize=(5.6, max(2.2, 0.35 * len(names) + 1)))
        ys = np.arange(len(names))
        ax.errorbar(thetas, ys, xerr=err, fmt="o", color="#1f77b4",
                    ecolor="#1f77b4", elinewidth=1.5, capsize=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.set_xlabel(r"Subject ability $\hat\theta_i$ (Rasch, with 95% Wald CI from bootstrap SE)")
        ax.set_title(f"Test-taker ability, n={B} bootstrap reps")
        ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout()
        os.makedirs(os.path.dirname(FIG_PDF), exist_ok=True)
        fig.savefig(FIG_PDF, bbox_inches="tight")
        fig.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
        print(f"[reliab] wrote {FIG_PDF}", flush=True)
    except Exception as e:
        print(f"[reliab] WARN: figure failed: {e}", flush=True)

    print(f"\n[reliab] DONE. marginal_reliability={rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
