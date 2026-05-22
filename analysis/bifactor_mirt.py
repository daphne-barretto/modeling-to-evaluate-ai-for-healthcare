"""Two-dimensional MIRT: test for a "medical vs general" latent factor.

Hypothesis (CS321M direction 1, headline analysis):
    With ≥3 medical-tuned VLMs (CheXagent, LLaVA-Med, MedGemma, ...) and
    ≥3 general VLMs (GPT-5.4, GPT-4o, Qwen-VL, InternVL3, Phi-3.5, ...),
    a 2-D MIRT model should reveal a second latent dimension that
    captures the medical-tuning gap.

Pipeline
--------
1. Load merged response matrix via the same observations used by
   ``fit_irt.py``.
2. Marginalize observations to per (subject, pathology) cells:
       y_{s,p} = mean correctness of subject s on pathology p.
   This collapses 212K items into 14·J cells but keeps the medical-vs-
   general signal which lives in pathology-level performance, not in
   per-image differences.
3. Fit a 2-D logistic factor model on cells:
       logit P(y_{s,p}) = u_{s,1}·v_{p,1} + u_{s,2}·v_{p,2} + b_p
4. Rotate U,V via varimax to make latent axes interpretable.
5. Report each subject's loading on the second factor; project onto
   {medical, general} group labels and compute the between-group
   separation.

Outputs:
  outputs/irt/bifactor2d_subjects.csv  — subject_id × {factor1, factor2}
  outputs/irt/bifactor2d_items.csv     — pathology × {load1, load2, intercept}
  outputs/irt/bifactor2d_summary.json  — group-mean factor loadings
  outputs/figures/fig_bifactor2d.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import PATHOLOGIES, load_all  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
OUT_SUBJ = os.path.join(IRT_DIR, "bifactor2d_subjects.csv")
OUT_ITEM = os.path.join(IRT_DIR, "bifactor2d_items.csv")
OUT_JSON = os.path.join(IRT_DIR, "bifactor2d_summary.json")
FIG_PDF = os.path.join(FIG_DIR, "fig_bifactor2d.pdf")
FIG_PNG = os.path.join(FIG_DIR, "fig_bifactor2d.png")

EPOCHS = 4000
LR = 0.05

# Medical vs general labels for subjects we expect to see in the response
# matrix. Anything else defaults to "general".
MEDICAL_SUBJECTS = {
    "LLaVA-Med-7B",
    "CheXagent-8B",
    "CheXagent-2-3B",
    "MedGemma-4B",
    "BiomedCLIP",
}


def varimax(L: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Standard Kaiser varimax rotation of a loadings matrix (n_vars × n_factors)."""
    n, m = L.shape
    R = np.eye(m)
    d = 0.0
    for _ in range(q):
        Lr = L @ R
        u, s, vh = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / n) * Lr @ np.diag(np.diag(Lr.T @ Lr))),
            full_matrices=False,
        )
        R = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def load_subject_pathology_cells() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (subjects, Y_correct, N_total) with shape (J, 14)."""
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations found.")

    subjects: list[str] = []
    subj_to_id: dict[str, int] = {}
    for o in obs:
        if o.subject not in subj_to_id:
            subj_to_id[o.subject] = len(subjects)
            subjects.append(o.subject)
    n_subj = len(subjects)
    P = len(PATHOLOGIES)
    path_to_id = {p: i for i, p in enumerate(PATHOLOGIES)}

    Y = np.zeros((n_subj, P), dtype=np.float64)
    N = np.zeros((n_subj, P), dtype=np.float64)
    for o in obs:
        s = subj_to_id[o.subject]
        p = path_to_id[o.pathology]
        Y[s, p] += int(o.response)
        N[s, p] += 1
    return subjects, Y, N


def fit_2d_logistic_fm(Y: np.ndarray, N: np.ndarray) -> dict:
    """Maximize the cell-weighted Bernoulli log-likelihood under
        logit P(y_{s,p}) = u_{s,:} . v_{p,:} + b_p
    with U ∈ R^{J×2}, V ∈ R^{14×2}, b ∈ R^{14}.

    Cells with N=0 are masked.
    """
    n_subj, P = Y.shape
    device = "cpu"
    torch.manual_seed(0)
    U = torch.zeros((n_subj, 2), device=device, requires_grad=True)
    V = torch.zeros((P, 2), device=device, requires_grad=True)
    b = torch.zeros(P, device=device, requires_grad=True)
    Y_t = torch.from_numpy(Y).to(device)
    N_t = torch.from_numpy(N).to(device)
    mask = N_t > 0

    # Initialize V with small random values to break symmetry.
    with torch.no_grad():
        V.add_(torch.randn_like(V) * 0.01)
        U.add_(torch.randn_like(U) * 0.01)
        # Initialize b at the marginal pathology logit
        p_marg = (Y.sum(axis=0) + 1.0) / (N.sum(axis=0) + 2.0)
        b.copy_(torch.from_numpy(np.log(p_marg / (1.0 - p_marg))))

    opt = torch.optim.Adam([U, V, b], lr=LR)
    prev = float("inf")
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits = U @ V.T + b
        log_p = F.logsigmoid(logits)
        log_1mp = F.logsigmoid(-logits)
        ll_cell = Y_t * log_p + (N_t - Y_t) * log_1mp
        nll = -(ll_cell * mask).sum()
        nll.backward()
        opt.step()
        if (ep + 1) % 500 == 0:
            cur = nll.item()
            print(f"  ep {ep + 1:4d}/{EPOCHS}  -ll = {cur:.2f}  Δ={prev - cur:+.4f}", flush=True)
            if abs(prev - cur) < 1e-3 and ep > 1000:
                break
            prev = cur

    return {
        "U": U.detach().cpu().numpy(),
        "V": V.detach().cpu().numpy(),
        "b": b.detach().cpu().numpy(),
        "nll": float(nll.item()),
    }


def fit_1d_logistic_fm(Y: np.ndarray, N: np.ndarray) -> dict:
    """1-factor baseline for likelihood comparison."""
    n_subj, P = Y.shape
    torch.manual_seed(0)
    u = torch.zeros(n_subj, requires_grad=True)
    v = torch.zeros(P, requires_grad=True)
    b = torch.zeros(P, requires_grad=True)
    Y_t = torch.from_numpy(Y).double()
    N_t = torch.from_numpy(N).double()
    mask = N_t > 0
    with torch.no_grad():
        u.add_(torch.randn_like(u) * 0.01)
        v.add_(torch.randn_like(v) * 0.01)
        p_marg = (Y.sum(axis=0) + 1.0) / (N.sum(axis=0) + 2.0)
        b.copy_(torch.from_numpy(np.log(p_marg / (1.0 - p_marg))))
    opt = torch.optim.Adam([u, v, b], lr=LR)
    prev = float("inf")
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits = u.unsqueeze(1) * v.unsqueeze(0) + b
        log_p = F.logsigmoid(logits)
        log_1mp = F.logsigmoid(-logits)
        ll_cell = Y_t * log_p + (N_t - Y_t) * log_1mp
        nll = -(ll_cell * mask).sum()
        nll.backward()
        opt.step()
        if (ep + 1) % 500 == 0:
            cur = nll.item()
            if abs(prev - cur) < 1e-3 and ep > 1000:
                break
            prev = cur
    return {
        "u": u.detach().numpy(),
        "v": v.detach().numpy(),
        "b": b.detach().numpy(),
        "nll": float(nll.item()),
    }


def main() -> None:
    print("[bifactor2d] loading observations + collapsing to (subject, pathology) cells ...", flush=True)
    subjects, Y, N = load_subject_pathology_cells()
    P = Y.shape[1]
    print(
        f"[bifactor2d] {len(subjects)} subjects × {P} pathologies, "
        f"total {int(N.sum()):,} obs",
        flush=True,
    )

    print("[bifactor2d] fitting 2-D logistic FM ...", flush=True)
    fit2 = fit_2d_logistic_fm(Y, N)
    print("[bifactor2d] fitting 1-D logistic FM (baseline) ...", flush=True)
    fit1 = fit_1d_logistic_fm(Y, N)

    delta_nll = fit1["nll"] - fit2["nll"]
    # Param counts: 1-D has J + 2P; 2-D has 2J + 3P. Diff = J + P.
    extra_params = len(subjects) + P
    chi2 = 2.0 * delta_nll
    print(
        f"[bifactor2d] -2ΔLL = {chi2:.2f} on {extra_params} extra params "
        f"(2-D nll {fit2['nll']:.2f} vs 1-D nll {fit1['nll']:.2f})",
        flush=True,
    )

    # Varimax-rotate V (loadings on pathologies); apply inverse to U for invariance.
    V_rot = varimax(fit2["V"])
    # Recover the rotation R such that V_rot = V @ R
    R, *_ = np.linalg.lstsq(fit2["V"], V_rot, rcond=None)
    U_rot = fit2["U"] @ R

    # Convention: orient axes so that "medical" subjects have higher factor-2 loading.
    med_mask = np.array([s in MEDICAL_SUBJECTS for s in subjects])
    gen_mask = ~med_mask
    if med_mask.any() and gen_mask.any():
        # Pick the factor with the largest group mean difference; make it factor 2.
        delta_f1 = U_rot[med_mask, 0].mean() - U_rot[gen_mask, 0].mean()
        delta_f2 = U_rot[med_mask, 1].mean() - U_rot[gen_mask, 1].mean()
        if abs(delta_f1) > abs(delta_f2):
            U_rot = U_rot[:, [1, 0]]
            V_rot = V_rot[:, [1, 0]]
            delta_f1, delta_f2 = delta_f2, delta_f1
        if delta_f2 < 0:
            U_rot[:, 1] *= -1
            V_rot[:, 1] *= -1
        if U_rot[:, 0].mean() < 0:
            U_rot[:, 0] *= -1
            V_rot[:, 0] *= -1

    os.makedirs(IRT_DIR, exist_ok=True)
    with open(OUT_SUBJ, "w") as f:
        w = csv.writer(f)
        w.writerow(["subject", "is_medical", "factor1_general", "factor2_medical"])
        for s, u in zip(subjects, U_rot):
            w.writerow([s, int(s in MEDICAL_SUBJECTS), f"{u[0]:+.6f}", f"{u[1]:+.6f}"])
    with open(OUT_ITEM, "w") as f:
        w = csv.writer(f)
        w.writerow(["pathology", "loading_general", "loading_medical", "intercept"])
        for p, v, b in zip(PATHOLOGIES, V_rot, fit2["b"]):
            w.writerow([p, f"{v[0]:+.6f}", f"{v[1]:+.6f}", f"{b:+.6f}"])

    summary = {
        "n_subjects": len(subjects),
        "n_pathologies": P,
        "nll_1d": fit1["nll"],
        "nll_2d": fit2["nll"],
        "chi2_LRT": chi2,
        "extra_params": extra_params,
        "varimax_applied": True,
        "subjects": subjects,
        "medical_subjects_seen": [s for s in subjects if s in MEDICAL_SUBJECTS],
    }
    if med_mask.any() and gen_mask.any():
        summary["mean_factor1_general_subj"] = float(U_rot[gen_mask, 0].mean())
        summary["mean_factor1_medical_subj"] = float(U_rot[med_mask, 0].mean())
        summary["mean_factor2_general_subj"] = float(U_rot[gen_mask, 1].mean())
        summary["mean_factor2_medical_subj"] = float(U_rot[med_mask, 1].mean())
        summary["factor2_separation"] = (
            float(U_rot[med_mask, 1].mean()) - float(U_rot[gen_mask, 1].mean())
        )

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bifactor2d] wrote {OUT_SUBJ}", flush=True)
    print(f"[bifactor2d] wrote {OUT_ITEM}", flush=True)
    print(f"[bifactor2d] wrote {OUT_JSON}", flush=True)
    if "factor2_separation" in summary:
        print(
            f"[bifactor2d] factor-2 (medical) separation: "
            f"med={summary['mean_factor2_medical_subj']:+.3f} "
            f"gen={summary['mean_factor2_general_subj']:+.3f} "
            f"Δ={summary['factor2_separation']:+.3f}",
            flush=True,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        colors = ["#d62728" if s in MEDICAL_SUBJECTS else "#1f77b4" for s in subjects]
        ax.scatter(U_rot[:, 0], U_rot[:, 1], c=colors, s=80, zorder=3)
        for s, u, c in zip(subjects, U_rot, colors):
            ax.annotate(
                s, (u[0], u[1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=c,
            )
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_xlabel("Factor 1 (general capability)")
        ax.set_ylabel("Factor 2 (medical-tuning loading)")
        ax.set_title(
            f"2-D MIRT subject loadings  (J = {len(subjects)})  "
            f"−2ΔLL = {chi2:.1f}"
        )
        ax.grid(True, alpha=0.3, ls=":")
        fig.tight_layout()
        fig.savefig(FIG_PDF, bbox_inches="tight")
        fig.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
        print(f"[bifactor2d] wrote {FIG_PDF}", flush=True)
    except Exception as e:
        print(f"[bifactor2d] WARN: figure failed: {e}", flush=True)


if __name__ == "__main__":
    main()
