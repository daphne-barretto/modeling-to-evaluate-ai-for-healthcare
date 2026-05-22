"""Within-family scaling law: θ̂ ≈ α·log₁₀(params) + β.

For each open-weight VLM family that has ≥2 sizes in the response
matrix, regress the IRT Rasch ability θ̂ on log model-parameter count.
We also fit a single pooled cross-family model with family-specific
intercepts for visualization.

Outputs:
  outputs/irt/scaling_law.csv      — per-subject (subject, family, params, theta_hat)
  outputs/irt/scaling_law.json     — per-family slope/intercept/R²
  outputs/figures/fig_scaling.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
OUT_CSV = os.path.join(IRT_DIR, "scaling_law.csv")
OUT_JSON = os.path.join(IRT_DIR, "scaling_law.json")
FIG_PDF = os.path.join(FIG_DIR, "fig_scaling.pdf")
FIG_PNG = os.path.join(FIG_DIR, "fig_scaling.png")

# Family / model-size table. Sizes are billions of parameters.
SUBJECT_INFO = {
    # Qwen2.5-VL family
    "Qwen2.5-VL-3B":  {"family": "Qwen2.5-VL",   "params_B":  3.0, "kind": "general"},
    "Qwen2.5-VL-7B":  {"family": "Qwen2.5-VL",   "params_B":  7.0, "kind": "general"},
    "Qwen2.5-VL-32B": {"family": "Qwen2.5-VL",   "params_B": 32.0, "kind": "general"},
    # CheXagent family
    "CheXagent-2-3B": {"family": "CheXagent",    "params_B":  3.0, "kind": "medical"},
    "CheXagent-8B":   {"family": "CheXagent",    "params_B":  8.0, "kind": "medical"},
    # LLaVA family — same 7B size but different fine-tune; gives the
    # medical-vs-non-medical ablation as a "scaling" data point with
    # params constant.
    "LLaVA-1.5-7B":   {"family": "LLaVA-1.5",    "params_B":  7.0, "kind": "general"},
    "LLaVA-Med-7B":   {"family": "LLaVA-Med",    "params_B":  7.0, "kind": "medical"},
    # Other single-size open VLMs (not used for slope fits, just plotted)
    "InternVL3-8B":   {"family": "InternVL3",    "params_B":  8.0, "kind": "general"},
    "Phi-3.5-Vision": {"family": "Phi-3.5-Vision","params_B":  4.2, "kind": "general"},
    "MedGemma-4B":    {"family": "MedGemma",     "params_B":  4.0, "kind": "medical"},
    "BiomedCLIP":     {"family": "BiomedCLIP",   "params_B":  0.2, "kind": "medical"},
    # Closed-source frontier models (no size disclosed)
    "GPT-5.4":        {"family": "GPT-frontier", "params_B": None, "kind": "general"},
    "gpt-4o":         {"family": "GPT-frontier", "params_B": None, "kind": "general"},
    "gpt-5":          {"family": "GPT-frontier", "params_B": None, "kind": "general"},
}


def load_rasch_abilities() -> dict[str, float]:
    out: dict[str, float] = {}
    with open(os.path.join(IRT_DIR, "abilities_rasch.csv")) as f:
        next(f)
        for ln in f:
            n, v = ln.strip().split(",")
            out[n] = float(v)
    return out


def linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([x, np.ones_like(x)])
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = slope * x + intercept
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def main() -> None:
    theta = load_rasch_abilities()
    print(f"[scaling] loaded {len(theta)} Rasch abilities", flush=True)
    rows: list[dict] = []
    for s, t in theta.items():
        info = SUBJECT_INFO.get(s)
        if info is None:
            print(f"  [skip] no metadata for subject {s!r}", flush=True)
            continue
        rows.append({
            "subject": s,
            "family": info["family"],
            "kind": info["kind"],
            "params_B": info["params_B"],
            "theta_hat": t,
        })

    os.makedirs(IRT_DIR, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(
            f, fieldnames=["subject", "family", "kind", "params_B", "theta_hat"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[scaling] wrote {OUT_CSV}", flush=True)

    # Group by family; fit slope only if family has ≥2 known sizes.
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)

    fits: dict[str, dict] = {}
    for fam, items in by_family.items():
        sized = [r for r in items if r["params_B"] is not None]
        if len(sized) < 2:
            continue
        x = np.array([math.log10(r["params_B"]) for r in sized])
        y = np.array([r["theta_hat"] for r in sized])
        slope, intercept, r2 = linreg(x, y)
        fits[fam] = {
            "n_subjects": len(sized),
            "subjects": [r["subject"] for r in sized],
            "slope_per_decade": slope,
            "intercept_at_1B": intercept,
            "r2": r2,
        }
        print(
            f"  [{fam}] n={len(sized)}  θ̂ = {slope:+.3f}·log₁₀(B) + {intercept:+.3f}  R²={r2:+.3f}",
            flush=True,
        )

    summary = {
        "n_subjects_total": len(rows),
        "n_with_known_size": sum(1 for r in rows if r["params_B"] is not None),
        "fits": fits,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[scaling] wrote {OUT_JSON}", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        family_colors = {
            "Qwen2.5-VL":     "#1f77b4",
            "CheXagent":      "#d62728",
            "LLaVA-1.5":      "#7f7f7f",
            "LLaVA-Med":      "#9467bd",
            "InternVL3":      "#2ca02c",
            "Phi-3.5-Vision": "#bcbd22",
            "MedGemma":       "#e377c2",
            "BiomedCLIP":     "#17becf",
            "GPT-frontier":   "#ff7f0e",
        }

        # Scatter all sized subjects.
        sized_rows = [r for r in rows if r["params_B"] is not None]
        for r in sized_rows:
            c = family_colors.get(r["family"], "#888888")
            marker = "s" if r["kind"] == "medical" else "o"
            ax.scatter(
                math.log10(r["params_B"]), r["theta_hat"],
                color=c, marker=marker, s=80, zorder=3, edgecolor="black", linewidth=0.5,
            )
            ax.annotate(
                r["subject"],
                (math.log10(r["params_B"]), r["theta_hat"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7, color=c, zorder=4,
            )

        # Draw per-family fit lines where we have ≥2 sizes.
        for fam, info in fits.items():
            xs = np.array([
                math.log10(SUBJECT_INFO[s]["params_B"]) for s in info["subjects"]
            ])
            xx = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 50)
            yy = info["slope_per_decade"] * xx + info["intercept_at_1B"]
            ax.plot(
                xx, yy,
                color=family_colors.get(fam, "#888888"),
                lw=1.6, alpha=0.7,
                label=f"{fam}: α={info['slope_per_decade']:+.2f} R²={info['r2']:+.2f}",
            )

        # Closed-source models go at a synthetic "x = 2.3" (≈200B) with a "?" annotation.
        closed = [r for r in rows if r["params_B"] is None]
        for r in closed:
            c = family_colors.get(r["family"], "#888888")
            ax.scatter(
                2.3, r["theta_hat"],
                color=c, marker="*", s=120, zorder=3, edgecolor="black", linewidth=0.5,
            )
            ax.annotate(
                f"{r['subject']} (size?)",
                (2.3, r["theta_hat"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7, color=c, zorder=4,
            )

        ax.set_xlabel("log₁₀(parameters in billions)")
        ax.set_ylabel("Rasch θ̂")
        ax.set_title(
            f"Within-family scaling laws on CheXpert IRT\n"
            f"(○ general, ■ medical, ★ closed-source frontier)"
        )
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_PDF, bbox_inches="tight")
        fig.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
        print(f"[scaling] wrote {FIG_PDF}", flush=True)
    except Exception as e:
        print(f"[scaling] WARN: figure failed: {e}", flush=True)


if __name__ == "__main__":
    main()
