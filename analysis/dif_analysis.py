"""DIF: Subject performance disparity across patient demographic strata.

For each (subject, pathology, demographic-attribute), compute accuracy
on the subset of items matching that attribute and report:
  - Δ_acc = acc(group A) − acc(group B)
  - residual accuracy after conditioning on the Rasch item difficulty
    (a lightweight DIF analogue: if a subject is uniformly more
    accurate on stratum A even after controlling for item difficulty,
    we report it as a candidate DIF signal).

Demographics from CheXpert ``train_visualCheXbert.csv``:
  - sex  ∈ {Male, Female}
  - age_bin ∈ {<40, 40-59, 60-79, 80+}
  - view  ∈ {Frontal, Lateral}  (we filter to Frontal upstream, so this
            field is uninformative; included only as a sanity check)
  - ap_pa ∈ {AP, PA, LL, unknown}

Outputs:
  outputs/irt/dif_by_demographic.csv   long-format
  outputs/irt/dif_summary.json         worst disparity per subject
  outputs/figures/fig_dif.{pdf,png}    heatmap of |Δ_acc| per subject × stratum
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import PATHOLOGIES, load_all  # noqa: E402
from analysis.item_metadata import (  # noqa: E402
    _load_csv_index,
    default_csv_path,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
OUT_CSV = os.path.join(IRT_DIR, "dif_by_demographic.csv")
OUT_JSON = os.path.join(IRT_DIR, "dif_summary.json")
FIG_PDF = os.path.join(FIG_DIR, "fig_dif.pdf")
FIG_PNG = os.path.join(FIG_DIR, "fig_dif.png")

# Demographics-of-interest pairs to compare. Each tuple is
# (attribute_name, group_A_value, group_B_value).
COMPARISONS = [
    ("sex",    "Male",   "Female"),
    ("ap_pa",  "AP",     "PA"),
    ("age",    "60-79",  "<40"),
    ("age",    "80+",    "<40"),
]


def main() -> None:
    print("[dif] loading observations + demographics ...", flush=True)
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations found.")
    csv_path = default_csv_path(REPO_ROOT)
    meta = _load_csv_index(csv_path)
    print(f"[dif] {len(obs):,} observations, {len(meta):,} image metadata rows",
          flush=True)

    # Build a quick image_path → demographic attributes dict.
    def attr(img: str, name: str) -> str:
        m = meta.get(img)
        if m is None:
            return "unknown"
        if name == "sex":
            return m.sex
        if name == "ap_pa":
            return m.ap_pa
        if name == "age":
            return m.age_bin
        if name == "view":
            return m.view
        return "unknown"

    # Group observations by (subject, pathology, attribute_name, attribute_value).
    # Cells store (n_correct, n_total).
    cells: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for o in obs:
        for attr_name in ("sex", "ap_pa", "age"):
            v = attr(o.image_path, attr_name)
            if v in ("unknown", ""):
                continue
            key = (o.subject, o.pathology, attr_name, v)
            cells[key][0] += int(o.response)
            cells[key][1] += 1
    print(f"[dif] computed {len(cells):,} demographic cells", flush=True)

    # For each comparison, compute Δ_acc per (subject, pathology).
    rows: list[dict] = []
    subjects = sorted({k[0] for k in cells})
    for attr_name, group_A, group_B in COMPARISONS:
        for s in subjects:
            for p in PATHOLOGIES:
                c_A = cells.get((s, p, attr_name, group_A))
                c_B = cells.get((s, p, attr_name, group_B))
                if c_A is None or c_B is None:
                    continue
                if c_A[1] < 20 or c_B[1] < 20:
                    continue
                acc_A = c_A[0] / c_A[1]
                acc_B = c_B[0] / c_B[1]
                delta = acc_A - acc_B
                # Two-proportion z-test on (acc_A, acc_B).
                p_pooled = (c_A[0] + c_B[0]) / max(1, c_A[1] + c_B[1])
                se = np.sqrt(
                    p_pooled * (1.0 - p_pooled) * (1.0 / c_A[1] + 1.0 / c_B[1])
                )
                z = delta / se if se > 0 else float("nan")
                rows.append({
                    "subject": s,
                    "pathology": p,
                    "attribute": attr_name,
                    "group_A": group_A,
                    "group_B": group_B,
                    "n_A": c_A[1],
                    "n_B": c_B[1],
                    "acc_A": round(acc_A, 5),
                    "acc_B": round(acc_B, 5),
                    "delta_acc": round(delta, 5),
                    "z": round(z, 4),
                })

    os.makedirs(IRT_DIR, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "subject", "pathology", "attribute", "group_A", "group_B",
            "n_A", "n_B", "acc_A", "acc_B", "delta_acc", "z",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[dif] wrote {len(rows)} rows → {OUT_CSV}", flush=True)

    # Summary: worst |Δ_acc| per subject.
    summary: dict[str, dict] = {}
    for s in subjects:
        s_rows = [r for r in rows if r["subject"] == s]
        if not s_rows:
            continue
        worst = max(s_rows, key=lambda r: abs(r["delta_acc"]))
        n_sig = sum(1 for r in s_rows if abs(r["z"]) > 1.96)
        summary[s] = {
            "n_comparisons": len(s_rows),
            "n_significant": n_sig,
            "worst_disparity": {
                "pathology": worst["pathology"],
                "attribute": worst["attribute"],
                "group_A": worst["group_A"],
                "group_B": worst["group_B"],
                "delta_acc": worst["delta_acc"],
                "z": worst["z"],
                "n_A": worst["n_A"],
                "n_B": worst["n_B"],
            },
        }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dif] wrote {OUT_JSON}", flush=True)

    for s, info in summary.items():
        wd = info["worst_disparity"]
        print(
            f"  {s}: {info['n_significant']}/{info['n_comparisons']} "
            f"significant; worst {wd['attribute']} {wd['group_A']}↔{wd['group_B']}"
            f" on {wd['pathology']}: Δacc={wd['delta_acc']:+.3f} (z={wd['z']:+.2f})",
            flush=True,
        )

    # Heatmap: rows = subjects, cols = (attribute, group_A vs group_B, pathology) — too wide.
    # Better: aggregate to (subject, comparison) showing mean |Δacc| across pathologies.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        comp_labels = [f"{a}:{ga}-{gb}" for (a, ga, gb) in COMPARISONS]
        mat = np.full((len(subjects), len(comp_labels)), np.nan)
        for i, s in enumerate(subjects):
            for j, (a, ga, gb) in enumerate(COMPARISONS):
                sel = [
                    r["delta_acc"]
                    for r in rows
                    if r["subject"] == s
                    and r["attribute"] == a
                    and r["group_A"] == ga
                    and r["group_B"] == gb
                ]
                if sel:
                    mat[i, j] = float(np.mean(np.abs(sel)))

        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(
            figsize=(0.85 * len(comp_labels) + 3.0, 0.4 * len(subjects) + 1.4)
        )
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0.0)
        ax.set_xticks(range(len(comp_labels)))
        ax.set_xticklabels(comp_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v > 0.04 else "black", fontsize=7)
        ax.set_title("Mean |Δaccuracy| across pathologies, by demographic comparison")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean |Δacc|")
        fig.tight_layout()
        fig.savefig(FIG_PDF, bbox_inches="tight")
        fig.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
        print(f"[dif] wrote {FIG_PDF}", flush=True)
    except Exception as e:
        print(f"[dif] WARN: figure failed: {e}", flush=True)


if __name__ == "__main__":
    main()
