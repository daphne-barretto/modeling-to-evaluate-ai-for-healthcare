"""Compare IRT-derived subject rankings against accuracy / stratification.

Addresses peer-review feedback #1: "Do we need IRT modeling to predict
what kinds of questions will be answered incorrectly by a model, or is
stratification along different metadata enough to capture the relationship?"

Produces:
    outputs/irt/ranking_comparison.csv     — Spearman/Pearson ρ between
                                            subject-orderings from each
                                            baseline vs each IRT model
    outputs/irt/discrimination_within_tier.csv
                                          — per-pathology discrimination
                                            spread inside each prevalence
                                            tier (the thing stratification
                                            CAN'T see)
    outputs/irt/dif_by_sex.csv            — item-difficulty shift across
                                            sex groups (DIF candidates)
    outputs/irt/dif_by_age_bin.csv        — same, across age bins
    outputs/irt/headline_findings.json    — single dict of headline numbers
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import load_all, PATHOLOGIES  # noqa: E402
from analysis.item_metadata import (  # noqa: E402
    ANATOMICAL_GROUP, PREVALENCE_TIER, get_image_meta,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def spearman(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    return pearson(rx, ry)


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom else float("nan")


def _rank(values):
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def ranking_comparison(obs):
    # baselines: aggregate accuracy + per-pathology mean accuracy
    subjects = sorted({o.subject for o in obs})

    def per_subject_metric(value_fn):
        return [value_fn(s) for s in subjects]

    # aggregate accuracy
    agg = {}
    for s in subjects:
        sub = [o.response for o in obs if o.subject == s]
        agg[s] = sum(sub) / len(sub) if sub else float("nan")
    agg_vec = [agg[s] for s in subjects]

    # per-pathology mean accuracy → average over pathologies (a "macro" baseline)
    macro = {}
    for s in subjects:
        per_p = []
        for p in PATHOLOGIES:
            cells = [o.response for o in obs if o.subject == s and o.pathology == p]
            if cells:
                per_p.append(sum(cells) / len(cells))
        macro[s] = sum(per_p) / len(per_p) if per_p else float("nan")
    macro_vec = [macro[s] for s in subjects]

    # per-prevalence-tier accuracy → macro-tier average
    tier_macro = {}
    for s in subjects:
        per_t = []
        for t in ["high", "mid", "low"]:
            cells = [o.response for o in obs
                     if o.subject == s and PREVALENCE_TIER.get(o.pathology) == t]
            if cells:
                per_t.append(sum(cells) / len(cells))
        tier_macro[s] = sum(per_t) / len(per_t) if per_t else float("nan")
    tier_vec = [tier_macro[s] for s in subjects]

    # IRT abilities
    irt_rankings = {}
    for model_name in ("rasch", "twopl", "threepl", "fm1", "fm2", "fm3", "bifactor"):
        path = os.path.join(OUT_DIR, f"abilities_{model_name}.csv")
        if not os.path.exists(path):
            continue
        rows = _load_csv(path)
        s_to_theta = {}
        for r in rows:
            # FM models have multi-D abilities; use the L2 norm as a single
            # composite ranking for comparison purposes.
            cols = [k for k in r if k.startswith("ability")]
            if len(cols) == 1:
                s_to_theta[r["subject"]] = float(r[cols[0]])
            else:
                vec = [float(r[c]) for c in cols]
                s_to_theta[r["subject"]] = math.sqrt(sum(v * v for v in vec))
        if all(s in s_to_theta for s in subjects):
            irt_rankings[model_name] = [s_to_theta[s] for s in subjects]

    # Pairwise correlations
    baselines = {
        "aggregate_accuracy": agg_vec,
        "macro_per_pathology_accuracy": macro_vec,
        "macro_prevalence_tier_accuracy": tier_vec,
    }

    rows = []
    for irt_name, vec in irt_rankings.items():
        for base_name, base_vec in baselines.items():
            rho = spearman(vec, base_vec)
            r = pearson(vec, base_vec)
            rows.append((irt_name, base_name, len(subjects),
                         f"{rho:+.4f}", f"{r:+.4f}"))
    # also cross IRT-vs-IRT
    irt_list = list(irt_rankings.items())
    for i in range(len(irt_list)):
        for j in range(i + 1, len(irt_list)):
            n1, v1 = irt_list[i]
            n2, v2 = irt_list[j]
            rho = spearman(v1, v2)
            r = pearson(v1, v2)
            rows.append((n1, n2, len(subjects), f"{rho:+.4f}", f"{r:+.4f}"))
    path = os.path.join(OUT_DIR, "ranking_comparison.csv")
    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "n_subjects", "spearman_rho", "pearson_r"])
        for row in rows:
            w.writerow(row)
    print(f"✓ ranking comparison rows: {len(rows)}")
    return {"subjects": subjects,
            "baselines": {k: dict(zip(subjects, v)) for k, v in baselines.items()},
            "irt": {k: dict(zip(subjects, v)) for k, v in irt_rankings.items()}}


def discrimination_within_tier():
    """For 2PL: show that within a prevalence tier, item discrimination
    varies — i.e. some items in the 'easy' tier are still highly
    discriminating, which simple per-tier accuracy stratification CANNOT
    detect."""
    path = os.path.join(OUT_DIR, "item_params_twopl.csv")
    if not os.path.exists(path):
        print("  (twopl not fit yet)")
        return
    rows = _load_csv(path)
    tier_discrim = defaultdict(list)
    for r in rows:
        tier = PREVALENCE_TIER.get(r["pathology"], "na")
        try:
            tier_discrim[tier].append(float(r["discrimination"]))
        except (KeyError, ValueError):
            continue
    out_rows = []
    for tier, vals in tier_discrim.items():
        if not vals:
            continue
        vals.sort()
        mean = sum(vals) / len(vals)
        p10 = vals[int(0.10 * len(vals))]
        p50 = vals[int(0.50 * len(vals))]
        p90 = vals[int(0.90 * len(vals))]
        spread = p90 - p10
        out_rows.append((tier, len(vals), f"{mean:.4f}", f"{p10:.4f}",
                         f"{p50:.4f}", f"{p90:.4f}", f"{spread:.4f}"))
    out_rows.sort()
    out_path = os.path.join(OUT_DIR, "discrimination_within_tier.csv")
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["prevalence_tier", "n_items", "mean_a", "p10", "p50", "p90", "p90_minus_p10"])
        for row in out_rows:
            w.writerow(row)
    print(f"✓ discrimination-within-tier rows: {len(out_rows)}")
    return {row[0]: row for row in out_rows}


def dif_by_metadata(axis_name, key_fn):
    """For each pathology, compute mean response by metadata bucket per
    subject, then average over subjects → 'group difficulty' per
    (pathology, bucket). Big swings flag DIF.
    """
    obs = load_all(REPO_ROOT)
    # Per (pathology, subject, bucket) → mean response
    cells = defaultdict(lambda: [0, 0])
    for o in obs:
        meta = get_image_meta(o.image_path, REPO_ROOT)
        if meta is None:
            continue
        bucket = key_fn(meta)
        if not bucket or bucket == "unknown":
            continue
        cells[(o.pathology, o.subject, bucket)][0] += 1
        cells[(o.pathology, o.subject, bucket)][1] += o.response

    # For each (pathology, bucket): macro-average accuracy across subjects
    macro = defaultdict(lambda: [0.0, 0])
    for (p, s, b), (n, c) in cells.items():
        if n < 10:
            continue
        acc = c / n
        macro[(p, b)][0] += acc
        macro[(p, b)][1] += 1
    macro = {k: v[0] / v[1] for k, v in macro.items() if v[1] > 0}

    pathologies = sorted({p for (p, _) in macro})
    buckets = sorted({b for (_, b) in macro})
    rows = []
    dif_signals = []
    for p in pathologies:
        vals = {b: macro.get((p, b)) for b in buckets}
        valid = [v for v in vals.values() if v is not None]
        if len(valid) < 2:
            continue
        spread = max(valid) - min(valid)
        rows.append([p] + [f"{v:.4f}" if v is not None else ""
                           for v in [vals[b] for b in buckets]] + [f"{spread:.4f}"])
        dif_signals.append((p, spread))
    dif_signals.sort(key=lambda x: -x[1])

    out_path = os.path.join(OUT_DIR, f"dif_by_{axis_name}.csv")
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["pathology"] + buckets + ["max_minus_min"])
        for row in rows:
            w.writerow(row)
    print(f"✓ DIF-by-{axis_name}: top 3 swings → "
          + ", ".join(f"{p}: {s:.3f}" for p, s in dif_signals[:3]))
    return {"top_dif": dif_signals[:5]}


def headline_findings(ranking_info, tier_info, dif_sex, dif_age):
    headline = {}

    # Best held-out IRT model from fit_table.csv
    fit_table_path = os.path.join(OUT_DIR, "fit_table.csv")
    if os.path.exists(fit_table_path):
        rows = _load_csv(fit_table_path)
        rows = [r for r in rows if r["test_nll_per_obs"]]
        if rows:
            best = min(rows, key=lambda r: float(r["test_nll_per_obs"]))
            headline["best_holdout_model"] = best["model"]
            headline["best_holdout_nll_per_obs"] = float(best["test_nll_per_obs"])
            headline["model_fit_table"] = [
                {k: r[k] for k in ("model", "test_nll_per_obs", "BIC", "n_params")}
                for r in rows
            ]

    # Spearman of best IRT model vs aggregate accuracy
    if ranking_info:
        with open(os.path.join(OUT_DIR, "ranking_comparison.csv")) as f:
            rc_rows = list(csv.DictReader(f))
        best_model = headline.get("best_holdout_model", "rasch")
        for r in rc_rows:
            if r["a"] == best_model and r["b"] == "aggregate_accuracy":
                headline["spearman_best_irt_vs_aggregate"] = float(r["spearman_rho"])
                break
            if r["b"] == best_model and r["a"] == "aggregate_accuracy":
                headline["spearman_best_irt_vs_aggregate"] = float(r["spearman_rho"])
                break

    # Within-tier discrimination spread (the key "stratification can't see this" point)
    if tier_info:
        headline["discrimination_spread_within_tier"] = {
            t: {"p10_to_p90_range": row[6]} for t, row in tier_info.items()
        }

    headline["top_dif_by_sex"] = dif_sex.get("top_dif") if dif_sex else None
    headline["top_dif_by_age"] = dif_age.get("top_dif") if dif_age else None

    out_path = os.path.join(OUT_DIR, "headline_findings.json")
    with open(out_path, "w") as f:
        json.dump(headline, f, indent=2)
    print(f"\n✓ Wrote outputs/irt/headline_findings.json")
    return headline


def main():
    obs = load_all(REPO_ROOT)

    print("\n──  Ranking comparison (IRT abilities vs baselines)  ──")
    ranking_info = ranking_comparison(obs)

    print("\n──  Discrimination spread within prevalence tier  ──")
    tier_info = discrimination_within_tier()

    print("\n──  DIF by sex (image metadata)  ──")
    dif_sex = dif_by_metadata("sex", lambda m: m.sex)

    print("\n──  DIF by age bin  ──")
    dif_age = dif_by_metadata("age_bin", lambda m: m.age_bin)

    print("\n──  Headline findings  ──")
    headline = headline_findings(ranking_info, tier_info, dif_sex, dif_age)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
