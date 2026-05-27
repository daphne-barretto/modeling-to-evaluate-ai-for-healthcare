"""Fit the IRT suite separately on each prevalence-tier sub-corpus.

For every tier in ``PREVALENCE_TIER`` (high / mid / low / na) we restrict the
unified response matrix from ``data_loader.load_all`` to observations whose
pathology belongs to that tier, persist the partitioned observations to
``data/observations_by_tier/{tier}.jsonl`` (so downstream analyses can
re-consume them without re-running every loader), and refit the same family
of measurement models that ``fit_irt.py`` runs on the pooled corpus —
**minus the bifactor model**, because anatomical-group factors collapse or
become unidentifiable when conditioned on a single prevalence regime
(see ``analysis/stratified_irt_methodology.md`` for the full rationale).

Outputs
-------

``data/observations_by_tier/``
    ``{high,mid,low,na}.jsonl``   tier-separated long-form observations
    ``manifest.json``             input files consumed, per-tier counts

``outputs/irt/by_tier/``
    ``summary.csv``               tier × model fit table (24 rows)
    ``{tier}/``
        ``fit_table.csv``         6 rows: rasch, twopl, threepl, fm1, fm2, fm3
        ``fit_summary.json``      counts + per-model headline stats
        ``abilities_{model}.csv``
        ``item_params_{model}.csv``
        ``predictions_{model}.npz``
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import fit_irt  # noqa: E402  (we monkey-patch OUT_DIR per tier)
from analysis.data_loader import discover_sources, load_all  # noqa: E402
from analysis.fit_irt import (  # noqa: E402
    build_index,
    fit_one,
    save_item_params,
    save_predictions,
    save_subject_params,
    summarize_corpus,
    to_tensors,
    train_test_split,
)
from analysis.item_metadata import PREVALENCE_TIER  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER_OBS_DIR = os.path.join(REPO_ROOT, "data", "observations_by_tier")
BY_TIER_OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "irt", "by_tier")
METHODOLOGY_DOC = "analysis/stratified_irt_methodology.md"

TIERS = ["high", "mid", "low", "na"]


def partition_by_tier(observations):
    """Group observations into {high, mid, low, na} buckets."""
    buckets: dict[str, list] = defaultdict(list)
    for o in observations:
        buckets[PREVALENCE_TIER.get(o.pathology, "na")].append(o)
    return buckets


def write_tier_observations(buckets, source_paths):
    os.makedirs(TIER_OBS_DIR, exist_ok=True)
    manifest = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generated_from": [os.path.relpath(p, REPO_ROOT) for p in source_paths],
        "tiers": {},
    }
    for tier in TIERS:
        items = buckets.get(tier, [])
        path = os.path.join(TIER_OBS_DIR, f"{tier}.jsonl")
        subjects: set[str] = set()
        item_keys: set[tuple[str, str]] = set()
        pathologies: set[str] = set()
        with open(path, "w") as f:
            for o in items:
                f.write(json.dumps({
                    "subject": o.subject,
                    "image_path": o.image_path,
                    "pathology": o.pathology,
                    "response": int(o.response),
                }) + "\n")
                subjects.add(o.subject)
                item_keys.add((o.image_path, o.pathology))
                pathologies.add(o.pathology)
        manifest["tiers"][tier] = {
            "n_obs": len(items),
            "n_subjects": len(subjects),
            "n_items": len(item_keys),
            "pathologies": sorted(pathologies),
            "file": os.path.relpath(path, REPO_ROOT),
        }
        print(f"  wrote {len(items):>8,} obs to {os.path.relpath(path, REPO_ROOT)} "
              f"({len(subjects)} subjects × {len(item_keys):,} items, "
              f"{len(pathologies)} pathologies)")

    manifest_path = os.path.join(TIER_OBS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote manifest: {os.path.relpath(manifest_path, REPO_ROOT)}")
    return manifest


def fit_tier(tier, observations):
    """Fit rasch through fm3 on a single tier's observations.

    Mirrors ``fit_irt.main()``'s persistence pattern but redirects
    ``fit_irt.OUT_DIR`` to a per-tier folder so the imported ``save_*``
    helpers land their files in the right place.
    """
    print(f"\n══════════════════════════════════════════════════════════")
    print(f"  Tier: {tier}   ({len(observations):,} observations)")
    print(f"══════════════════════════════════════════════════════════")

    if not observations:
        print(f"  (no observations for tier {tier} — skipping)")
        return []

    out_dir = os.path.join(BY_TIER_OUT_ROOT, tier)
    os.makedirs(out_dir, exist_ok=True)
    fit_irt.OUT_DIR = out_dir  # redirect save_* helpers

    subject_to_id, item_to_id, subject_names, item_keys = build_index(observations)
    s_idx, i_idx, y = to_tensors(observations, subject_to_id, item_to_id)
    n_subjects = len(subject_names)
    n_items = len(item_keys)

    s_train, i_train, y_train, s_test, i_test, y_test = train_test_split(
        s_idx, i_idx, y, test_frac=0.10, seed=0,
    )

    summary = summarize_corpus(observations, subject_names, item_keys)
    print(f"\nCorpus: {n_subjects} subjects × {n_items:,} items × "
          f"{len(observations):,} obs")

    from torch_measure.models import LogisticFM, Rasch, ThreePL, TwoPL

    fits: dict[str, dict] = {}

    def _persist(name, stats, model):
        fits[name] = stats
        if name == "rasch":
            save_subject_params(name, model.ability.detach().cpu().numpy(), subject_names)
            save_item_params(name, item_keys, {
                "difficulty": model.difficulty.detach().cpu().numpy(),
            })
        elif name == "twopl":
            save_subject_params(name, model.ability.detach().cpu().numpy(), subject_names)
            save_item_params(name, item_keys, {
                "difficulty": model.difficulty.detach().cpu().numpy(),
                "discrimination": model.discrimination.detach().cpu().numpy(),
            })
        elif name == "threepl":
            save_subject_params(name, model.ability.detach().cpu().numpy(), subject_names)
            save_item_params(name, item_keys, {
                "difficulty": model.difficulty.detach().cpu().numpy(),
                "discrimination": model.discrimination.detach().cpu().numpy(),
                "guessing": model.guessing.detach().cpu().numpy(),
            })
        elif name.startswith("fm"):
            save_subject_params(name, model.U.detach().cpu().numpy(), subject_names)
            save_item_params(name, item_keys, {
                "intercept_Z": model.Z.detach().cpu().numpy(),
                "loading_V": model.V.detach().cpu().numpy(),
            })
        else:
            raise ValueError(f"No persistence configured for model {name!r}")
        save_predictions(name, stats["p_hat"], s_idx, i_idx, y)

    def fit_and_persist(name, ctor):
        torch.manual_seed(0)
        model = ctor()
        stats = fit_one(
            name, model, s_train, i_train, y_train,
            s_test=s_test, i_test=i_test, y_test=y_test,
        )
        _persist(name, stats, model)
        return model

    fit_and_persist("rasch",   lambda: Rasch(n_subjects, n_items))
    fit_and_persist("twopl",   lambda: TwoPL(n_subjects, n_items))
    fit_and_persist("threepl", lambda: ThreePL(n_subjects, n_items))
    fit_and_persist("fm1",     lambda: LogisticFM(n_subjects, n_items, n_factors=1))
    fit_and_persist("fm2",     lambda: LogisticFM(n_subjects, n_items, n_factors=2))
    fit_and_persist("fm3",     lambda: LogisticFM(n_subjects, n_items, n_factors=3))

    # Per-tier fit_table.csv
    table_path = os.path.join(out_dir, "fit_table.csv")
    with open(table_path, "w") as f:
        f.write("model,n_params,n_obs,log_likelihood,AIC,BIC,"
                "test_log_likelihood,test_nll_per_obs,seconds,final_loss\n")
        for name, st in fits.items():
            tll = st["test_log_likelihood"]
            tnll = st["test_nll_per_obs"]
            tll_s = "" if tll is None else f"{tll:.4f}"
            tnll_s = "" if tnll is None else f"{tnll:.6f}"
            f.write(
                f"{name},{st['n_params']},{st['n_obs']},"
                f"{st['log_likelihood']:.4f},{st['AIC']:.4f},{st['BIC']:.4f},"
                f"{tll_s},{tnll_s},"
                f"{st['seconds']:.2f},"
                f"{st['final_loss']:.6f}\n"
            )
    print(f"  ✓ wrote {os.path.relpath(table_path, REPO_ROOT)}")

    fit_summary = {
        "tier": tier,
        "corpus": summary,
        "subject_names": subject_names,
        "fits": {
            name: {k: v for k, v in st.items() if k != "p_hat"}
            for name, st in fits.items()
        },
        "methodology_notes": METHODOLOGY_DOC,
        "bifactor_skipped": True,
        "bifactor_skipped_reason": (
            "Bifactor partitions item variance by anatomical group; within a "
            "single prevalence tier we deliberately do not construct "
            "tier-local item_groups from ANATOMICAL_GROUP. See "
            "stratified_irt_methodology.md §5."
        ),
    }
    summary_path = os.path.join(out_dir, "fit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(fit_summary, f, indent=2)
    print(f"  ✓ wrote {os.path.relpath(summary_path, REPO_ROOT)}")

    # Per-tier ranked table
    print(f"\n  Models ranked by held-out NLL/obs (tier={tier}):")
    ranked = sorted(
        fits.items(),
        key=lambda kv: (kv[1]["test_nll_per_obs"] if kv[1]["test_nll_per_obs"] is not None else float("inf")),
    )
    for name, st in ranked:
        ts = st["test_nll_per_obs"]
        ts_fmt = f"{ts:.4f}" if ts is not None else "  n/a "
        print(f"    {name:<10s}  test-NLL/obs={ts_fmt}   "
              f"BIC={st['BIC']:>11,.1f}   LL_train={st['log_likelihood']:>11,.1f}   "
              f"params={st['n_params']:>6,}")

    return [
        {
            "tier": tier,
            "model": name,
            "n_params": st["n_params"],
            "n_obs": st["n_obs"],
            "log_likelihood": st["log_likelihood"],
            "AIC": st["AIC"],
            "BIC": st["BIC"],
            "test_log_likelihood": st["test_log_likelihood"],
            "test_nll_per_obs": st["test_nll_per_obs"],
            "seconds": st["seconds"],
        }
        for name, st in fits.items()
    ]


def write_combined_summary(rows):
    os.makedirs(BY_TIER_OUT_ROOT, exist_ok=True)
    path = os.path.join(BY_TIER_OUT_ROOT, "summary.csv")
    cols = [
        "tier", "model", "n_params", "n_obs",
        "log_likelihood", "AIC", "BIC",
        "test_log_likelihood", "test_nll_per_obs", "seconds",
    ]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.6f}")
                else:
                    cells.append(str(v))
            f.write(",".join(cells) + "\n")
    print(f"\n✓ wrote combined summary: {os.path.relpath(path, REPO_ROOT)} "
          f"({len(rows)} rows)")


def main():
    print("──  Discovering inference sources  ──")
    source_paths = discover_sources(REPO_ROOT)
    for p in source_paths:
        print(f"  • {os.path.relpath(p, REPO_ROOT)}")
    if not source_paths:
        raise SystemExit("No inference output files found under outputs/.")

    print("\n──  Loading observations  ──")
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations available.")
    print(f"  total observations: {len(obs):,}")

    print("\n──  Partitioning by prevalence tier  ──")
    buckets = partition_by_tier(obs)
    for tier in TIERS:
        n = len(buckets.get(tier, []))
        print(f"  tier {tier:<5s}: {n:>9,} observations")

    print("\n──  Persisting tier-separated observations  ──")
    write_tier_observations(buckets, source_paths)

    print("\n──  Fitting per-tier IRT suite (rasch, twopl, threepl, fm1, fm2, fm3) ──")
    all_rows: list[dict] = []
    for tier in TIERS:
        rows = fit_tier(tier, buckets.get(tier, []))
        all_rows.extend(rows)

    write_combined_summary(all_rows)


if __name__ == "__main__":
    main()
