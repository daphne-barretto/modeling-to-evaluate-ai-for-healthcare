"""Fit a full panel of measurement models on the merged response matrix.

Runs:
    rasch         — 1PL
    twopl         — 2PL
    threepl       — 3PL (guessing parameter)
    fm1           — 1-factor logistic FM
    fm2           — 2-factor logistic FM
    fm3           — 3-factor logistic FM
    bifactor      — general factor + anatomical-group factors

Outputs (under ``outputs/irt/``):
    abilities_{model}.csv       — subject ability (1-D Rasch/2PL/3PL or multi-D for FM)
    item_params_{model}.csv     — item difficulty / discrimination / guessing / loadings
    predictions_{model}.npz     — per-observation P_hat (for diagnostics)
    fit_table.csv               — model | n_params | log_likelihood | AIC | BIC | seconds
    fit_summary.json            — corpus summary + per-model headline stats
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_loader import PATHOLOGIES, load_all  # noqa: E402
from analysis.item_metadata import ANATOMICAL_GROUP  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")

MAX_EPOCHS = int(os.environ.get("IRT_MAX_EPOCHS", "1500"))
LR = float(os.environ.get("IRT_LR", "0.05"))


def build_index(observations):
    subject_to_id: dict[str, int] = {}
    item_to_id: dict[tuple[str, str], int] = {}
    subject_names: list[str] = []
    item_keys: list[tuple[str, str]] = []
    for o in observations:
        if o.subject not in subject_to_id:
            subject_to_id[o.subject] = len(subject_names)
            subject_names.append(o.subject)
        key = (o.image_path, o.pathology)
        if key not in item_to_id:
            item_to_id[key] = len(item_keys)
            item_keys.append(key)
    return subject_to_id, item_to_id, subject_names, item_keys


def to_tensors(observations, subject_to_id, item_to_id):
    s = torch.tensor([subject_to_id[o.subject] for o in observations], dtype=torch.long)
    i = torch.tensor([item_to_id[(o.image_path, o.pathology)] for o in observations], dtype=torch.long)
    y = torch.tensor([o.response for o in observations], dtype=torch.float32)
    return s, i, y


def train_test_split(s_idx, i_idx, y, test_frac=0.1, seed=0):
    """Hold out a random subset of observations as a test set.

    Items that occur ONLY in the test split would be unidentifiable at
    eval time, so we drop those (rare in practice). Returns
    (s_train, i_train, y_train, s_test, i_test, y_test).
    """
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_items = set(i_idx[train_idx].tolist())
    train_subjects = set(s_idx[train_idx].tolist())
    keep_mask = torch.zeros(n_test, dtype=torch.bool)
    for k in range(n_test):
        if (int(i_idx[test_idx[k]]) in train_items
                and int(s_idx[test_idx[k]]) in train_subjects):
            keep_mask[k] = True
    test_idx_kept = test_idx[keep_mask]
    n_dropped = n_test - keep_mask.sum().item()
    print(f"  train/test split: {len(train_idx):,} train, "
          f"{len(test_idx_kept):,} test ({n_dropped:,} dropped — "
          "item or subject unseen in train)")
    return (
        s_idx[train_idx], i_idx[train_idx], y[train_idx],
        s_idx[test_idx_kept], i_idx[test_idx_kept], y[test_idx_kept],
    )


def log_likelihood(model, s_idx, i_idx, y):
    model.eval()
    with torch.no_grad():
        p = model.predict({"subject_idx": s_idx, "item_idx": i_idx}).clamp(1e-7, 1 - 1e-7)
        ll = (y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum().item()
    return ll, p.cpu().numpy()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_subject_params(name, abilities, subject_names):
    abilities = np.asarray(abilities)
    path = os.path.join(OUT_DIR, f"abilities_{name}.csv")
    with open(path, "w") as f:
        if abilities.ndim == 1:
            f.write("subject,ability\n")
            for s, a in zip(subject_names, abilities.tolist()):
                f.write(f"{s},{a:.6f}\n")
        else:
            k = abilities.shape[1]
            f.write("subject," + ",".join(f"ability_f{i}" for i in range(k)) + "\n")
            for s, a in zip(subject_names, abilities.tolist()):
                f.write(f"{s}," + ",".join(f"{v:.6f}" for v in a) + "\n")
    return path


def save_item_params(name, item_keys, params: dict):
    """``params`` is a dict mapping column-name -> 1-D array (len n_items),
    or, for K-dim factor loadings, name -> 2-D array (n_items, K)."""
    path = os.path.join(OUT_DIR, f"item_params_{name}.csv")
    columns = ["item_idx", "image_path", "pathology"]
    scalar_cols = []
    loading_cols = []
    for col, arr in params.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            columns.append(col)
            scalar_cols.append((col, arr))
        else:
            for k in range(arr.shape[1]):
                columns.append(f"{col}_{k}")
                loading_cols.append((f"{col}_{k}", arr[:, k]))
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for idx, (img, p) in enumerate(item_keys):
            cells = [str(idx), img, p]
            for col, arr in scalar_cols:
                cells.append(f"{float(arr[idx]):.6f}")
            for col, arr in loading_cols:
                cells.append(f"{float(arr[idx]):.6f}")
            f.write(",".join(cells) + "\n")
    return path


def save_predictions(name, p_hat, s_idx, i_idx, y):
    path = os.path.join(OUT_DIR, f"predictions_{name}.npz")
    np.savez_compressed(
        path,
        p_hat=p_hat.astype(np.float32),
        subject_idx=s_idx.cpu().numpy().astype(np.int32),
        item_idx=i_idx.cpu().numpy().astype(np.int32),
        response=y.cpu().numpy().astype(np.int8),
    )
    return path


def fit_one(model_name, model, s_idx, i_idx, y, *, lr=LR, max_epochs=MAX_EPOCHS,
            s_test=None, i_test=None, y_test=None):
    n_obs = len(y)
    n_params = count_params(model)
    print(f"\n──  Fitting {model_name}: {n_params:,} params, {n_obs:,} obs  ──")
    from torch_measure.fitting import mle_fit

    t0 = time.time()
    history = mle_fit(model, s_idx, i_idx, y, max_epochs=max_epochs, lr=lr, verbose=False)
    seconds = time.time() - t0
    ll, p_hat = log_likelihood(model, s_idx, i_idx, y)
    aic = 2 * n_params - 2 * ll
    bic = math.log(n_obs) * n_params - 2 * ll
    test_ll = None
    test_nll_per_obs = None
    if s_test is not None and len(s_test) > 0:
        test_ll, _ = log_likelihood(model, s_test, i_test, y_test)
        test_nll_per_obs = -test_ll / len(y_test)
    print(f"  train LL: {ll:,.1f}   AIC: {aic:,.1f}   BIC: {bic:,.1f}   "
          f"test LL: {test_ll if test_ll is None else f'{test_ll:,.1f}'}   "
          f"test NLL/obs: {test_nll_per_obs if test_nll_per_obs is None else f'{test_nll_per_obs:.4f}'}   "
          f"{seconds:.1f}s")
    return {
        "model": model_name,
        "n_params": n_params,
        "n_obs": n_obs,
        "log_likelihood": ll,
        "AIC": aic,
        "BIC": bic,
        "test_log_likelihood": test_ll,
        "test_nll_per_obs": test_nll_per_obs,
        "seconds": seconds,
        "final_loss": history["losses"][-1] if history["losses"] else None,
        "p_hat": p_hat,
    }


def summarize_corpus(observations, subject_names, item_keys):
    per_subject = Counter(o.subject for o in observations)
    per_pathology = Counter(o.pathology for o in observations)
    acc_correct = defaultdict(int)
    acc_total = defaultdict(int)
    for o in observations:
        acc_total[o.subject] += 1
        acc_correct[o.subject] += o.response
    return {
        "n_subjects": len(subject_names),
        "n_items": len(item_keys),
        "n_observations": len(observations),
        "obs_per_subject": dict(per_subject),
        "obs_per_pathology": dict(per_pathology),
        "raw_accuracy_per_subject": {
            s: acc_correct[s] / acc_total[s] for s in subject_names
        },
    }


def main():
    print("──  Loading observations  ──")
    obs = load_all(REPO_ROOT)
    if not obs:
        raise SystemExit("No observations available.")
    subject_to_id, item_to_id, subject_names, item_keys = build_index(obs)
    s_idx, i_idx, y = to_tensors(obs, subject_to_id, item_to_id)
    n_subjects = len(subject_names)
    n_items = len(item_keys)
    os.makedirs(OUT_DIR, exist_ok=True)

    s_train, i_train, y_train, s_test, i_test, y_test = train_test_split(
        s_idx, i_idx, y, test_frac=0.10, seed=0,
    )

    summary = summarize_corpus(obs, subject_names, item_keys)
    print(f"\nCorpus: {n_subjects} subjects × {n_items:,} items × {len(obs):,} obs")
    print("Raw accuracy per subject:")
    for s, a in sorted(summary["raw_accuracy_per_subject"].items(), key=lambda x: -x[1]):
        print(f"  {s:<28s} {a:.4f}")

    fits = {}

    from torch_measure.models import (
        Bifactor, LogisticFM, Rasch, ThreePL, TwoPL,
    )

    def fit_and_persist(name, ctor, save_fn):
        torch.manual_seed(0)
        model = ctor()
        stats = fit_one(
            name, model, s_train, i_train, y_train,
            s_test=s_test, i_test=i_test, y_test=y_test,
        )
        fits[name] = stats
        save_fn(name, model)
        return model

    # --- Saving helpers ---
    def save_rasch(name, m: Rasch):
        save_subject_params(name, m.ability.detach().cpu().numpy(), subject_names)
        save_item_params(name, item_keys, {"difficulty": m.difficulty.detach().cpu().numpy()})
        save_predictions(name, fits[name]["p_hat"], s_idx, i_idx, y)

    def save_twopl(name, m: TwoPL):
        save_subject_params(name, m.ability.detach().cpu().numpy(), subject_names)
        save_item_params(name, item_keys, {
            "difficulty": m.difficulty.detach().cpu().numpy(),
            "discrimination": m.discrimination.detach().cpu().numpy(),
        })
        save_predictions(name, fits[name]["p_hat"], s_idx, i_idx, y)

    def save_threepl(name, m: ThreePL):
        save_subject_params(name, m.ability.detach().cpu().numpy(), subject_names)
        save_item_params(name, item_keys, {
            "difficulty": m.difficulty.detach().cpu().numpy(),
            "discrimination": m.discrimination.detach().cpu().numpy(),
            "guessing": m.guessing.detach().cpu().numpy(),
        })
        save_predictions(name, fits[name]["p_hat"], s_idx, i_idx, y)

    def save_fm(name, m: LogisticFM):
        save_subject_params(name, m.U.detach().cpu().numpy(), subject_names)
        save_item_params(name, item_keys, {
            "intercept_Z": m.Z.detach().cpu().numpy(),
            "loading_V": m.V.detach().cpu().numpy(),
        })
        save_predictions(name, fits[name]["p_hat"], s_idx, i_idx, y)

    def save_bifactor(name, m: Bifactor):
        save_subject_params(name, m.general_ability.detach().cpu().numpy(), subject_names)
        # Save general/group loadings as item params
        save_item_params(name, item_keys, {
            "intercept_Z": m.Z.detach().cpu().numpy(),
            "general_loading": m.general_loading.detach().cpu().numpy(),
            "group_loading": m.group_loading.detach().cpu().numpy(),
            "group_idx": m.item_groups.detach().cpu().numpy().astype(np.float32),
        })
        # Also save group abilities side-by-side
        group_path = os.path.join(OUT_DIR, f"group_abilities_{name}.csv")
        with open(group_path, "w") as f:
            n_groups = m.group_ability.shape[1]
            f.write("subject," + ",".join(f"group_{i}" for i in range(n_groups)) + "\n")
            for s, a in zip(subject_names, m.group_ability.detach().cpu().tolist()):
                f.write(f"{s}," + ",".join(f"{v:.6f}" for v in a) + "\n")
        save_predictions(name, fits[name]["p_hat"], s_idx, i_idx, y)

    fit_and_persist("rasch",    lambda: Rasch(n_subjects, n_items),    save_rasch)
    fit_and_persist("twopl",    lambda: TwoPL(n_subjects, n_items),    save_twopl)
    fit_and_persist("threepl",  lambda: ThreePL(n_subjects, n_items),  save_threepl)
    fit_and_persist("fm1",      lambda: LogisticFM(n_subjects, n_items, n_factors=1), save_fm)
    fit_and_persist("fm2",      lambda: LogisticFM(n_subjects, n_items, n_factors=2), save_fm)
    fit_and_persist("fm3",      lambda: LogisticFM(n_subjects, n_items, n_factors=3), save_fm)

    anat_groups = sorted(set(ANATOMICAL_GROUP.values()))
    group_to_id = {g: i for i, g in enumerate(anat_groups)}
    item_groups = torch.tensor(
        [group_to_id[ANATOMICAL_GROUP[p]] for (_img, p) in item_keys], dtype=torch.long
    )
    fit_and_persist(
        "bifactor",
        lambda: Bifactor(n_subjects, n_items, len(anat_groups), item_groups),
        save_bifactor,
    )

    # --- master fit table ---
    table_path = os.path.join(OUT_DIR, "fit_table.csv")
    with open(table_path, "w") as f:
        f.write("model,n_params,n_obs,log_likelihood,AIC,BIC,test_log_likelihood,test_nll_per_obs,seconds,final_loss\n")
        for name, st in fits.items():
            tll = st["test_log_likelihood"]
            tnll = st["test_nll_per_obs"]
            tll_s = "" if tll is None else f"{tll:.4f}"
            tnll_s = "" if tnll is None else f"{tnll:.6f}"
            f.write(
                f"{name},{st['n_params']},{st['n_obs']},"
                f"{st['log_likelihood']:.4f},{st['AIC']:.4f},{st['BIC']:.4f},"
                f"{tll_s},{tnll_s},"
                f"{st['seconds']:.2f},{st['final_loss']:.6f}\n"
            )
    print(f"\n✓ Wrote master fit table: outputs/irt/fit_table.csv")

    fit_summary = {
        "corpus": summary,
        "subject_names": subject_names,
        "fits": {
            name: {k: v for k, v in st.items() if k != "p_hat"}
            for name, st in fits.items()
        },
        "anatomical_groups": anat_groups,
    }
    with open(os.path.join(OUT_DIR, "fit_summary.json"), "w") as f:
        json.dump(fit_summary, f, indent=2)
    print(f"✓ Wrote summary: outputs/irt/fit_summary.json")

    # --- ranked table by test NLL (held-out) — much more honest than BIC at this density ---
    print(f"\n──  Models ranked by held-out NLL/obs (lower = better)  ──")
    ranked = sorted(fits.items(), key=lambda kv: kv[1]["test_nll_per_obs"] if kv[1]["test_nll_per_obs"] is not None else float("inf"))
    for name, st in ranked:
        ts = st["test_nll_per_obs"]
        ts_fmt = f"{ts:.4f}" if ts is not None else "  n/a "
        print(f"  {name:<10s}  test-NLL/obs={ts_fmt}   "
              f"BIC={st['BIC']:>11,.1f}   LL_train={st['log_likelihood']:>11,.1f}   "
              f"params={st['n_params']:>6,}")


if __name__ == "__main__":
    main()
