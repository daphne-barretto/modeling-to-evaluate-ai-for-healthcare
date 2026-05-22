"""Cold-start amortized IRT.

Trains an MLP to predict (b_j, a_j) — the 2PL item difficulty and
discrimination — from per-item metadata (image age/sex/view/AP-PA
plus pathology anatomical group and prevalence tier). Holds out 20%
of *unique images* (so all 14 pathology-rows for those images go to
the test split) and reports cold-start NLL on the test items using
the trained amortizer + frozen subject abilities θ̂.

Features
--------
For each item j = (image i, pathology p), the input is the
concatenation of:
  - one-hot age bin (5 dims)            [<40, 40-59, 60-79, 80+, unknown]
  - one-hot sex (3 dims)                [Male, Female, unknown]
  - one-hot view (3 dims)               [Frontal, Lateral, unknown]
  - one-hot AP/PA (4 dims)              [AP, PA, LL/RL, unknown]
  - one-hot anatomical group (4 dims)   [cardiac, pulmonary, pleural, other]
  - one-hot prevalence tier (4 dims)    [high, mid, low, na]

Outputs (under ``outputs/irt/``):
  amortized_irt.json    — summary metrics
  amortized_pred.csv    — per-item predicted (b̂, â) vs. fitted (b, a) on test
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.item_metadata import (  # noqa: E402
    ANATOMICAL_GROUP,
    PREVALENCE_TIER,
    age_bin,
    _load_csv_index,
    default_csv_path,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRT_DIR = os.path.join(REPO_ROOT, "outputs", "irt")
OUT_JSON = os.path.join(IRT_DIR, "amortized_irt.json")
OUT_CSV = os.path.join(IRT_DIR, "amortized_pred.csv")

TEST_FRAC = 0.20
HIDDEN = 64
EPOCHS = 200
LR = 5e-3
BATCH = 4096
SEED = 0

AGE_BINS = ["<40", "40-59", "60-79", "80+", "unknown"]
SEXES = ["Male", "Female", "unknown"]
VIEWS = ["Frontal", "Lateral", "unknown"]
AP_PAS = ["AP", "PA", "LL", "unknown"]
GROUPS = ["cardiac", "pulmonary", "pleural", "other"]
TIERS = ["high", "mid", "low", "na"]


def one_hot(value: str, categories: list[str]) -> np.ndarray:
    v = np.zeros(len(categories), dtype=np.float32)
    if value in categories:
        v[categories.index(value)] = 1.0
    else:
        v[categories.index("unknown") if "unknown" in categories else 0] = 1.0
    return v


def featurize(img_meta, pathology: str) -> np.ndarray:
    sex = img_meta.sex if img_meta else "unknown"
    age = img_meta.age_bin if img_meta else "unknown"
    view = img_meta.view if img_meta else "unknown"
    ap = img_meta.ap_pa if img_meta else "unknown"
    group = ANATOMICAL_GROUP.get(pathology, "other")
    tier = PREVALENCE_TIER.get(pathology, "na")
    return np.concatenate([
        one_hot(age, AGE_BINS),
        one_hot(sex, SEXES),
        one_hot(view, VIEWS),
        one_hot(ap, AP_PAS),
        one_hot(group, GROUPS),
        one_hot(tier, TIERS),
    ])


def load_items() -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    with open(os.path.join(IRT_DIR, "item_params_twopl.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "item_idx": int(row["item_idx"]),
                "image_path": row["image_path"],
                "pathology": row["pathology"],
                "difficulty": float(row["difficulty"]),
                "discrimination": float(row["discrimination"]),
            })
    return rows, None


def load_abilities() -> tuple[list[str], np.ndarray]:
    names = []
    vals = []
    with open(os.path.join(IRT_DIR, "abilities_twopl.csv")) as f:
        next(f)
        for ln in f:
            n, v = ln.strip().split(",")
            names.append(n)
            vals.append(float(v))
    return names, np.array(vals, dtype=np.float32)


def load_responses():
    data = np.load(os.path.join(IRT_DIR, "predictions_twopl.npz"))
    return (
        data["subject_idx"].astype(np.int32),
        data["item_idx"].astype(np.int32),
        data["response"].astype(np.int8),
    )


class Amortizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # outputs (b̂, raw â). â = softplus(raw â).
        )

    def forward(self, x):
        out = self.net(x)
        b = out[..., 0]
        a = F.softplus(out[..., 1]) + 1e-3
        return b, a


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("[amort] loading items + metadata + responses ...", flush=True)
    items, _ = load_items()
    n_items = len(items)
    print(f"[amort] {n_items:,} items", flush=True)

    meta = _load_csv_index(default_csv_path(REPO_ROOT))
    print(f"[amort] loaded metadata for {len(meta):,} images", flush=True)

    X = np.zeros((n_items, 23), dtype=np.float32)
    b_tgt = np.zeros(n_items, dtype=np.float32)
    a_tgt = np.zeros(n_items, dtype=np.float32)
    image_idx = np.zeros(n_items, dtype=np.int64)
    img_to_id: dict[str, int] = {}
    for j, it in enumerate(items):
        m = meta.get(it["image_path"])
        X[j] = featurize(m, it["pathology"])
        b_tgt[j] = it["difficulty"]
        a_tgt[j] = it["discrimination"]
        if it["image_path"] not in img_to_id:
            img_to_id[it["image_path"]] = len(img_to_id)
        image_idx[j] = img_to_id[it["image_path"]]
    n_images = len(img_to_id)
    print(f"[amort] {n_images:,} unique images, feature dim={X.shape[1]}", flush=True)

    # Hold out 20% of images (so all 14 pathologies per image are in test).
    n_test_img = int(round(n_images * TEST_FRAC))
    perm = rng.permutation(n_images)
    test_imgs = set(perm[:n_test_img].tolist())
    train_mask = np.array([image_idx[j] not in test_imgs for j in range(n_items)])
    test_mask = ~train_mask
    print(
        f"[amort] split: {train_mask.sum():,} train items ({n_images - n_test_img:,} images), "
        f"{test_mask.sum():,} test items ({n_test_img:,} images)",
        flush=True,
    )

    # Robustify the discrimination target: 2PL MLE produces some extreme |a|>>1
    # (items where subjects all gave same response). Clamp to [-5, 5] so the
    # MLP's softplus head can express the target range without saturating.
    a_tgt_clipped = np.clip(a_tgt, 0.05, 5.0)
    b_tgt_clipped = np.clip(b_tgt, -8.0, 8.0)

    X_train = torch.from_numpy(X[train_mask])
    b_train = torch.from_numpy(b_tgt_clipped[train_mask])
    a_train = torch.from_numpy(a_tgt_clipped[train_mask])
    X_test = torch.from_numpy(X[test_mask])
    b_test_true = torch.from_numpy(b_tgt_clipped[test_mask])
    a_test_true = torch.from_numpy(a_tgt_clipped[test_mask])

    model = Amortizer(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("[amort] training amortizer ...", flush=True)
    n_train = X_train.shape[0]
    for ep in range(EPOCHS):
        idx = torch.randperm(n_train)
        ep_loss = 0.0
        n_b = 0
        for i in range(0, n_train, BATCH):
            sl = idx[i:i + BATCH]
            model.train()
            opt.zero_grad()
            b_hat, a_hat = model(X_train[sl])
            loss = ((b_hat - b_train[sl]) ** 2).mean() + ((a_hat - a_train[sl]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * sl.numel()
            n_b += sl.numel()
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep + 1:3d}/{EPOCHS}  loss = {ep_loss / n_b:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        b_hat_test, a_hat_test = model(X_test)

    b_mse = float(((b_hat_test - b_test_true) ** 2).mean())
    a_mse = float(((a_hat_test - a_test_true) ** 2).mean())
    b_r2 = 1.0 - b_mse / float(b_test_true.var(unbiased=False).item() + 1e-9)
    a_r2 = 1.0 - a_mse / float(a_test_true.var(unbiased=False).item() + 1e-9)
    print(
        f"[amort] held-out item param recovery:\n"
        f"  difficulty:    MSE={b_mse:.4f}  R²={b_r2:+.4f}\n"
        f"  discrimination: MSE={a_mse:.4f}  R²={a_r2:+.4f}",
        flush=True,
    )

    # Now compute cold-start NLL on test items using subject θ̂.
    subjects, theta = load_abilities()
    print(
        f"[amort] {len(subjects)} subjects, θ̂ = "
        + ", ".join(f"{s}={t:+.2f}" for s, t in zip(subjects, theta)),
        flush=True,
    )
    s_idx_all, i_idx_all, y_all = load_responses()

    test_item_global = np.where(test_mask)[0]
    test_set = set(test_item_global.tolist())

    obs_mask = np.array([int(j) in test_set for j in i_idx_all])
    s_obs = s_idx_all[obs_mask]
    i_obs = i_idx_all[obs_mask]
    y_obs = y_all[obs_mask].astype(np.float32)
    print(f"[amort] {obs_mask.sum():,} held-out observations across {len(subjects)} subjects",
          flush=True)

    test_idx_to_local = {int(g): k for k, g in enumerate(test_item_global)}
    obs_local = np.array([test_idx_to_local[int(g)] for g in i_obs], dtype=np.int64)

    b_hat_np = b_hat_test.cpu().numpy()
    a_hat_np = a_hat_test.cpu().numpy()
    b_true_np = b_test_true.cpu().numpy()
    a_true_np = a_test_true.cpu().numpy()
    theta_np = np.array(theta, dtype=np.float64)

    def nll(b_arr: np.ndarray, a_arr: np.ndarray) -> float:
        b_obs = b_arr[obs_local]
        a_obs = a_arr[obs_local]
        th_obs = theta_np[s_obs]
        z = a_obs * (th_obs - b_obs)
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y_obs * np.log(p) + (1 - y_obs) * np.log(1 - p)).mean())

    nll_amort = nll(b_hat_np, a_hat_np)
    nll_oracle = nll(b_true_np, a_true_np)
    # Naive baseline: constant (b=0, a=1) — i.e. Rasch with mean difficulty.
    nll_null = nll(np.zeros_like(b_hat_np), np.ones_like(a_hat_np))

    print(
        f"[amort] cold-start NLL per held-out observation:\n"
        f"  amortizer:    {nll_amort:.4f}\n"
        f"  oracle 2PL:   {nll_oracle:.4f}\n"
        f"  null (Rasch): {nll_null:.4f}\n",
        flush=True,
    )

    out = {
        "n_items_train": int(train_mask.sum()),
        "n_items_test": int(test_mask.sum()),
        "n_images_train": int(n_images - n_test_img),
        "n_images_test": int(n_test_img),
        "n_test_obs": int(obs_mask.sum()),
        "feature_dim": int(X.shape[1]),
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "lr": LR,
        "difficulty_mse_test": b_mse,
        "difficulty_r2_test": b_r2,
        "discrimination_mse_test": a_mse,
        "discrimination_r2_test": a_r2,
        "nll_per_obs_test": {
            "amortizer": nll_amort,
            "oracle_2pl": nll_oracle,
            "null_rasch": nll_null,
        },
        "subjects": list(subjects),
        "theta_hat": [float(t) for t in theta_np.tolist()],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[amort] wrote {OUT_JSON}", flush=True)

    with open(OUT_CSV, "w") as f:
        w = csv.writer(f)
        w.writerow(["item_idx", "image_path", "pathology",
                    "b_true", "a_true", "b_hat", "a_hat"])
        global_indices = test_item_global.tolist()
        for k, g in enumerate(global_indices):
            it = items[g]
            w.writerow([
                it["item_idx"], it["image_path"], it["pathology"],
                f"{b_true_np[k]:+.6f}", f"{a_true_np[k]:+.6f}",
                f"{b_hat_np[k]:+.6f}", f"{a_hat_np[k]:+.6f}",
            ])
    print(f"[amort] wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
