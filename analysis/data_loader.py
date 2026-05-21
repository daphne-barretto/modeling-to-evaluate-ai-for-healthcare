"""Load per-model inference outputs into a unified long-form IRT dataset.

Each observation is one (subject, item) cell with a binary `response`
(1 = model got the (image, pathology) cell correct, 0 = wrong). Missing
cells (refusal, content filter, exception, unanswered pathology) are
*omitted*, not coded as 0 — IRT treats absent rows as MAR.

Supported sources
-----------------

1. **Daphne GPT-style** outputs (``outputs/daphne_gpt5_outputs.json``,
   ``outputs/daphne_gpt4o_outputs.json``). Top-level dict keyed by item
   path; each value has ``deployment``, ``per_label_correct``,
   ``predictions`` (or ``correct=None`` for refusals).

2. **Izhan Qwen-style** outputs (``qwen3b_results_10000.json``,
   ``qwen7b_results_10000.json``). Top-level list; each entry has
   ``image_path``, ``subject``, plus ``{pathology}__correct`` for each
   of the 14 pathologies (None for unanswered).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class Observation:
    subject: str
    image_path: str
    pathology: str
    response: int


def _load_daphne_outputs(path: str) -> list[Observation]:
    with open(path) as f:
        data = json.load(f)
    obs: list[Observation] = []
    for image_path, entry in data.items():
        if entry is None:
            continue
        if entry.get("error") or entry.get("missing_reason"):
            continue
        if entry.get("predictions") in (None, {}) and entry.get("raw") in (None, ""):
            continue
        per_label = entry.get("per_label_correct") or {}
        subject = entry.get("deployment") or "unknown"
        for p in PATHOLOGIES:
            v = per_label.get(p)
            if v is None:
                continue
            obs.append(Observation(subject, image_path, p, int(v)))
    return obs


def _load_izhan_qwen_outputs(path: str) -> list[Observation]:
    with open(path) as f:
        data = json.load(f)
    obs: list[Observation] = []
    for entry in data:
        image_path = entry.get("image_path")
        subject = entry.get("subject") or "unknown"
        if not image_path:
            continue
        for p in PATHOLOGIES:
            v = entry.get(f"{p}__correct")
            if v is None:
                continue
            obs.append(Observation(subject, image_path, p, int(v)))
    return obs


# Source filename → loader. The loader is selected by checking which
# top-level structure the file has, not by name pattern, so it works on
# any future drop-in file.
def load_source(path: str) -> list[Observation]:
    with open(path) as f:
        head = f.read(64).lstrip()
    if head.startswith("["):
        return _load_izhan_qwen_outputs(path)
    if head.startswith("{"):
        return _load_daphne_outputs(path)
    raise ValueError(f"Unrecognised top-level JSON in {path}: {head[:32]!r}")


def discover_sources(repo_root: str) -> list[str]:
    """Return paths to all inference output files we know how to load."""
    paths: list[str] = []
    candidates = [
        # Daphne validation outputs
        os.path.join(repo_root, "outputs/daphne_gpt5_outputs.json"),
        os.path.join(repo_root, "outputs/daphne_gpt4o_outputs.json"),
        # Daphne 10K train1 outputs (will be created once Modal run finishes)
        os.path.join(repo_root, "outputs/daphne_gpt5_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_gpt4o_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_gpt5_train1_15000.json"),
        os.path.join(repo_root, "outputs/daphne_gpt4o_train1_15000.json"),
        # Izhan Qwen outputs (top-level list format)
        os.path.join(repo_root, "qwen3b_results_10000.json"),
        os.path.join(repo_root, "qwen7b_results_10000.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            paths.append(p)
    return paths


def load_all(repo_root: str) -> list[Observation]:
    obs: list[Observation] = []
    for p in discover_sources(repo_root):
        n_before = len(obs)
        obs.extend(load_source(p))
        print(f"  loaded {len(obs) - n_before:>7d} obs from {os.path.relpath(p, repo_root)}")
    return obs
