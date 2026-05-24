"""Load per-model inference outputs into a unified long-form IRT dataset.

Each observation is one (subject, item) cell with a binary `response`
(1 = model got the (image, pathology) cell correct, 0 = wrong). Missing
cells (refusal, content filter, exception, unanswered pathology) are
*omitted*, not coded as 0 — IRT treats absent rows as MAR.

Supported sources
-----------------

1. **Daphne GPT-style** outputs (``outputs/daphne_gpt5_outputs.json``,
   ``outputs/daphne_gpt4o_outputs.json``, and Shannon's
   ``outputs/pixtral_outputs.json`` / ``outputs/llama_vision_outputs.json``).
   Top-level dict keyed by item path; each value has ``deployment``,
   ``per_label_correct``, ``predictions`` (or ``correct=None`` for refusals).

2. **Izhan Qwen-style** outputs (``qwen3b_results_10000.json``,
   ``qwen7b_results_10000.json``, ``outputs/qwen{3b,7b}_combined.json``).
   Top-level list; each entry has ``image_path``, ``subject``, plus
   ``{pathology}__correct`` for each of the 14 pathologies (None for
   unanswered).

All loaders normalise ``image_path`` to the canonical CheXpert form
``CheXpert-v1.0/train/patientXXXXX/studyN/viewN_orientation.jpg`` so that
items align across subjects regardless of the prefix used upstream.
"""

from __future__ import annotations

import json
import os
import re
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


_CANONICAL_PREFIX = "CheXpert-v1.0/train/"
_PATIENT_RE = re.compile(r"(patient\d+/study\d+/view\d+_[A-Za-z]+\.jpg)")

# Map raw HuggingFace / Azure deployment strings to the friendlier
# subject IDs we use throughout the manuscript and figures. Anything
# not in the map passes through unchanged so legacy IDs keep working.
_SUBJECT_ALIASES = {
    "pixtral-12b-2409": "Pixtral-12B",
    "llama-3.2-vision-11b": "Llama-3.2-Vision-11B",
}


def _normalize_subject(name: str) -> str:
    return _SUBJECT_ALIASES.get(name, name)


def _normalize_image_path(p: str) -> str:
    """Coerce any CheXpert path variant to ``CheXpert-v1.0/train/patientXXXXX/...``.

    Handles three observed forms:
      - already canonical: ``CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg``
      - Shannon's Azure: ``CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)/patient00001/...``
      - Izhan's combined: ``patient00001/study1/view1_frontal.jpg``
    Unknown formats pass through unchanged so the caller can decide.
    """
    if not p:
        return p
    m = _PATIENT_RE.search(p)
    if m:
        return _CANONICAL_PREFIX + m.group(1)
    return p


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
        subject = _normalize_subject(entry.get("deployment") or "unknown")
        canonical = _normalize_image_path(image_path)
        for p in PATHOLOGIES:
            v = per_label.get(p)
            if v is None:
                continue
            obs.append(Observation(subject, canonical, p, int(v)))
    return obs


def _load_izhan_qwen_outputs(path: str) -> list[Observation]:
    with open(path) as f:
        data = json.load(f)
    obs: list[Observation] = []
    for entry in data:
        image_path = entry.get("image_path")
        subject = _normalize_subject(entry.get("subject") or "unknown")
        if not image_path:
            continue
        canonical = _normalize_image_path(image_path)
        for p in PATHOLOGIES:
            v = entry.get(f"{p}__correct")
            if v is None:
                continue
            obs.append(Observation(subject, canonical, p, int(v)))
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
        # Open VLM runners (Izhan-format list of records)
        os.path.join(repo_root, "outputs/daphne_chexagent_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_chexagent3b_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_llava_med_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_llava15_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_llama32_vision_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_medgemma_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_internvl3_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_phi35_vision_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_qwen25vl_32b_train1_10000.json"),
        os.path.join(repo_root, "outputs/daphne_biomedclip_train1_10000.json"),
        # Shannon's Azure-deployed open VLMs (dict-keyed, batch-2 path prefix)
        os.path.join(repo_root, "outputs/pixtral_outputs.json"),
        os.path.join(repo_root, "outputs/llama_vision_outputs.json"),
        # Izhan Qwen combined frontal+lateral outputs (preferred over
        # frontal-only ``qwen{3b,7b}_results_10000.json``, which use the
        # same subject IDs and would double-count).
        os.path.join(repo_root, "outputs/qwen3b_combined.json"),
        os.path.join(repo_root, "outputs/qwen7b_combined.json"),
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
