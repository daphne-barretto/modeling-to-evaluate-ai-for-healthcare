"""Load per-model inference outputs into a unified long-form IRT dataset.

Each observation is one (subject, item) cell with a binary `response`
(1 = model got the (image, pathology) cell correct, 0 = wrong). Missing
cells (refusal, content filter, exception, unanswered pathology) are
*omitted*, not coded as 0 — IRT treats absent rows as MAR.

Layout
------

All inference outputs live under ``data/inference/`` with one JSON file
per model, named with the canonical (lowercased, dash-separated)
manuscript subject ID — e.g. ``gpt-5.4.json``, ``pixtral-12b.json``,
``qwen2.5-vl-3b.json``. Each file contains every row that model
produced; row counts vary between models (most are ~10 000 train1
images, some include additional val / lateral / top-up images).

Supported file shapes
---------------------

1. **Daphne GPT-style** outputs (``gpt-5.4.json``, ``gpt-4o.json``, plus
   ``pixtral-12b.json`` / ``llama-3.2-vision-11b.json``). Top-level dict
   keyed by item path; each value has ``deployment``,
   ``per_label_correct``, ``predictions`` (or ``correct=None`` for
   refusals).

2. **Izhan Qwen-style** outputs (everything else under
   ``data/inference/``, e.g. ``qwen2.5-vl-3b.json``,
   ``chexagent-8b.json``). Top-level list; each entry has
   ``image_path``, ``subject``, plus ``{pathology}__correct`` for each
   of the 14 pathologies (None for unanswered).

All loaders normalise ``image_path`` to the canonical CheXpert form
``CheXpert-v1.0/train/patientXXXXX/studyN/viewN_orientation.jpg`` so that
items align across subjects regardless of the prefix used upstream.

Pixtral-12B and Llama-3.2-Vision-11B are *open-weight* HuggingFace models
that we run on Modal-hosted GPUs (see ``inference/inference_pixtral.py``
and ``inference/inference_llama_vision.py``); their in-file
``deployment`` strings (``pixtral-12b-2409`` / ``llama-3.2-vision-11b``)
are normalised to manuscript IDs via ``_SUBJECT_ALIASES`` below.
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
    "gpt-4o": "GPT-4o",
}


def _normalize_subject(name: str) -> str:
    return _SUBJECT_ALIASES.get(name, name)


def _normalize_image_path(p: str) -> str:
    """Coerce any CheXpert path variant to ``CheXpert-v1.0/train/patientXXXXX/...``.

    Handles three observed forms:
      - already canonical: ``CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg``
      - HF-on-Modal batch-prefixed: ``CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)/patient00001/...``
      - bare patient-relative: ``patient00001/study1/view1_frontal.jpg``
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
    """Return paths to all inference output files we know how to load.

    Picks up every ``data/inference/<subject>.json`` file (one per model).
    Row counts vary between models — this loader does not enforce a
    shared image set; ``load_source`` will yield whatever cells each
    file contains.
    """
    inference_dir = os.path.join(repo_root, "data", "inference")
    if not os.path.isdir(inference_dir):
        return []
    paths: list[str] = []
    for name in sorted(os.listdir(inference_dir)):
        if not name.endswith(".json"):
            continue
        full = os.path.join(inference_dir, name)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def load_all(repo_root: str) -> list[Observation]:
    obs: list[Observation] = []
    for p in discover_sources(repo_root):
        n_before = len(obs)
        obs.extend(load_source(p))
        print(f"  loaded {len(obs) - n_before:>7d} obs from {os.path.relpath(p, repo_root)}")
    return obs
