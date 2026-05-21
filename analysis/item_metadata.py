"""Per-item and per-cell metadata for the response matrix.

Joins each (image, pathology) observation with CheXpert CSV columns
(Sex, Age, Frontal/Lateral, AP/PA) plus per-pathology priors (prevalence
tier, anatomical group). Designed to be cheap to call repeatedly.

Conventions
-----------

* **Anatomical groups** (used for factor-model interpretation):
    * cardiac:    Enlarged Cardiomediastinum, Cardiomegaly
    * pulmonary:  Lung Opacity, Lung Lesion, Edema, Consolidation,
                  Pneumonia, Atelectasis
    * pleural:    Pneumothorax, Pleural Effusion, Pleural Other
    * other:      Fracture, Support Devices, No Finding

* **Prevalence tiers** (from Appendix A of the pre-analysis plan,
  cross-checked against actual train1 prevalence):
    * High  ("Easy"): Support Devices, Pleural Effusion, Lung Opacity,
                      Atelectasis, Cardiomegaly, Edema
    * Mid  ("Medium"): Enlarged Cardiomediastinum, Consolidation, Pneumonia
    * Low   ("Hard"): Pneumothorax, Fracture, Lung Lesion, Pleural Other
    * NA           : No Finding (absence of pathology)

* **Age bins**: <40, 40--59, 60--79, 80+, unknown.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache

PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices",
]

ANATOMICAL_GROUP = {
    "Enlarged Cardiomediastinum": "cardiac",
    "Cardiomegaly":                "cardiac",
    "Lung Opacity":   "pulmonary",
    "Lung Lesion":    "pulmonary",
    "Edema":          "pulmonary",
    "Consolidation":  "pulmonary",
    "Pneumonia":      "pulmonary",
    "Atelectasis":    "pulmonary",
    "Pneumothorax":     "pleural",
    "Pleural Effusion": "pleural",
    "Pleural Other":    "pleural",
    "Fracture":       "other",
    "Support Devices": "other",
    "No Finding":     "other",
}

PREVALENCE_TIER = {
    "Support Devices":   "high",
    "Pleural Effusion":  "high",
    "Lung Opacity":      "high",
    "Atelectasis":       "high",
    "Cardiomegaly":      "high",
    "Edema":             "high",
    "Enlarged Cardiomediastinum": "mid",
    "Consolidation":     "mid",
    "Pneumonia":         "mid",
    "Pneumothorax":      "low",
    "Fracture":          "low",
    "Lung Lesion":       "low",
    "Pleural Other":     "low",
    "No Finding":        "na",
}


def age_bin(age) -> str:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return "unknown"
    if a < 40:  return "<40"
    if a < 60:  return "40-59"
    if a < 80:  return "60-79"
    return "80+"


@dataclass(frozen=True)
class ImageMeta:
    sex: str
    age: int | None
    age_bin: str
    view: str       # Frontal / Lateral / unknown
    ap_pa: str      # AP / PA / unknown


@lru_cache(maxsize=1)
def _load_csv_index(csv_path: str) -> dict[str, ImageMeta]:
    """Map CheXpert ``Path`` -> ImageMeta. Cached so repeated callers are free."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Expected CheXpert CSV at {csv_path}. "
            "Pull with `modal volume get chexpert-vol-v2 "
            "/CheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv "
            "data/train_visualCheXbert.csv` (or equivalent)."
        )
    out: dict[str, ImageMeta] = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            path = row.get("Path", "").strip()
            if not path:
                continue
            age_raw = row.get("Age", "").strip()
            try:
                age = int(float(age_raw))
            except (TypeError, ValueError):
                age = None
            out[path] = ImageMeta(
                sex=row.get("Sex", "").strip() or "unknown",
                age=age,
                age_bin=age_bin(age_raw),
                view=row.get("Frontal/Lateral", "").strip() or "unknown",
                ap_pa=row.get("AP/PA", "").strip() or "unknown",
            )
    return out


def default_csv_path(repo_root: str) -> str:
    return os.path.join(repo_root, "data", "train_visualCheXbert.csv")


def get_image_meta(image_path: str, repo_root: str) -> ImageMeta | None:
    """Resolve image-level metadata; returns None if image is not in CSV
    (e.g. an HF validation image that lives at ``valid/...`` rather than
    ``CheXpert-v1.0/train/...``)."""
    idx = _load_csv_index(default_csv_path(repo_root))
    return idx.get(image_path)


def get_pathology_meta(pathology: str) -> dict:
    return {
        "anatomical_group": ANATOMICAL_GROUP.get(pathology, "other"),
        "prevalence_tier":  PREVALENCE_TIER.get(pathology, "na"),
    }
