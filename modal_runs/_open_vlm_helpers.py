"""Shared helpers for open-weight VLM Modal runners (LLaVA-Med, CheXagent, ...).

Each model-specific runner imports these to keep per-record parsing,
scoring, and JSON output identical across subjects.  The output format
is a JSON list with one record per image, each holding
``{pathology}__gt | __answer | __correct`` columns for the 14 CheXpert
pathologies, so ``analysis/data_loader.py`` ingests it without changes.
"""

from __future__ import annotations

import json
import os
import tempfile

PATHOLOGIES = [
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
    "No Finding",
]

PROMPT = (
    "This is a chest X-ray. For each of the following findings, "
    "answer yes or no. Reply in exactly this format, one per line:\n"
    + "\n".join(f"{p}: yes/no" for p in PATHOLOGIES)
)


def parse_response(response: str) -> dict[str, bool | None]:
    """Parse one yes/no per pathology from the model response.

    Behaviour is fixed across every subject so cross-model rows are
    scored under identical rules.
    """
    result: dict[str, bool | None] = {}
    lines = response.strip().lower().splitlines()
    for p in PATHOLOGIES:
        key = p.lower()
        found: bool | None = None
        for line in lines:
            if key in line:
                if "yes" in line:
                    found = True
                elif "no" in line:
                    found = False
                break
        result[p] = found
    return result


def build_record(
    *,
    subject: str,
    item_id: int,
    rel_path: str,
    raw_response: str,
    row: dict,
) -> dict:
    """Build one analysis-pipeline-ready record from a single inference."""
    parsed = parse_response(raw_response)
    record: dict = {
        "image_path": rel_path,
        "subject": subject,
        "item_id": str(item_id),
        "raw_response": raw_response.strip(),
    }
    n_valid = 0
    n_correct = 0
    for p in PATHOLOGIES:
        gt_raw = row.get(p, "")
        try:
            gt_val = (
                float(gt_raw) if str(gt_raw).strip() not in ("", "nan") else None
            )
        except (TypeError, ValueError):
            gt_val = None

        if gt_val is None or gt_val == -1.0:
            record[f"{p}__gt"] = None
            record[f"{p}__answer"] = None
            record[f"{p}__correct"] = None
            continue

        pred = parsed.get(p)
        gt_pos = gt_val == 1.0
        if pred is None:
            record[f"{p}__gt"] = gt_val
            record[f"{p}__answer"] = "unclear"
            record[f"{p}__correct"] = None
            continue

        correct = int(pred == gt_pos)
        record[f"{p}__gt"] = gt_val
        record[f"{p}__answer"] = "yes" if pred else "no"
        record[f"{p}__correct"] = correct
        n_valid += 1
        n_correct += correct

    record["n_valid"] = n_valid
    record["n_correct"] = n_correct
    record["all_correct"] = (
        int(n_correct == n_valid) if n_valid > 0 else None
    )
    return record


def load_existing(output_path: str) -> tuple[list[dict], set[str]]:
    """Load prior results (resumable). Returns (records, set of done image paths)."""
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return [], set()
    if not isinstance(data, list):
        return [], set()
    done = {r["image_path"] for r in data if "image_path" in r}
    return data, done


def atomic_write_json(records: list[dict], output_path: str) -> None:
    """Atomic write to avoid corrupted files on preemption."""
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=out_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f)
        os.replace(tmp, output_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def summarize(records: list[dict]) -> dict:
    """Compute aggregate accuracy + per-pathology accuracy from records."""
    summary: dict = {"n": len(records)}
    if not records:
        return summary
    n_all = sum(1 for r in records if r.get("all_correct") is not None)
    n_all_correct = sum(1 for r in records if r.get("all_correct") == 1)
    summary["all_correct_rate"] = (
        n_all_correct / n_all if n_all else None
    )
    summary["per_pathology"] = {}
    for p in PATHOLOGIES:
        col = f"{p}__correct"
        vals = [r[col] for r in records if r.get(col) is not None]
        summary["per_pathology"][p] = {
            "n_scored": len(vals),
            "accuracy": (sum(vals) / len(vals)) if vals else None,
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Free-text findings prompt + synonym parser
#
# Used by models trained on free-text radiology reports (LLaVA-Med, CheXagent)
# that do not reliably follow structured "Cardiomegaly: yes/no" prompts.
# ─────────────────────────────────────────────────────────────────────────────

FINDINGS_PROMPT = (
    "Generate the findings section for this chest X-ray. "
    "Be specific about which conditions are present or absent."
)

PATHOLOGY_SYNONYMS = {
    "Enlarged Cardiomediastinum": [
        "enlarged cardiomediastinum",
        "widened mediastinum",
        "mediastinal widening",
    ],
    "Cardiomegaly": [
        "cardiomegaly",
        "heart enlargement",
        "cardiac enlargement",
        "enlarged heart",
        "enlarged cardiac silhouette",
    ],
    "Lung Opacity": [
        "lung opacity",
        "pulmonary opacity",
        "opacities",
        "opacification",
    ],
    "Lung Lesion": ["lung lesion", "pulmonary lesion", "nodule", "mass"],
    "Edema": ["edema", "oedema", "pulmonary congestion"],
    "Consolidation": ["consolidation"],
    "Pneumonia": ["pneumonia", "infection", "infectious process"],
    "Atelectasis": ["atelectasis", "collapse"],
    "Pneumothorax": ["pneumothorax"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Pleural Other": [
        "pleural thickening",
        "pleural scarring",
        "calcified pleura",
    ],
    "Fracture": ["fracture"],
    "Support Devices": [
        "support device",
        "tube",
        "line",
        "catheter",
        "pacemaker",
        "wire",
        "icd",
        "hardware",
    ],
    "No Finding": [
        "no finding",
        "no findings",
        "no acute",
        "normal",
        "unremarkable",
        "no abnormalit",
    ],
}

NEG_PATTERNS = [
    "no ",
    "without ",
    "absent",
    "negative for ",
    "no evidence of ",
    "no significant ",
]


def _is_mentioned(p: str, resp_lower: str) -> bool | None:
    """True if pathology positively mentioned, False if negated, None if absent."""
    terms = PATHOLOGY_SYNONYMS.get(p, [p.lower()])
    for term in terms:
        idx = resp_lower.find(term)
        if idx == -1:
            continue
        left = resp_lower[max(0, idx - 25):idx]
        if any(neg in left for neg in NEG_PATTERNS):
            return False
        return True
    return None


def build_findings_record(
    *,
    subject: str,
    item_id: int,
    rel_path: str,
    response: str,
    row: dict,
) -> dict:
    """Build a record from a free-text findings response using synonym + negation.

    Models like LLaVA-Med and CheXagent are trained on free-text radiology
    reports rather than structured yes/no prompts and routinely echo the
    template back unchanged when forced into a list format. This parser
    extracts a per-pathology call from the resulting narrative.
    """
    record: dict = {
        "image_path": rel_path,
        "subject": subject,
        "item_id": str(item_id),
        "raw_response": response.strip(),
    }
    resp_lower = response.lower()
    responded = bool(resp_lower.strip())
    n_valid = 0
    n_correct = 0

    for p in PATHOLOGIES:
        gt_raw = row.get(p, "")
        try:
            gt_val = (
                float(gt_raw)
                if str(gt_raw).strip() not in ("", "nan")
                else None
            )
        except (TypeError, ValueError):
            gt_val = None
        if gt_val is None or gt_val == -1.0:
            record[f"{p}__gt"] = None
            record[f"{p}__answer"] = None
            record[f"{p}__correct"] = None
            continue
        if not responded:
            record[f"{p}__gt"] = gt_val
            record[f"{p}__answer"] = "unclear"
            record[f"{p}__correct"] = None
            continue

        mention = _is_mentioned(p, resp_lower)
        if p == "No Finding":
            positive_findings = any(
                _is_mentioned(q, resp_lower) is True
                for q in PATHOLOGIES
                if q != "No Finding"
            )
            pred = (mention is True) or (
                mention is None and not positive_findings
            )
        else:
            pred = mention is True

        gt_pos = gt_val == 1.0
        correct = int(pred == gt_pos)
        record[f"{p}__gt"] = gt_val
        record[f"{p}__answer"] = "yes" if pred else "no"
        record[f"{p}__correct"] = correct
        n_valid += 1
        n_correct += correct
    record["n_valid"] = n_valid
    record["n_correct"] = n_correct
    record["all_correct"] = (
        int(n_correct == n_valid) if n_valid > 0 else None
    )
    return record
