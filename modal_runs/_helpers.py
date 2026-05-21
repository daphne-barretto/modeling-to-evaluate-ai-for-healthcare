"""Shared inference helpers for Modal-hosted Azure OpenAI runs.

This module is a self-contained copy of the Azure-call logic from
`inference/openai_inference.py` so the Modal containers don't need to
import the HuggingFace-streaming code path. PATHOLOGIES is duplicated
here intentionally (it's stable / spec-defined).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

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

PROMPT = (
    "You are an expert radiologist reviewing a single chest radiograph.\n"
    "For each of the 14 thoracic conditions listed below, decide whether the\n"
    "condition is PRESENT (1) or ABSENT (0) in this image.\n\n"
    "Conditions (use these exact keys, in this order):\n"
    + "\n".join(f"  - {p}" for p in PATHOLOGIES)
    + "\n\n"
    "Respond with ONLY a single valid JSON object. The keys must be the\n"
    "exact condition names above and the values must be the integers 0 or 1.\n"
    "Do not include any prose, markdown fences, units, or explanations.\n"
    'Example: {"Atelectasis": 0, "Cardiomegaly": 1, ...}'
)


def make_azure_client(api_version: str):
    import os
    from openai import AzureOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version
    )


def image_path_to_data_url(image_path: str) -> str:
    from PIL import Image

    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_response(raw: str) -> dict[str, int]:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[str, int] = {}
    for p in PATHOLOGIES:
        if p not in obj:
            continue
        v = obj[p]
        if isinstance(v, bool):
            out[p] = int(v)
        elif isinstance(v, (int, float)):
            out[p] = 1 if v >= 0.5 else 0
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "present", "positive"}:
                out[p] = 1
            elif s in {"0", "false", "no", "absent", "negative"}:
                out[p] = 0
    return out


def binarize_chexpert_label(raw_label, uncertain_policy: str = "zero") -> int | None:
    """CheXpert CSV label -> 0/1, or None if blank/unmentioned.

    CSV cells: '1.0'=positive, '0.0'=negative, '-1.0'=uncertain, ''=unmentioned.
    U-Zeros policy: uncertain -> 0.
    """
    if raw_label is None:
        return None
    s = str(raw_label).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v == 1.0:
        return 1
    if v == -1.0:
        return {"zero": 0, "one": 1}.get(uncertain_policy, 0)
    return 0


def score(predictions: dict[str, int], labels: dict[str, Any]) -> dict[str, Any]:
    per_label: dict[str, int | None] = {}
    bin_labels: dict[str, int | None] = {}
    matchable = True
    all_match = True
    for p in PATHOLOGIES:
        gt = binarize_chexpert_label(labels.get(p))
        bin_labels[p] = gt
        if gt is None:
            per_label[p] = None
            continue
        pred = predictions.get(p)
        ok = pred is not None and int(pred) == gt
        per_label[p] = int(ok)
        if not ok:
            all_match = False
        matchable = True
    return {
        "correct": int(all_match) if matchable else None,
        "per_label_correct": per_label,
        "binarized_labels": bin_labels,
    }


def _flagged_only(cfr) -> dict | None:
    if not cfr or not isinstance(cfr, dict):
        return None
    flagged = {
        k: v
        for k, v in cfr.items()
        if isinstance(v, dict) and (v.get("filtered") or v.get("detected"))
    }
    return flagged or None


def call_chat_completions(
    client, deployment: str, image_path: str, extra_kwargs: dict | None = None
) -> dict:
    data_url = image_path_to_data_url(image_path)
    kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    full = resp.model_dump()
    diagnostics = {
        "finish_reason": choice.finish_reason,
        "refusal": getattr(choice.message, "refusal", None),
        "completion_content_filter": _flagged_only(
            full["choices"][0].get("content_filter_results")
        ),
        "prompt_content_filter": [
            _flagged_only(p.get("content_filter_results"))
            for p in (full.get("prompt_filter_results") or [])
        ],
        "usage": full.get("usage"),
        "model": full.get("model"),
    }
    return {"text": text, "diagnostics": diagnostics}


def call_responses(
    client, deployment: str, image_path: str, extra_kwargs: dict | None = None
) -> dict:
    data_url = image_path_to_data_url(image_path)
    kwargs: dict[str, Any] = {
        "model": deployment,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None) or ""
    full = resp.model_dump()

    refusal: str | None = None
    reasoning_summary_parts: list[str] = []
    if not text:
        parts: list[str] = []
        for item in full.get("output") or []:
            if item.get("type") == "reasoning":
                for s in item.get("summary") or []:
                    if s.get("text"):
                        reasoning_summary_parts.append(s["text"])
            for c in item.get("content") or []:
                ct = c.get("type")
                if ct in ("output_text", "text") and c.get("text"):
                    parts.append(c["text"])
                elif ct == "refusal" and c.get("refusal"):
                    refusal = c["refusal"]
        if parts:
            text = "".join(parts)
    else:
        for item in full.get("output") or []:
            if item.get("type") == "reasoning":
                for s in item.get("summary") or []:
                    if s.get("text"):
                        reasoning_summary_parts.append(s["text"])

    usage = full.get("usage") or {}
    reasoning_tokens = (usage.get("output_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    diagnostics = {
        "status": full.get("status"),
        "incomplete_details": full.get("incomplete_details"),
        "refusal": refusal,
        "reasoning_summary": "\n".join(reasoning_summary_parts) or None,
        "reasoning_tokens": reasoning_tokens,
        "usage": usage or None,
        "model": full.get("model"),
    }
    return {"text": text, "diagnostics": diagnostics}


CALLERS = {"chat": call_chat_completions, "responses": call_responses}


def classify_missing(text: str, preds: dict, diagnostics: dict) -> str | None:
    """Return missingness reason or None if the response is usable."""
    if diagnostics.get("refusal"):
        return "model_refusal"
    if diagnostics.get("incomplete_details"):
        return "incomplete"
    if diagnostics.get("finish_reason") == "content_filter":
        return "azure_content_filter"
    if not text:
        return "empty_response"
    if not preds:
        return "unparseable"
    return None


# ─── Volume layout (used by Modal scripts) ─────────────────────────────────────
VOLUME_PATH = "/data"
EXTRACTED_DIR = f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408/extracted"
TRAIN_CSV = (
    f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv"
)
OUTPUTS_DIR = f"{VOLUME_PATH}/outputs"


def find_image_on_volume(rel_path: str) -> str | None:
    """Resolve a CSV Path entry (e.g. 'CheXpert-v1.0/train/patient.../...jpg')
    to an absolute path under EXTRACTED_DIR. The extracted train batches each
    place their patients under their own subdirectory, so we scan each batch."""
    import os

    parts = rel_path.split("/")
    if len(parts) < 4:
        return None
    patient = parts[2]
    rest = parts[3:]
    if not os.path.isdir(EXTRACTED_DIR):
        return None
    for batch_dir in os.listdir(EXTRACTED_DIR):
        candidate = os.path.join(EXTRACTED_DIR, batch_dir, patient, *rest)
        if os.path.exists(candidate):
            return candidate
    return None


def load_csv_rows() -> list[dict]:
    """Load all rows from train_visualCheXbert.csv as list of dicts."""
    import csv

    with open(TRAIN_CSV) as f:
        return list(csv.DictReader(f))


def select_frontal_train_rows(n: int) -> list[dict]:
    """Take rows where Frontal/Lateral == 'Frontal' AND the image exists on the
    volume. Returns the first `n` such rows in CSV order, with `_image_path`
    set to the resolved on-disk path."""
    out: list[dict] = []
    for row in load_csv_rows():
        if row.get("Frontal/Lateral", "").strip() != "Frontal":
            continue
        img = find_image_on_volume(row["Path"])
        if img is None:
            continue
        row["_image_path"] = img
        out.append(row)
        if len(out) >= n:
            break
    return out


def select_train1_rows(n: int) -> list[dict]:
    """Take any train1 rows (Frontal + Lateral) whose image exists on the
    volume. Returns the first `n` such rows in CSV order, with `_image_path`
    set to the resolved on-disk path. No view filter."""
    out: list[dict] = []
    for row in load_csv_rows():
        img = find_image_on_volume(row["Path"])
        if img is None:
            continue
        row["_image_path"] = img
        out.append(row)
        if len(out) >= n:
            break
    return out


def run_modal_inference(
    *,
    deployment: str,
    api_type: str,
    api_version: str,
    output_filename: str,
    n_images: int = 10_000,
    extra_kwargs: dict | None = None,
    max_workers: int = 8,
    commit_every: int = 50,
    frontal_only: bool = True,
    seed_from: str | None = None,
) -> int:
    """Body of a Modal function: read CSV, run Azure calls in a thread pool,
    write resumable JSON to volume. Returns count of items processed in this
    invocation (excludes already-done items).

    The caller is responsible for the @app.function decorator and for calling
    volume.commit() at the end (we commit periodically during the run too).

    Args:
        frontal_only: if True (default), use `select_frontal_train_rows`;
            otherwise use `select_train1_rows` (any view).
        seed_from: optional filename inside OUTPUTS_DIR whose contents are
            copied to `output_filename` if the target doesn't already exist.
            Used to pre-seed a top-up run with prior results so resume logic
            skips already-done items keyed by Path."""
    import json
    import os
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if api_type not in CALLERS:
        raise ValueError(f"api_type must be one of {list(CALLERS)}; got {api_type!r}")
    caller = CALLERS[api_type]

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    if seed_from and not os.path.exists(output_path):
        seed_path = os.path.join(OUTPUTS_DIR, seed_from)
        if os.path.exists(seed_path):
            import shutil

            shutil.copyfile(seed_path, output_path)
            print(
                f"[{deployment}] seeded {output_filename} from {seed_from}",
                flush=True,
            )
        else:
            print(
                f"[{deployment}] seed_from={seed_from} not found at {seed_path}; "
                f"starting fresh.",
                flush=True,
            )

    view_desc = "frontal" if frontal_only else "any-view"
    print(
        f"[{deployment}] selecting first {n_images} {view_desc} train rows...",
        flush=True,
    )
    selector = select_frontal_train_rows if frontal_only else select_train1_rows
    rows = selector(n_images)
    print(f"[{deployment}] selected {len(rows)} rows.", flush=True)

    results: dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                results = json.load(f)
        except Exception:
            results = {}
    done = set(results.keys())
    todo = [r for r in rows if r["Path"] not in done]
    print(
        f"[{deployment}] resume state: {len(done)} done / {len(rows)} target; "
        f"{len(todo)} to do.",
        flush=True,
    )

    client = make_azure_client(api_version=api_version)
    lock = threading.Lock()
    counter = {"done": 0, "errors": 0}
    start = time.time()

    def labels_dict(row: dict) -> dict[str, str]:
        return {p: row.get(p, "") for p in PATHOLOGIES}

    def process_one(row: dict) -> tuple[str, dict]:
        item_id = row["Path"]
        labels = labels_dict(row)
        try:
            out = caller(client, deployment, row["_image_path"], extra_kwargs)
            text = out["text"]
            diagnostics = out["diagnostics"]
            preds = parse_response(text)
            missing_reason = classify_missing(text, preds, diagnostics)
            if missing_reason:
                record = {
                    "deployment": deployment,
                    "api_type": api_type,
                    "predictions": None,
                    "labels": labels,
                    "raw": text,
                    "diagnostics": diagnostics,
                    "missing_reason": missing_reason,
                    "correct": None,
                    "per_label_correct": None,
                    "binarized_labels": None,
                }
            else:
                scored = score(preds, labels)
                record = {
                    "deployment": deployment,
                    "api_type": api_type,
                    "predictions": preds,
                    "labels": labels,
                    "raw": text,
                    "diagnostics": diagnostics,
                    "missing_reason": None,
                    **scored,
                }
        except Exception as e:  # noqa: BLE001
            record = {
                "deployment": deployment,
                "api_type": api_type,
                "error": f"{type(e).__name__}: {e}",
                "missing_reason": "exception",
                "correct": None,
            }
        return item_id, record

    def flush() -> None:
        tmp = output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, output_path)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, row): row["Path"] for row in todo}
        for fut in as_completed(futures):
            item_id, rec = fut.result()
            with lock:
                results[item_id] = rec
                counter["done"] += 1
                if rec.get("missing_reason") == "exception":
                    counter["errors"] += 1
                if counter["done"] % commit_every == 0:
                    flush()
                    rate = counter["done"] / max(time.time() - start, 1e-6)
                    print(
                        f"[{deployment}] {len(results)}/{len(rows)} "
                        f"({rate:.1f} items/s, errors={counter['errors']})",
                        flush=True,
                    )

    flush()
    elapsed = time.time() - start
    print(
        f"[{deployment}] complete: {len(results)} / {len(rows)} in "
        f"{elapsed/60:.1f} min, errors={counter['errors']}",
        flush=True,
    )
    return counter["done"]
