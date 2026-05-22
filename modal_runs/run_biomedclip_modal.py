"""Modal: BiomedCLIP zero-shot classifier baseline on the first 10K frontal train1 rows.

`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` is a
CLIP-style biomedical foundation model (image encoder + text encoder).
For each (image, pathology) pair we compute zero-shot binary
classification by comparing the image embedding to a "yes" prompt and a
"no" prompt and picking whichever has higher cosine similarity.

This is the *floor anchor* in our IRT analysis: a non-generative,
discriminative-only baseline that should sit clearly below the
generative VLMs but well above a random/prevalence baseline.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach \\
        modal_runs/run_biomedclip_modal.py::run_biomedclip
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("biomedclip-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
SUBJECT_NAME = "BiomedCLIP"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "open_clip_torch==2.24.0",
        "transformers==4.45.2",
        "Pillow",
        "tqdm",
        "numpy<2",
        "ftfy",
        "regex",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-biomedclip-chexpert", image=image)


PROMPT_TEMPLATES = {
    "Enlarged Cardiomediastinum": (
        "a chest x-ray showing enlarged cardiomediastinum",
        "a chest x-ray with normal cardiomediastinal contour",
    ),
    "Cardiomegaly": (
        "a chest x-ray showing cardiomegaly with an enlarged cardiac silhouette",
        "a chest x-ray with a normal heart size",
    ),
    "Lung Opacity": (
        "a chest x-ray showing lung opacity",
        "a chest x-ray with clear lung fields",
    ),
    "Lung Lesion": (
        "a chest x-ray showing a lung lesion, nodule, or mass",
        "a chest x-ray with no lung lesions, nodules, or masses",
    ),
    "Edema": (
        "a chest x-ray showing pulmonary edema",
        "a chest x-ray with no pulmonary edema",
    ),
    "Consolidation": (
        "a chest x-ray showing consolidation",
        "a chest x-ray with no consolidation",
    ),
    "Pneumonia": (
        "a chest x-ray showing pneumonia",
        "a chest x-ray with no signs of pneumonia",
    ),
    "Atelectasis": (
        "a chest x-ray showing atelectasis",
        "a chest x-ray with no atelectasis",
    ),
    "Pneumothorax": (
        "a chest x-ray showing pneumothorax",
        "a chest x-ray with no pneumothorax",
    ),
    "Pleural Effusion": (
        "a chest x-ray showing pleural effusion",
        "a chest x-ray with no pleural effusion",
    ),
    "Pleural Other": (
        "a chest x-ray showing pleural thickening or pleural abnormality",
        "a chest x-ray with no pleural abnormality",
    ),
    "Fracture": (
        "a chest x-ray showing a rib or clavicle fracture",
        "a chest x-ray with no fractures",
    ),
    "Support Devices": (
        "a chest x-ray showing support devices such as tubes, lines, or pacemakers",
        "a chest x-ray with no support devices",
    ),
    "No Finding": (
        "a normal chest x-ray with no acute cardiopulmonary findings",
        "a chest x-ray showing one or more abnormalities",
    ),
}


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 6,
    memory=16384,
    cpu=4,
)
def run_biomedclip(n_images: int = 10_000, save_every: int = 200):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    import open_clip

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        PATHOLOGIES,
        atomic_write_json,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, f"daphne_biomedclip_train1_{n_images}.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[BiomedCLIP] resume: {len(done)} already done.", flush=True)

    print(
        f"[BiomedCLIP] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[BiomedCLIP] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[BiomedCLIP] loading {MODEL_ID} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, cache_dir=MODEL_CACHE
    )
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to("cuda").eval()
    print("[BiomedCLIP] model loaded.", flush=True)

    text_pos: list[str] = []
    text_neg: list[str] = []
    path_order: list[str] = []
    for p in PATHOLOGIES:
        pos, neg = PROMPT_TEMPLATES[p]
        text_pos.append(pos)
        text_neg.append(neg)
        path_order.append(p)
    text_all = text_pos + text_neg
    with torch.inference_mode():
        text_tokens = tokenizer(text_all).to("cuda")
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    pos_features = text_features[: len(path_order)]
    neg_features = text_features[len(path_order):]
    print(f"[BiomedCLIP] encoded {len(text_all)} prompts.", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="BiomedCLIP")):
        rel_path = row["Path"]
        if rel_path in done:
            continue
        img_path = row["_image_path"]
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        try:
            with torch.inference_mode():
                img_input = preprocess(pil_image).unsqueeze(0).to("cuda")
                img_feat = model.encode_image(img_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                pos_sim = (img_feat @ pos_features.t()).squeeze(0)
                neg_sim = (img_feat @ neg_features.t()).squeeze(0)
                pred_pos = (pos_sim > neg_sim).cpu().tolist()
                pos_sim_v = pos_sim.cpu().tolist()
                neg_sim_v = neg_sim.cpu().tolist()
        except Exception as e:
            print(f"  [error] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        record: dict = {
            "image_path": rel_path,
            "subject": SUBJECT_NAME,
            "item_id": str(i),
            "raw_response": "",
        }
        n_valid = 0
        n_correct = 0
        for k, p in enumerate(path_order):
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
            pred = bool(pred_pos[k])
            gt_pos = gt_val == 1.0
            correct = int(pred == gt_pos)
            record[f"{p}__gt"] = gt_val
            record[f"{p}__answer"] = "yes" if pred else "no"
            record[f"{p}__correct"] = correct
            record[f"{p}__sim_yes"] = pos_sim_v[k]
            record[f"{p}__sim_no"] = neg_sim_v[k]
            n_valid += 1
            n_correct += correct
        record["n_valid"] = n_valid
        record["n_correct"] = n_correct
        record["all_correct"] = int(n_correct == n_valid) if n_valid > 0 else None
        records.append(record)
        done.add(rel_path)
        n_processed += 1

        if n_processed % save_every == 0:
            atomic_write_json(records, output_path)
            volume.commit()
            print(
                f"  [save] {len(records)} records ({n_errors} errors)",
                flush=True,
            )

    atomic_write_json(records, output_path)
    volume.commit()
    print(f"\n[BiomedCLIP] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}
