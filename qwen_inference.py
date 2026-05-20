"""
Modal script to run Qwen2.5-VL-7B-Instruct on CheXpert images.
One prompt per image asking about all 14 pathologies at once.
Parses yes/no for each pathology from the single response.

Run with: modal run qwen_inference.py
"""

import modal

# ── Volumes ────────────────────────────────────────────────────────────────────
volume      = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol   = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
EXTRACTED   = f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408/extracted"
CSV_PATH    = f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers>=4.45.0",
        "accelerate", "qwen-vl-utils", "Pillow", "pandas", "tqdm",
    )
)

app = modal.App("chexpert-qwen-inference", image=image)


def find_image(rel_path: str) -> str | None:
    import os
    parts   = rel_path.split("/")
    patient = parts[2]
    rest    = parts[3:]
    for batch_dir in os.listdir(EXTRACTED):
        candidate = os.path.join(EXTRACTED, batch_dir, patient, *rest)
        if os.path.exists(candidate):
            return candidate
    return None


def parse_response(response: str) -> dict[str, bool | None]:
    """Parse one yes/no per pathology from the model response."""
    result = {}
    lines  = response.strip().lower().splitlines()
    for p in PATHOLOGIES:
        key = p.lower()
        found = None
        for line in lines:
            if key in line:
                if "yes" in line:
                    found = True
                elif "no" in line:
                    found = False
                break
        result[p] = found
    return result


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A100",
    timeout=60 * 60 * 6,
    memory=32768,
    cpu=4,
)
def run_inference(n_images: int = 100):
    import os, csv
    import torch
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # ── Load CSV ───────────────────────────────────────────────────────────────
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Frontal/Lateral", "").strip() == "Frontal":
                rows.append(row)
    rows = rows[:n_images]
    print(f"✓ Loaded {len(rows)} frontal images")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", cache_dir=MODEL_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
    model.eval()
    print("✓ Model loaded")

    records = []

    for i, row in enumerate(tqdm(rows, desc="Images")):
        rel_path = row["Path"]
        img_path = find_image(rel_path)

        if img_path is None:
            print(f"  [skip] not found: {rel_path}")
            continue

        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {e}")
            continue

        # ── Single prompt for all pathologies ─────────────────────────────────
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": PROMPT},
            ]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)

        response = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        parsed = parse_response(response)

        # ── Score ──────────────────────────────────────────────────────────────
        record = {
            "image_path": rel_path,
            "subject":    "Qwen2.5-VL-7B",
            "item_id":    str(i),
            "raw_response": response.strip(),
        }

        n_valid = 0
        n_correct = 0

        for p in PATHOLOGIES:
            gt_raw = row.get(p, "")
            try:
                gt_val = float(gt_raw) if gt_raw.strip() not in ("", "nan") else None
            except ValueError:
                gt_val = None

            # Skip uncertain or missing
            if gt_val is None or gt_val == -1.0:
                record[f"{p}__gt"]      = None
                record[f"{p}__answer"]  = None
                record[f"{p}__correct"] = None
                continue

            qwen_pos = parsed.get(p)
            gt_pos   = gt_val == 1.0

            if qwen_pos is None:
                # Model didn't answer for this pathology
                record[f"{p}__gt"]      = gt_val
                record[f"{p}__answer"]  = "unclear"
                record[f"{p}__correct"] = None
                continue

            correct = int(qwen_pos == gt_pos)
            record[f"{p}__gt"]      = gt_val
            record[f"{p}__answer"]  = "yes" if qwen_pos else "no"
            record[f"{p}__correct"] = correct
            n_valid   += 1
            n_correct += correct

        record["n_valid"]      = n_valid
        record["n_correct"]    = n_correct
        record["all_correct"]  = int(n_correct == n_valid) if n_valid > 0 else None
        records.append(record)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir  = f"{VOLUME_PATH}/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/qwen_results_{n_images}.csv"

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    volume.commit()

    print(f"\n✓ Saved {len(df)} rows → {out_path}")
    print(f"  all_correct rate: {df['all_correct'].mean():.1%}")
    for p in PATHOLOGIES:
        col = f"{p}__correct"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                print(f"  {p}: {vals.mean():.1%} ({len(vals)} scored)")

    return len(records)


@app.local_entrypoint()
def main():
    print("Running Qwen2.5-VL-7B on first 100 CheXpert images (1 prompt/image)...")
    n = run_inference.remote(n_images=100)
    print(f"\n✓ Done. {n} records saved.")