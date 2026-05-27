"""
Modal inference script: Pixtral 12B on CheXpert chest X-rays.

Usage:
    modal run inference_pixtral.py

This script:
1. Loads Pixtral 12B (2409) on an A100 GPU
2. Reads the first 10,000 CheXpert images from the Modal volume
3. Runs inference with a structured prompt for 14 pathology predictions
4. Saves results as JSON to the Modal volume

.venv\Scripts\activate.bat

Prerequisites:
 # Same local setup as the LLaMA script:
 modal token new
 modal secret create huggingface-token HF_TOKEN=hf_YOUR_TOKEN

 # No license gating — Pixtral is Apache 2.0, no approval needed.

 # Run inference
 modal run inference_pixtral.py

 # Download results
 modal run download_results.py
"""

import modal
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistral-community/pixtral-12b"
VOLUME_NAME = "chexpert-vol-v2"
DATASET_DIR = "CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)"
LABELS_CSV_PATH = "CheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv"
OUTPUT_PATH = "inference_outputs/pixtral-12b.json"
CHECKPOINT_DIR = "inference_outputs/pixtral_checkpoints"
MAX_IMAGES = 10_000
CHECKPOINT_EVERY = 500

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

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("chexpert-pixtral-inference")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers>=4.48.0",
        "accelerate>=0.33.0",
        "Pillow>=10.0",
        "pandas",
    )
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def discover_images(dataset_root: str, max_images: int) -> list[str]:
    """Walk the CheXpert directory tree and collect image paths (sorted)."""
    image_paths = []
    extensions = {".jpg", ".jpeg", ".png"}

    for root, dirs, files in os.walk(dataset_root):
        dirs.sort()
        for fname in sorted(files):
            if Path(fname).suffix.lower() in extensions:
                image_paths.append(os.path.join(root, fname))
                if len(image_paths) >= max_images:
                    return image_paths
    return image_paths


def parse_predictions(raw_text: str) -> dict:
    """Try to extract a JSON dict of predictions from model output."""
    try:
        obj = json.loads(raw_text.strip())
        if isinstance(obj, dict):
            return {k: int(v) for k, v in obj.items() if k in PATHOLOGIES}
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if isinstance(obj, dict):
                return {k: int(v) for k, v in obj.items() if k in PATHOLOGIES}
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def load_labels(volume_root: str) -> dict:
    """Load ground-truth labels from the CSV on the volume."""
    import pandas as pd

    csv_path = os.path.join(volume_root, LABELS_CSV_PATH)
    if not os.path.exists(csv_path):
        print(f"WARNING: Labels CSV not found at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    labels_dict = {}
    for _, row in df.iterrows():
        path_val = row.get("Path", "")
        if not path_val:
            continue
        parts = path_val.replace("\\", "/").split("/")
        patient_idx = next((i for i, p in enumerate(parts) if p.startswith("patient")), None)
        if patient_idx is not None:
            norm_key = "/".join(parts[patient_idx:])
        else:
            norm_key = path_val

        entry = {}
        for p in PATHOLOGIES:
            if p in row:
                entry[p] = row[p]
            else:
                entry[p] = float("nan")
        labels_dict[norm_key] = entry
    return labels_dict


def match_label_key(image_path: str, labels_dict: dict) -> str | None:
    """Find matching label key using normalized patient/study/view path."""
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part.startswith("patient"):
            norm_key = "/".join(parts[i:])
            if norm_key in labels_dict:
                return norm_key
    return None


def binarize_label(val) -> int:
    """Convert CheXpert label to binary: 1=positive, 0=negative."""
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    if float(val) == 1.0 or float(val) == -1.0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Modal function: main inference
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=86400,
)
def run_inference():
    """Main inference function that runs on Modal GPU."""
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from PIL import Image as PILImage

    volume_root = "/data"
    dataset_root = os.path.join(volume_root, DATASET_DIR)
    output_file = os.path.join(volume_root, OUTPUT_PATH)
    checkpoint_dir = os.path.join(volume_root, CHECKPOINT_DIR)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load ground-truth labels
    print("Loading ground-truth labels...")
    labels_dict = load_labels(volume_root)
    print(f"Loaded labels for {len(labels_dict)} images")

    # Discover images
    print(f"Discovering images in {dataset_root}...")
    image_paths = discover_images(dataset_root, MAX_IMAGES)
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("ERROR: No images found! Check the dataset path.")
        return

    # Check for existing checkpoint to resume from
    results = {}
    start_idx = 0
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith(".json")],
            reverse=True,
        )
        if checkpoint_files:
            latest = os.path.join(checkpoint_dir, checkpoint_files[0])
            print(f"Resuming from checkpoint: {latest}")
            with open(latest, "r") as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from image index {start_idx}")

    # Load model
    print(f"Loading model {MODEL_ID}...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    print("Model loaded successfully!")

    # Inference loop
    for idx in range(start_idx, len(image_paths)):
        img_path = image_paths[idx]

        try:
            pil_image = PILImage.open(img_path).convert("RGB")

            # Pixtral uses a chat template with [IMG] token
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]

            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(
                text=input_text, images=[pil_image], return_tensors="pt"
            ).to(model.device)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode only new tokens
            generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
            raw_output = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Parse predictions
            predictions = parse_predictions(raw_output)

            # Build relative path key
            rel_path = os.path.relpath(img_path, volume_root)

            # Look up ground-truth labels
            label_key = match_label_key(img_path, labels_dict)
            raw_labels = labels_dict.get(label_key, {}) if label_key else {}

            binarized_labels = {p: binarize_label(raw_labels.get(p)) for p in PATHOLOGIES}
            labels_float = {p: float(binarized_labels[p]) for p in PATHOLOGIES}

            # Compute correctness
            per_label_correct = {}
            for p in PATHOLOGIES:
                if p in predictions:
                    per_label_correct[p] = int(predictions[p] == binarized_labels[p])
                else:
                    per_label_correct[p] = 0

            correct = int(all(per_label_correct.get(p, 0) == 1 for p in PATHOLOGIES))

            results[rel_path] = {
                "deployment": "pixtral-12b-2409",
                "api_type": "local",
                "predictions": predictions,
                "labels": labels_float,
                "raw": raw_output,
                "correct": correct,
                "per_label_correct": per_label_correct,
                "binarized_labels": binarized_labels,
            }

        except Exception as e:
            rel_path = os.path.relpath(img_path, volume_root)
            results[rel_path] = {
                "deployment": "pixtral-12b-2409",
                "api_type": "local",
                "predictions": {},
                "labels": {},
                "raw": f"ERROR: {str(e)}",
                "correct": 0,
                "per_label_correct": {},
                "binarized_labels": {},
            }
            print(f"[{idx}] ERROR processing {img_path}: {e}")

        # Progress logging
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{idx + 1:06d}.json")
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2)
            volume.commit()
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nDone! Results saved to {output_file}")
    print(f"Total images processed: {len(results)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run with: modal run --detach inference_pixtral.py"""
    print("Starting CheXpert inference with Pixtral 12B...")
    print(f"Model: {MODEL_ID}")
    print(f"Max images: {MAX_IMAGES}")
    fc = run_inference.spawn()
    print(f"Job spawned! Function call ID: {fc.object_id}")
    print("Job is running detached on Modal. Safe to close terminal.")
