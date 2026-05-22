"""Modal: run InternVL3-8B on the first 10K frontal train1 rows.

InternVL3 is OpenGVLab's open-weights frontier VLM (Qwen-style chat).
Uses the structured 14-pathology yes/no prompt from `_open_vlm_helpers.PROMPT`.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach modal_runs/run_internvl3_modal.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("internvl3-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "OpenGVLab/InternVL3-8B"
SUBJECT_NAME = "InternVL3-8B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",
        "Pillow",
        "tqdm",
        "einops",
        "timm",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-internvl3-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A100",
    timeout=60 * 60 * 8,
    memory=49152,
    cpu=4,
)
def run_internvl3(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        PROMPT,
        atomic_write_json,
        build_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, f"daphne_internvl3_train1_{n_images}.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[InternVL3] resume: {len(done)} already done.", flush=True)

    print(
        f"[InternVL3] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[InternVL3] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[InternVL3] loading {MODEL_ID} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=MODEL_CACHE, use_fast=False
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    print("[InternVL3] model loaded.", flush=True)

    # InternVL3 ships its own pixel_values preprocessor via `model.chat`.
    # We pass the image through their standard load_image transform.
    from torchvision import transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size=448):
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    transform = build_transform()

    generation_config = dict(max_new_tokens=512, do_sample=False)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="InternVL3")):
        rel_path = row["Path"]
        if rel_path in done:
            continue
        img_path = row["_image_path"]
        try:
            pil_image = Image.open(img_path).convert("RGB")
            pixel_values = transform(pil_image).unsqueeze(0).to("cuda").to(torch.bfloat16)
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        try:
            question = f"<image>\n{PROMPT}"
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                )
            response = response.strip() if response else ""
        except Exception as e:
            print(f"  [error] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        record = build_record(
            subject=SUBJECT_NAME,
            item_id=i,
            rel_path=rel_path,
            raw_response=response,
            row=row,
        )
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
    print(f"\n[InternVL3] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}


@app.local_entrypoint()
def main(n_images: int = 10_000, save_every: int = 100):
    print(f"── Spawning InternVL3-8B on {n_images} frontal train1 rows ──")
    call = run_internvl3.spawn(n_images=n_images, save_every=save_every)
    print(f"✓ Spawned function call: {call.object_id}")
    print("  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/daphne_internvl3_train1_{n_images}.json")
