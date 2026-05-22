"""Modal: run MedGemma-4B-it (Google) on the first 10K frontal train1 rows.

MedGemma is Google's medical-domain Gemma-3 VLM (4B params). Requires
accepting the gated model usage policy at
https://huggingface.co/google/medgemma-4b-it and providing an HF token
through the Modal secret `huggingface-token` (which must expose
`HF_TOKEN`).

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach \\
        modal_runs/run_medgemma_modal.py::run_medgemma
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "google/medgemma-4b-it"
SUBJECT_NAME = "MedGemma-4B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.50.0",
        "accelerate==0.34.2",
        "Pillow",
        "tqdm",
        "numpy<2",
        "sentencepiece",
        "protobuf",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-medgemma-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    cpu=4,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_medgemma(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        FINDINGS_PROMPT,
        atomic_write_json,
        build_findings_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, f"daphne_medgemma_train1_{n_images}.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[MedGemma] resume: {len(done)} already done.", flush=True)

    print(
        f"[MedGemma] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[MedGemma] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[MedGemma] HF token loaded.", flush=True)
    else:
        print("[MedGemma] WARNING: no HF_TOKEN; gated model load will fail.", flush=True)
    print(f"[MedGemma] loading {MODEL_ID} ...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE, token=hf_token
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=MODEL_CACHE,
        token=hf_token,
    ).eval()
    print("[MedGemma] model loaded.", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="MedGemma")):
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": FINDINGS_PROMPT},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        record = build_findings_record(
            subject=SUBJECT_NAME,
            item_id=i,
            rel_path=rel_path,
            response=response,
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
    print(f"\n[MedGemma] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}
