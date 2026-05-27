"""Modal: run LLaVA-1.5-7B (NON-medical baseline) on the first 10K frontal train1 rows.

Paired with `run_llava_med_modal.py` for a clean ablation: same base
architecture (Vicuna-7B + CLIP-ViT-L/14) but trained on general image
data rather than PMC medical images. The difference in θ̂ isolates the
contribution of medical fine-tuning under IRT.

Uses the free-text findings prompt + synonym parser (LLaVA-1.5 also
echoes structured templates).

Run (detached):
    modal run --detach \\
        modal_runs/run_llava15_modal.py::run_llava15
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("llava15-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SUBJECT_NAME = "LLaVA-1.5-7B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.45.2",
        "accelerate==0.30.0",
        "Pillow",
        "tqdm",
        "numpy<2",
        "sentencepiece",
        "protobuf",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("llava15-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    cpu=4,
)
def run_llava15(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        FINDINGS_PROMPT,
        atomic_write_json,
        build_findings_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, "llava-1.5-7b.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[LLaVA-1.5] resume: {len(done)} already done.", flush=True)

    print(f"[LLaVA-1.5] selecting first {n_images} frontal train rows...", flush=True)
    rows = select_frontal_train_rows(n_images)
    print(f"[LLaVA-1.5] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[LLaVA-1.5] loading {MODEL_ID} ...", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, cache_dir=MODEL_CACHE
    ).to("cuda").eval()
    print("[LLaVA-1.5] model loaded.", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="LLaVA-1.5")):
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
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": FINDINGS_PROMPT},
                    ],
                }
            ]
            prompt_str = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = processor(
                images=pil_image, text=prompt_str, return_tensors="pt"
            ).to("cuda", torch.float16)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            response = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
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
    print(f"\n[LLaVA-1.5] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}
