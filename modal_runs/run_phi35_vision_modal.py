"""Modal: run Phi-3.5-vision-instruct on the first 10K frontal train1 rows.

Phi-3.5-vision is Microsoft's open-weights 4.2B-param VLM, fits comfortably on A10G.
Uses the structured 14-pathology yes/no prompt from `_open_vlm_helpers.PROMPT`.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach modal_runs/run_phi35_vision_modal.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("phi35-vision-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
SUBJECT_NAME = "Phi-3.5-Vision"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.43.0",
        "accelerate==0.30.0",
        "Pillow",
        "tqdm",
        "numpy<2",
        "flash-attn==2.6.3",
        "einops",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-phi35-vision-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    cpu=4,
)
def run_phi35_vision(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoProcessor

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        PROMPT,
        atomic_write_json,
        build_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, f"daphne_phi35_vision_train1_{n_images}.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[Phi-3.5-Vision] resume: {len(done)} already done.", flush=True)

    print(
        f"[Phi-3.5-Vision] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[Phi-3.5-Vision] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[Phi-3.5-Vision] loading {MODEL_ID} ...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        num_crops=4,
        cache_dir=MODEL_CACHE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
        _attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    print("[Phi-3.5-Vision] model loaded.", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="Phi-3.5-Vision")):
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
                {"role": "user", "content": f"<|image_1|>\n{PROMPT}"},
            ]
            prompt_str = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(prompt_str, [pil_image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            # Strip the prompt tokens out
            response = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
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
    print(f"\n[Phi-3.5-Vision] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}


@app.local_entrypoint()
def main(n_images: int = 10_000, save_every: int = 100):
    print(f"── Spawning Phi-3.5-Vision on {n_images} frontal train1 rows ──")
    call = run_phi35_vision.spawn(n_images=n_images, save_every=save_every)
    print(f"✓ Spawned function call: {call.object_id}")
    print("  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/daphne_phi35_vision_train1_{n_images}.json")
