"""Modal: run Qwen2.5-VL-32B-Instruct on the first 10K frontal train1 rows.

The "large" end of the Qwen2.5-VL family (3B / 7B / 32B). Together with
``data/inference/qwen2.5-vl-3b.json`` and ``data/inference/qwen2.5-vl-7b.json``
(already on disk) this enables a clean 3-point scaling-law fit for a
single VLM family on chest X-ray reasoning.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach \\
        modal_runs/run_qwen25vl_32b_modal.py::run_qwen25vl_32b
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("qwen25vl-32b-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
SUBJECT_NAME = "Qwen2.5-VL-32B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.49.0",
        "accelerate==0.34.2",
        "qwen-vl-utils==0.0.8",
        "Pillow",
        "tqdm",
        "numpy<2",
        "sentencepiece",
        "protobuf",
        "einops",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-qwen25vl-32b-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    memory=65536,
    cpu=4,
)
def run_qwen25vl_32b(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        FINDINGS_PROMPT,
        atomic_write_json,
        build_findings_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, "qwen2.5-vl-32b.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[Qwen-32B] resume: {len(done)} already done.", flush=True)

    print(
        f"[Qwen-32B] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[Qwen-32B] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[Qwen-32B] loading {MODEL_ID} ...", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=MODEL_CACHE,
    ).eval()
    print("[Qwen-32B] model loaded.", flush=True)

    n_processed = 0
    n_errors = 0

    # Cap image resolution at ~768×768 effective. Default Qwen2.5-VL
    # max_pixels is ~12.8M pixels which causes 30-70 GB activation
    # allocations and OOMs even on an A100-80GB once the 32B weights
    # are loaded (~64 GB bf16).
    QWEN_MAX_PIXELS = 768 * 768
    QWEN_MIN_PIXELS = 256 * 256

    for i, row in enumerate(tqdm(rows, desc="Qwen-32B")):
        rel_path = row["Path"]
        if rel_path in done:
            continue
        img_path = row["_image_path"]
        try:
            # Validate image
            Image.open(img_path).convert("RGB").verify()
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                            "min_pixels": QWEN_MIN_PIXELS,
                            "max_pixels": QWEN_MAX_PIXELS,
                        },
                        {"type": "text", "text": FINDINGS_PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            # Eager-release activation memory between samples.
            del inputs, output_ids, generated
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [error] {rel_path}: {e}", flush=True)
            n_errors += 1
            torch.cuda.empty_cache()
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
    print(f"\n[Qwen-32B] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}
