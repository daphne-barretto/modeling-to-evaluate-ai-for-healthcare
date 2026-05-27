"""Modal: run CheXagent-2-3b (Stanford AIMI) on the first 10K frontal train1 rows.

Paired with `run_chexagent_modal.py` (8B) for a within-family Stanford-AIMI
scaling-law fit. Otherwise identical pipeline to the 8B runner.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach \\
        modal_runs/run_chexagent3b_modal.py::run_chexagent3b
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("chexagent3b-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
SUBJECT_NAME = "CheXagent-2-3B"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.40.0",
        "accelerate>=0.30,<0.35",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "tqdm",
        "einops",
        "opencv-python-headless",
        "albumentations",
        "pyarrow",
        "matplotlib",
        "numpy<2",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-chexagent3b-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    cpu=4,
)
def run_chexagent3b(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        FINDINGS_PROMPT,
        atomic_write_json,
        build_findings_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, "chexagent-2-3b.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[CheXagent-3B] resume: {len(done)} already done.", flush=True)

    print(
        f"[CheXagent-3B] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[CheXagent-3B] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[CheXagent-3B] loading {MODEL_ID} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=MODEL_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
    )
    model = model.to(torch.bfloat16)
    model.eval()
    print("[CheXagent-3B] model loaded.", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="CheXagent-3B")):
        rel_path = row["Path"]
        if rel_path in done:
            continue
        img_path = row["_image_path"]
        try:
            # Validate (CheXagent-3B opens the image itself via the path)
            Image.open(img_path).convert("RGB").verify()
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        try:
            query = tokenizer.from_list_format([
                {"image": img_path},
                {"text": FINDINGS_PROMPT},
            ])
            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt"
            )
            with torch.inference_mode():
                output = model.generate(
                    input_ids.to(model.device),
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=256,
                )[0]
            response = tokenizer.decode(
                output[input_ids.size(1):-1]
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
    print(f"\n[CheXagent-3B] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}
