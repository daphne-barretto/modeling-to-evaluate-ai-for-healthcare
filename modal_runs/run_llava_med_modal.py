"""Modal: run LLaVA-Med v1.5 (Mistral-7B) on the first 10K frontal train1 rows.

Covers the test-taker slot originally assigned to Shannon's first model.
Image selection matches Qwen 3B/7B and the GPT runs (same `select_frontal_train_rows`)
so the response matrix joins cleanly across subjects.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach modal_runs/run_llava_med_modal.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("llava-med-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"
SUBJECT_NAME = "LLaVA-Med-7B"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "numpy<2",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.36.2",
        "tokenizers>=0.15.0",
        "sentencepiece==0.1.99",
        "shortuuid",
        "accelerate==0.21.0",
        "peft==0.4.0",
        "bitsandbytes==0.41.0",
        "pydantic<2,>=1",
        "protobuf",
        "scikit-learn==1.2.2",
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.9.12",
        "Pillow",
        "tqdm",
    )
    .pip_install("git+https://github.com/microsoft/LLaVA-Med.git")
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-llava-med-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A100",
    timeout=60 * 60 * 6,
    memory=32768,
    cpu=4,
)
def run_llava_med(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        FINDINGS_PROMPT,
        atomic_write_json,
        build_findings_record,
        load_existing,
        summarize,
    )

    output_path = os.path.join(
        OUTPUTS_DIR, f"daphne_llava_med_train1_{n_images}.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[LLaVA-Med] resume: {len(done)} already done.", flush=True)

    print(
        f"[LLaVA-Med] selecting first {n_images} frontal train rows...", flush=True
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[LLaVA-Med] selected {len(rows)} rows.", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[LLaVA-Med] loading {MODEL_ID} ...", flush=True)

    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_ID)
    tokenizer, model, image_processor, _ctx_len = load_pretrained_model(
        MODEL_ID, None, model_name, device_map="auto"
    )
    model.eval()
    conv_mode = "mistral_instruct"
    print(f"[LLaVA-Med] model loaded; conv_mode={conv_mode}", flush=True)

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="LLaVA-Med")):
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

        # Build LLaVA-Med prompt (free-text findings; structured yes/no
        # prompts cause the model to echo the template back unchanged).
        qs = FINDINGS_PROMPT
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        image_tensor = process_images(
            [pil_image], image_processor, model.config
        )[0]

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    num_beams=1,
                    max_new_tokens=256,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
            stop_str = (
                conv.sep
                if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2
            )
            if stop_str and response.endswith(stop_str):
                response = response[: -len(stop_str)].strip()
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
    print(f"\n[LLaVA-Med] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}


@app.local_entrypoint()
def main(n_images: int = 10_000, save_every: int = 100):
    print(f"── Spawning LLaVA-Med on {n_images} frontal train1 rows ──")
    call = run_llava_med.spawn(n_images=n_images, save_every=save_every)
    print(f"✓ Spawned function call: {call.object_id}")
    print("  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/daphne_llava_med_train1_{n_images}.json")
