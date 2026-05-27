"""Modal: run CheXagent-8b (Stanford AIMI) on the first 10K frontal train1 rows.

CheXagent-8b is Stanford's CXR foundation model. Public on HF, requires
`trust_remote_code=True` because it ships custom model code.

Uses CheXagent's native "disease identification" prompt format
(`'Given the CXR, identify any diseases. Options: ...'`) and parses the
free-form response by checking whether each pathology name appears.
This matches how the model was trained, in contrast to forcing the model
into a 14-condition yes/no list format it does not follow.

Run (detached):
    MODAL_PROFILE=daphne-personal modal run --detach modal_runs/run_chexagent_modal.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
model_vol = modal.Volume.from_name("chexagent-model-cache", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_CACHE = "/model-cache"
MODEL_ID = "StanfordAIMI/CheXagent-8b"
SUBJECT_NAME = "CheXagent-8B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.36,<4.45",
        "accelerate",
        "sentencepiece",
        "Pillow",
        "tqdm",
        "einops",
    )
    .add_local_python_source("_open_vlm_helpers", "_helpers")
)

app = modal.App("daphne-chexagent-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume, MODEL_CACHE: model_vol},
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    cpu=4,
)
def run_chexagent(n_images: int = 10_000, save_every: int = 100):
    import os
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        GenerationConfig,
    )

    from _helpers import OUTPUTS_DIR, select_frontal_train_rows
    from _open_vlm_helpers import (
        PATHOLOGIES,
        atomic_write_json,
        load_existing,
        summarize,
    )

    # CheXagent-8b is trained for short focused queries; the
    # multi-disease "list" framing makes it default to "No Finding".
    # We instead ask for a findings narrative and then check whether
    # each pathology name (or a common synonym) appears.
    PROMPT = (
        "Generate the findings section for this chest X-ray. "
        "Be specific about which conditions are present or absent."
    )

    # Synonym lookups so terms like "heart enlargement" / "cardiac enlargement"
    # map to Cardiomegaly, etc.  Keys must be lowercase substrings of the
    # raw response that imply the listed pathology.
    PATHOLOGY_SYNONYMS = {
        "Enlarged Cardiomediastinum": [
            "enlarged cardiomediastinum",
            "widened mediastinum",
            "mediastinal widening",
        ],
        "Cardiomegaly": [
            "cardiomegaly",
            "heart enlargement",
            "cardiac enlargement",
            "enlarged heart",
            "enlarged cardiac silhouette",
        ],
        "Lung Opacity": [
            "lung opacity",
            "pulmonary opacity",
            "opacities",
            "opacification",
        ],
        "Lung Lesion": ["lung lesion", "pulmonary lesion", "nodule", "mass"],
        "Edema": ["edema", "oedema", "pulmonary congestion"],
        "Consolidation": ["consolidation"],
        "Pneumonia": ["pneumonia", "infection", "infectious process"],
        "Atelectasis": ["atelectasis", "collapse"],
        "Pneumothorax": ["pneumothorax"],
        "Pleural Effusion": ["pleural effusion", "effusion"],
        "Pleural Other": [
            "pleural thickening",
            "pleural scarring",
            "calcified pleura",
        ],
        "Fracture": ["fracture"],
        "Support Devices": [
            "support device",
            "tube",
            "line",
            "catheter",
            "pacemaker",
            "wire",
            "icd",
            "hardware",
        ],
        "No Finding": [
            "no finding",
            "no findings",
            "no acute",
            "normal",
            "unremarkable",
            "no abnormalit",
        ],
    }
    NEG_PATTERNS = [
        "no ",
        "without ",
        "absent",
        "negative for ",
        "no evidence of ",
        "no significant ",
    ]

    def build_chexagent_record(item_id: int, rel_path: str, response: str, row: dict) -> dict:
        """Parse CheXagent's free-form findings response using synonym + negation."""
        record: dict = {
            "image_path": rel_path,
            "subject": SUBJECT_NAME,
            "item_id": str(item_id),
            "raw_response": response.strip(),
        }
        resp_lower = response.lower()
        responded = bool(resp_lower.strip())
        n_valid = 0
        n_correct = 0

        def is_mentioned(p: str) -> bool | None:
            """True if pathology positively mentioned, False if negated, None if absent."""
            terms = PATHOLOGY_SYNONYMS.get(p, [p.lower()])
            for term in terms:
                idx = resp_lower.find(term)
                if idx == -1:
                    continue
                # Check 25-char left context for negation
                left = resp_lower[max(0, idx - 25):idx]
                if any(neg in left for neg in NEG_PATTERNS):
                    return False
                return True
            return None

        for p in PATHOLOGIES:
            gt_raw = row.get(p, "")
            try:
                gt_val = (
                    float(gt_raw)
                    if str(gt_raw).strip() not in ("", "nan")
                    else None
                )
            except (TypeError, ValueError):
                gt_val = None
            if gt_val is None or gt_val == -1.0:
                record[f"{p}__gt"] = None
                record[f"{p}__answer"] = None
                record[f"{p}__correct"] = None
                continue
            if not responded:
                record[f"{p}__gt"] = gt_val
                record[f"{p}__answer"] = "unclear"
                record[f"{p}__correct"] = None
                continue

            mention = is_mentioned(p)
            if p == "No Finding":
                # If the response itself is healthy-language, pred=True; if any
                # other pathology was positively mentioned, pred=False.
                positive_findings = any(
                    is_mentioned(q) is True
                    for q in PATHOLOGIES
                    if q != "No Finding"
                )
                pred = (mention is True) or (
                    mention is None and not positive_findings
                )
            else:
                # Pathology mentioned positively => pred=True;
                # mentioned and negated => pred=False;
                # not mentioned at all => pred=False (assume absent if model didn't note).
                if mention is True:
                    pred = True
                else:
                    pred = False

            gt_pos = gt_val == 1.0
            correct = int(pred == gt_pos)
            record[f"{p}__gt"] = gt_val
            record[f"{p}__answer"] = "yes" if pred else "no"
            record[f"{p}__correct"] = correct
            n_valid += 1
            n_correct += correct
        record["n_valid"] = n_valid
        record["n_correct"] = n_correct
        record["all_correct"] = (
            int(n_correct == n_valid) if n_valid > 0 else None
        )
        return record

    output_path = os.path.join(
        OUTPUTS_DIR, "chexagent-8b.json"
    )
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    records, done = load_existing(output_path)
    print(f"[CheXagent] resume: {len(done)} already done.", flush=True)

    print(
        f"[CheXagent] selecting first {n_images} frontal train rows...",
        flush=True,
    )
    rows = select_frontal_train_rows(n_images)
    print(f"[CheXagent] selected {len(rows)} rows.", flush=True)

    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
    print(f"[CheXagent] loading {MODEL_ID} ...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=MODEL_CACHE
    )
    gen_config = GenerationConfig.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE
    )
    gen_config.max_new_tokens = 180
    gen_config.do_sample = False
    gen_config.temperature = 1.0
    gen_config.num_beams = 1
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
    ).to("cuda")
    model.eval()
    print("[CheXagent] model loaded.", flush=True)

    formatted_prompt = f" USER: <s>{PROMPT} ASSISTANT: <s>"

    n_processed = 0
    n_errors = 0

    for i, row in enumerate(tqdm(rows, desc="CheXagent")):
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
            inputs = processor(
                images=[pil_image],
                text=formatted_prompt,
                return_tensors="pt",
            ).to(device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, generation_config=gen_config
                )[0]
            response = processor.tokenizer.decode(
                output, skip_special_tokens=True
            )
            # Strip the prompt-echo prefix if present
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:", 1)[-1]
            response = response.strip()
        except Exception as e:
            print(f"  [error] {rel_path}: {e}", flush=True)
            n_errors += 1
            continue

        record = build_chexagent_record(
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
    print(f"\n[CheXagent] complete: {len(records)} records, {n_errors} errors")
    s = summarize(records)
    print(f"  all_correct rate: {s.get('all_correct_rate')}")
    return {"records": len(records), "errors": n_errors, "summary": s}


@app.local_entrypoint()
def main(n_images: int = 10_000, save_every: int = 100):
    print(f"── Spawning CheXagent-8B on {n_images} frontal train1 rows ──")
    call = run_chexagent.spawn(n_images=n_images, save_every=save_every)
    print(f"✓ Spawned function call: {call.object_id}")
    print("  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/chexagent-8b.json")
