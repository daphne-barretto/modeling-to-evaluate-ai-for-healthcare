"""Modal: top up GPT-5.4 (Azure Responses API) from 10K frontal → first 15K
train1 rows in CSV order (frontal + lateral mixed, no view filter).

Pre-seeds ``data/inference/gpt-5.4.json`` with the prior 10K-frontal run so
the resumable loop skips already-done items (keyed by row['Path']) and
processes only the ~5,000 new rows (lateral + a few extra frontal).

Run:
    MODAL_PROFILE=daphne-personal modal run --detach modal_runs/run_gpt5_topup_modal.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.40", "Pillow")
    .add_local_python_source("_helpers")
)

app = modal.App("daphne-gpt5-chexpert-topup15k", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("azure-openai-creds")],
    timeout=60 * 60 * 12,
    memory=4096,
    cpu=2,
)
def run_gpt5_topup(n_images: int = 15_000, max_workers: int = 8):
    from _helpers import run_modal_inference

    n_done = run_modal_inference(
        deployment="GPT-5.4",
        api_type="responses",
        api_version="2025-04-01-preview",
        output_filename="gpt-5.4.json",
        n_images=n_images,
        extra_kwargs={"max_output_tokens": 2000},
        max_workers=max_workers,
        frontal_only=False,
        seed_from="gpt-5.4.json",
    )
    volume.commit()
    return n_done


@app.local_entrypoint()
def main(n_images: int = 15_000, max_workers: int = 8):
    print(
        f"── Spawning GPT-5.4 top-up to {n_images} train1 rows (any view) ──"
    )
    call = run_gpt5_topup.spawn(n_images=n_images, max_workers=max_workers)
    print(f"✓ Spawned function call: {call.object_id}")
    print(f"  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/gpt-5.4.json")
    print(f"  Monitor:  modal app logs <app_id>  (see dashboard)")
