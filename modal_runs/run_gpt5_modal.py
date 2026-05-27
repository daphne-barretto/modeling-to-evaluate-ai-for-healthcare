"""Modal: run GPT-5.4 (Azure Responses API) on first 10,000 frontal train1 images.

Run:
    MODAL_PROFILE=daphne-personal modal run modal/run_gpt5_modal.py
    # to override the count:
    MODAL_PROFILE=daphne-personal modal run modal/run_gpt5_modal.py --n-images 100
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.40", "Pillow")
    .add_local_python_source("_helpers")
)

app = modal.App("daphne-gpt5-chexpert", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("azure-openai-creds")],
    timeout=60 * 60 * 12,
    memory=4096,
    cpu=2,
)
def run_gpt5(n_images: int = 10_000, max_workers: int = 8):
    from _helpers import run_modal_inference

    n_done = run_modal_inference(
        deployment="GPT-5.4",
        api_type="responses",
        api_version="2025-04-01-preview",
        output_filename="gpt-5.4.json",
        n_images=n_images,
        extra_kwargs={"max_output_tokens": 2000},
        max_workers=max_workers,
    )
    volume.commit()
    return n_done


@app.local_entrypoint()
def main(n_images: int = 10_000, max_workers: int = 8):
    print(f"── Spawning GPT-5.4 inference on first {n_images} frontal train1 images ──")
    call = run_gpt5.spawn(n_images=n_images, max_workers=max_workers)
    print(f"✓ Spawned function call: {call.object_id}")
    print(f"  Runs autonomously; safe to disconnect.")
    print(f"  Output → /data/outputs/gpt-5.4.json")
    print(f"  Monitor:  modal app logs <app_id>  (see dashboard)")
