"""Unzip a single CheXpert batch (train 1 by default) inside `chexpert-vol-v2`.

Run:
    modal run modal_runs/unzip_chexpert.py
    # to unzip a different batch:
    modal run modal_runs/unzip_chexpert.py --batch "CheXpert-v1.0 batch 3 (train 2).zip"
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
VOLUME_PATH = "/data"
BASE = f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408"

image = modal.Image.debian_slim(python_version="3.11").pip_install("tqdm")

app = modal.App("chexpert-unzip", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 6,
    memory=16384,
    cpu=4,
    ephemeral_disk=1024 * 1024,
)
def unzip_batch(batch_filename: str):
    import os
    import zipfile

    from tqdm import tqdm

    zip_path = os.path.join(BASE, batch_filename)
    out_dir = os.path.join(BASE, "extracted")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Not found: {zip_path}")

    size = os.path.getsize(zip_path)
    print(f"Opening: {batch_filename} ({size:,} bytes)", flush=True)

    skipped = 0
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"Files inside: {len(members)}; sample: {members[:3]}", flush=True)
        for m in tqdm(members, desc=batch_filename[:30]):
            target = os.path.join(out_dir, m)
            # Skip already-extracted files (resume-safe).
            if not m.endswith("/") and os.path.exists(target) and os.path.getsize(target) > 0:
                skipped += 1
                continue
            zf.extract(m, out_dir)
            extracted += 1

    print(
        f"✓ Done. Extracted {extracted}, skipped {skipped} (already present) → {out_dir}",
        flush=True,
    )
    volume.commit()
    print("✓ Committed to volume", flush=True)


@app.local_entrypoint()
def main(batch: str = "CheXpert-v1.0 batch 2 (train 1).zip"):
    print(f"── Spawning unzip for: {batch} ──")
    call = unzip_batch.spawn(batch)
    print(f"✓ Spawned function call: {call.object_id}")
    print(f"  This runs autonomously on Modal; safe to disconnect.")
    print(f"  Monitor: modal app logs <app_id>  (see dashboard)")
    print(f"  Or wait: modal volume ls chexpert-vol-v2 /CheXpert/chexpertchestxrays-u20210408/extracted/")
