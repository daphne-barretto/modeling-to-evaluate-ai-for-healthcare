"""
Modal script to unzip CheXpert train batches inside chexpert-vol-v2.
Run with: modal run unzip_chexpert.py
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", version=2)
VOLUME_PATH = "/data"
BASE = f"{VOLUME_PATH}/CheXpert/chexpertchestxrays-u20210408"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)

app = modal.App("chexpert-unzip", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 6,
    memory=16384,
    cpu=4,
    ephemeral_disk=1024 * 1024,  # 1 TiB scratch
)
def unzip_batch(batch_filename: str):
    import zipfile
    import os
    from tqdm import tqdm

    zip_path = os.path.join(BASE, batch_filename)
    out_dir  = os.path.join(BASE, "extracted")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Not found: {zip_path}")

    size = os.path.getsize(zip_path)
    print(f"Opening: {batch_filename} ({size:,} bytes)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"Files inside: {len(members)}")
        print(f"Sample: {members[:3]}")
        for member in tqdm(members, desc=batch_filename[:30]):
            zf.extract(member, out_dir)

    print(f"✓ Extracted {len(members)} files → {out_dir}")
    volume.commit()
    print(f"✓ Committed to volume")


@app.local_entrypoint()
def main():
    batches = [
        "CheXpert-v1.0 batch 2 (train 1).zip",
        "CheXpert-v1.0 batch 3 (train 2).zip",
        "CheXpert-v1.0 batch 4 (train 3).zip",
    ]

    for batch in batches:
        print(f"\n── Unzipping: {batch} ──")
        unzip_batch.remote(batch)

    print("\n✓ All batches done.")