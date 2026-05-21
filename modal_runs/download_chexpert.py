"""Download CheXpert from Stanford AIMI into Modal volume `chexpert-vol-v2`.

Prereqs (one-time, in this workspace):
    MODAL_PROFILE=daphne-personal modal secret create chexpert-secret \
        CHEXPERT_SAS_URL="https://..."

Run:
    MODAL_PROFILE=daphne-personal modal run modal/download_chexpert.py

Total size ~471 GB. Allow ~2-6 hours.
"""

import modal

volume = modal.Volume.from_name("chexpert-vol-v2", create_if_missing=True, version=2)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "tar")
    .run_commands(
        "wget -q -O /tmp/azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux",
        "tar -xzf /tmp/azcopy.tar.gz -C /tmp/",
        "mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy",
        "chmod +x /usr/local/bin/azcopy",
    )
)

app = modal.App("chexpert-download", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("chexpert-secret")],
    timeout=60 * 60 * 6,
    memory=8192,
    cpu=4,
    ephemeral_disk=1024 * 1024,
)
def download_chexpert():
    import os
    import subprocess

    sas_url = os.environ["CHEXPERT_SAS_URL"]
    out_dir = os.path.join(VOLUME_PATH, "CheXpert")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting azcopy → {out_dir}")
    result = subprocess.run(
        [
            "azcopy", "copy",
            sas_url,
            out_dir,
            "--recursive=true",
            "--log-level=INFO",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"azcopy failed: exit {result.returncode}")

    print("\nCommitting volume...")
    volume.commit()
    print("✓ Download complete and committed.")


@app.function(volumes={VOLUME_PATH: volume})
def check_volume():
    import os

    total = 0
    for root, dirs, files in os.walk(VOLUME_PATH):
        depth = root.replace(VOLUME_PATH, "").count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or 'data'}/")
        for f in files[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  ... ({len(files)} files total)")
        total += len(files)
    print(f"\nTotal files: {total}")


@app.local_entrypoint()
def main():
    print("Downloading CheXpert from Stanford AIMI → volume 'chexpert-vol-v2'")
    download_chexpert.remote()
    print("\nVolume contents:")
    check_volume.remote()
