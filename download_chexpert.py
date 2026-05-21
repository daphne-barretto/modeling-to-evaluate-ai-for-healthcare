"""
Modal script to create a persistent volume and download
CheXpert from Stanford AIMI (Azure Blob Storage) using azcopy.

Steps before running:
1. Register and accept terms at:
   https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
2. Copy the SAS URL shown on the download page
3. Store it as a Modal secret:
   modal secret create chexpert-secret CHEXPERT_SAS_URL="https://..."
4. Run:
   modal run download_chexpert.py
"""

import modal

# ── Volume (v2) ────────────────────────────────────────────────────────────────
volume = modal.Volume.from_name("chexpert-vol-v2", create_if_missing=True, version=2)
VOLUME_PATH = "/data"

# ── Image — install azcopy ─────────────────────────────────────────────────────
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

# ── App ────────────────────────────────────────────────────────────────────────
app = modal.App("chexpert-download", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("chexpert-secret")],
    timeout=60 * 60 * 6,    # 6 hours
    memory=8192,
    cpu=4,
    ephemeral_disk=1024 * 1024,  # 1 TiB local scratch — prevents ENOSPC
)
def download_chexpert():
    import os
    import subprocess

    sas_url = os.environ["CHEXPERT_SAS_URL"]
    out_dir = os.path.join(VOLUME_PATH, "CheXpert")
    os.makedirs(out_dir, exist_ok=True)

    print("Starting azcopy download from Stanford AIMI...")
    print(f"Destination: {out_dir}")

    result = subprocess.run(
        [
            "azcopy", "copy",
            sas_url,
            out_dir,
            "--recursive=true",
            "--log-level=INFO",
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"azcopy failed with return code {result.returncode}")

    print("\nCommitting to volume...")
    volume.commit()
    print("✓ Download complete and committed.")


@app.function(
    volumes={VOLUME_PATH: volume},
)
def check_volume():
    """List top-level contents and file count."""
    import os

    total_files = 0
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
        total_files += len(files)

    print(f"\nTotal files found: {total_files}")


@app.local_entrypoint()
def main():
    print("Downloading CheXpert from Stanford AIMI → Modal volume 'chexpert-vol-v2'")
    download_chexpert.remote()
    print("\nChecking volume contents...")
    check_volume.remote()