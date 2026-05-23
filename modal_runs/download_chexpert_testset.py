"""One-shot Modal job: stream the CheXpert test-set images and labels CSV
from Stanford AIMI's chexlocalize Azure blob container directly onto our
chexpert-vol-v2 Modal volume.

Target layout on the volume::

    CheXpert/chexpertchestxrays-u20210408/extracted/CheXpert-v1.0/test/
        patient64741/study1/view1_frontal.jpg
        patient64742/study1/view1_frontal.jpg
        ...
    CheXpert/chexpertchestxrays-u20210408/test_labels.csv

The layout matches what runners already expect (their image_root is
``.../extracted/CheXpert-v1.0`` so a ``test/patient.../view.jpg`` path
resolves naturally).

Run with::

    SAS_URL='https://...'  \
    MODAL_PROFILE=daphne-personal modal run modal_runs/download_chexpert_testset.py
"""
from __future__ import annotations

import os
import re

import modal

app = modal.App("daphne-download-chexpert-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests==2.32.3", "tqdm==4.66.4")
)

vol = modal.Volume.from_name("chexpert-vol-v2")

CONTAINER_BASE = (
    "https://aimistanforddatasets01.blob.core.windows.net/chexlocalize"
)
PREFIX_IMAGES = "CheXpert/test/"
PREFIX_LABELS = "CheXpert/test_labels.csv"


def _build_blob_url(sas_url: str, blob_name: str) -> str:
    base, qs = sas_url.split("?", 1)
    base = base.rstrip("/")
    return f"{base}/{blob_name}?{qs}"


def _list_image_blobs(sas_url: str) -> list[str]:
    """List all .jpg blobs under CheXpert/test/ in the SAS container."""
    import requests

    sep = "&" if "?" in sas_url else "?"
    list_url = (
        sas_url
        + f"{sep}restype=container&comp=list&prefix={PREFIX_IMAGES}"
        f"&maxresults=10000"
    )
    r = requests.get(list_url, timeout=120)
    r.raise_for_status()
    names = re.findall(r"<Name>([^<]+)</Name>", r.text)
    # Filter to .jpg only (defensive).
    return [n for n in names if n.lower().endswith(".jpg")]


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_dict({"SAS_URL": os.environ.get("SAS_URL", "")})],
)
def download_test_set() -> None:
    import requests
    from tqdm import tqdm

    sas_url = os.environ.get("SAS_URL")
    if not sas_url:
        raise RuntimeError(
            "SAS_URL env var must be set; pass it through with "
            "`SAS_URL='...' modal run ...` so the Modal secret can capture it."
        )

    # Target dirs on the mounted volume.
    image_root = "/data/CheXpert/chexpertchestxrays-u20210408/extracted/CheXpert-v1.0"
    labels_dest = "/data/CheXpert/chexpertchestxrays-u20210408/test_labels.csv"

    os.makedirs(image_root, exist_ok=True)

    # --- Labels CSV (small, single file). ---
    labels_url = _build_blob_url(sas_url, PREFIX_LABELS)
    print(f"[download] labels: {labels_url[:120]}...", flush=True)
    r = requests.get(labels_url, timeout=60)
    r.raise_for_status()
    with open(labels_dest, "wb") as f:
        f.write(r.content)
    print(f"[download] wrote {labels_dest} ({len(r.content)} bytes)", flush=True)

    # --- Image blobs. ---
    names = _list_image_blobs(sas_url)
    print(f"[download] {len(names):,} image blobs to fetch", flush=True)

    n_ok = 0
    n_skip = 0
    n_err = 0
    for name in tqdm(names, desc="test images"):
        # Strip the "CheXpert/" prefix so we land at .../CheXpert-v1.0/test/...
        rel = name[len("CheXpert/"):] if name.startswith("CheXpert/") else name
        dest = os.path.join(image_root, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            n_skip += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            url = _build_blob_url(sas_url, name)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                tmp = dest + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dest)
            n_ok += 1
        except Exception as e:
            print(f"  [err] {name}: {e}", flush=True)
            n_err += 1

    vol.commit()
    print(
        f"[download] done. ok={n_ok}  skipped={n_skip}  err={n_err}",
        flush=True,
    )


@app.local_entrypoint()
def main() -> None:
    sas = os.environ.get("SAS_URL")
    if not sas:
        raise SystemExit("Set SAS_URL env var locally before invoking.")
    download_test_set.remote()
