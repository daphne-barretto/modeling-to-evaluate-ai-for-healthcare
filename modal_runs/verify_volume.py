"""
Utility script to verify Modal volume access and list dataset structure.

Usage:
    modal run verify_volume.py
"""

import modal
import os

app = modal.App("chexpert-verify-volume")
volume = modal.Volume.from_name("chexpert-vol-v2")


@app.function(volumes={"/data": volume})
def check_volume():
    """List top-level contents and verify dataset path."""
    volume_root = "/data"
    dataset_dir = "CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)"
    labels_csv = "CheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv"

    print("=== Volume root contents ===")
    for item in sorted(os.listdir(volume_root))[:20]:
        print(f"  {item}")

    # Check dataset directory
    dataset_path = os.path.join(volume_root, dataset_dir)
    print(f"\n=== Dataset path: {dataset_path} ===")
    if os.path.exists(dataset_path):
        items = sorted(os.listdir(dataset_path))[:10]
        print(f"  Found {len(os.listdir(dataset_path))} items. First 10:")
        for item in items:
            print(f"    {item}")
            # Show contents of first patient dir
            sub_path = os.path.join(dataset_path, item)
            if os.path.isdir(sub_path):
                for sub in sorted(os.listdir(sub_path))[:3]:
                    sub_sub = os.path.join(sub_path, sub)
                    if os.path.isdir(sub_sub):
                        files = os.listdir(sub_sub)
                        print(f"      {sub}/ -> {files[:5]}")
    else:
        print("  NOT FOUND!")
        # Try to find what exists
        parent = os.path.dirname(dataset_path)
        while not os.path.exists(parent) and parent != volume_root:
            parent = os.path.dirname(parent)
        if os.path.exists(parent):
            print(f"  Nearest existing path: {parent}")
            print(f"  Contents: {sorted(os.listdir(parent))[:10]}")

    # Check labels CSV
    csv_path = os.path.join(volume_root, labels_csv)
    print(f"\n=== Labels CSV: {csv_path} ===")
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"  Found! Size: {size_mb:.1f} MB")
        # Show first few lines
        with open(csv_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line.rstrip()}")
    else:
        print("  NOT FOUND!")
        parent = os.path.dirname(csv_path)
        if os.path.exists(parent):
            print(f"  Contents of {parent}: {sorted(os.listdir(parent))[:10]}")

    # Count total images
    print("\n=== Counting images (first 100 patients) ===")
    if os.path.exists(dataset_path):
        count = 0
        patients = sorted(os.listdir(dataset_path))[:100]
        for patient in patients:
            patient_path = os.path.join(dataset_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                for f in os.listdir(study_path):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        count += 1
        print(f"  Images in first 100 patients: {count}")


@app.local_entrypoint()
def main():
    check_volume.remote()
