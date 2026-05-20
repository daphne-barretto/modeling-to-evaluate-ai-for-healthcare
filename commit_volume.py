import modal

volume = modal.Volume.from_name("chexpert-vol")
app = modal.App("chexpert-commit", image=modal.Image.debian_slim())

@app.function(volumes={"/data": volume}, timeout=600)
def commit_volume():
    import os
    base = "/data/CheXpert/chexpertchestxrays-u20210408"
    files = os.listdir(base)
    print(f"Files in volume: {len(files)}")
    for f in files:
        size = os.path.getsize(os.path.join(base, f))
        print(f"  {f}  ({size:,} bytes)")
    volume.commit()
    print("✓ Volume committed.")

@app.local_entrypoint()
def main():
    commit_volume.remote()