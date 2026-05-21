import modal

volume = modal.Volume.from_name("chexpert-vol")
app = modal.App("chexpert-ls", image=modal.Image.debian_slim())

@app.function(volumes={"/data": volume})
def list_files():
    import os
    base = "/data/CheXpert/chexpertchestxrays-u20210408"
    for f in sorted(os.listdir(base)):
        full = os.path.join(base, f)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"[{kind}] {repr(f)}  ({size:,} bytes)")

@app.local_entrypoint()
def main():
    list_files.remote()