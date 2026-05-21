import modal

volume = modal.Volume.from_name("chexpert-vol")
app = modal.App("chexpert-inspect", image=modal.Image.debian_slim())

@app.function(volumes={"/data": volume})
def inspect():
    import os
    base = "/data/CheXpert/chexpertchestxrays-u20210408"
    prefix = ".azDownload-9353752f-ad06-fe4b-6cdb-871a9529d4d0-"
    path = f"{base}/{prefix}CheXpert-v1.0 batch 2 (train 1).zip"
    with open(path, "rb") as f:
        header = f.read(32)
    print("hex:", header.hex())
    print("raw:", header)

@app.local_entrypoint()
def main():
    inspect.remote()