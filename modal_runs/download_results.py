"""
Post-processing: Compile checkpoint JSONs into a single results file,
save the compiled result to Modal volume, and download locally.

Usage:
    modal run download_results.py
"""

import modal
import json
import os

app = modal.App("chexpert-download-results")
volume = modal.Volume.from_name("chexpert-vol-v2")

LLAMA_OUTPUT = "inference_outputs/llama_vision_outputs.json"
LLAMA_CKPT_DIR = "inference_outputs/checkpoints"
PIXTRAL_OUTPUT = "inference_outputs/pixtral_outputs.json"
PIXTRAL_CKPT_DIR = "inference_outputs/pixtral_checkpoints"


@app.function(volumes={"/data": volume})
def compile_and_get_results(model: str) -> str:
    """Read all checkpoint JSONs for a model, compile into one, save to volume, and return."""
    if model == "llama":
        output_path = os.path.join("/data", LLAMA_OUTPUT)
        ckpt_dir = os.path.join("/data", LLAMA_CKPT_DIR)
    elif model == "pixtral":
        output_path = os.path.join("/data", PIXTRAL_OUTPUT)
        ckpt_dir = os.path.join("/data", PIXTRAL_CKPT_DIR)
    else:
        return "{}"

    compiled = {}

    # Load all checkpoint files in order and merge
    if os.path.exists(ckpt_dir):
        ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".json"))
        print(f"Found {len(ckpt_files)} checkpoint files in {ckpt_dir}")
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            with open(ckpt_path, "r") as f:
                data = json.load(f)
            compiled.update(data)
            print(f"  {ckpt_file}: {len(data)} entries (total: {len(compiled)})")
    else:
        print(f"No checkpoint directory found at {ckpt_dir}")

    # Also check for a final output file and merge (may have entries beyond last checkpoint)
    if os.path.exists(output_path):
        print(f"Found final output file: {output_path}")
        with open(output_path, "r") as f:
            final_data = json.load(f)
        compiled.update(final_data)
        print(f"  Final file: {len(final_data)} entries (total: {len(compiled)})")

    if not compiled:
        print("No results found!")
        return "{}"

    # Save compiled result back to volume
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(compiled, f, indent=2)
    volume.commit()
    print(f"Compiled {len(compiled)} results saved to {output_path}")

    return json.dumps(compiled)


@app.local_entrypoint()
def main():
    """Compile checkpoints and download results for both models."""
    for model, local_file in [("llama", "llama_vision_outputs.json"), ("pixtral", "pixtral_outputs.json")]:
        print(f"\n{'='*60}")
        print(f"Processing {model} results...")
        print(f"{'='*60}")

        results_json = compile_and_get_results.remote(model)
        results = json.loads(results_json)

        if not results:
            print(f"No results found for {model}.")
            continue

        print(f"Downloaded {len(results)} image results for {model}")

        with open(local_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {local_file}")
