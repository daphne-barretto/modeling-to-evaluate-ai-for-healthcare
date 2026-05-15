import os, sys
# Ensure project root is on sys.path so 'data' can be imported when running from any CWD
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
from data.chexpert_dataset import iter_batched, load_sample_ids

OUTPUT_PATH = "outputs/shannon_outputs.json"

# TODO: implement model-specific inference
def call_your_model(image):
    raise NotImplementedError

def parse_response(raw):
    raise NotImplementedError


def main():
    sample_ids = load_sample_ids()
    results = {}
    for batch in iter_batched(batch_size=4, sample_ids=sample_ids):
        for item in batch:
            image = item["image"]
            item_id = item["item_id"]
            raw = call_your_model(image)
            preds = parse_response(raw)
            results[item_id] = preds
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
    print(f"Done. {len(results)} items written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
