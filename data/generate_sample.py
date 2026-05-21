import json, random
from chexpert_dataset import iter_chexpert

random.seed(42)

if __name__ == "__main__":
    print("Streaming item IDs (this may take a few minutes)...")
    # WARNING: this will stream the dataset and collect all ids locally once.
    all_ids = [item["item_id"] for item in iter_chexpert(split="validation")]
    sample = random.sample(all_ids, k=200)
    with open("data/sample_ids.json", "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Saved {len(sample)} item IDs to data/sample_ids.json")
