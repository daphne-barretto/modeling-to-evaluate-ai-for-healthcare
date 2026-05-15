from datasets import load_dataset
from typing import Any, Iterable
import json

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]


def load_sample_ids(path: str = "data/sample_ids.json") -> set:
    """Load the fixed set of item IDs everyone uses."""
    with open(path) as f:
        return set(json.load(f))


def _load_hf_stream(split: str = "train") -> Iterable[Any]:
    """Load a streaming HuggingFace dataset directly."""
    return load_dataset("danjacobellis/chexpert", split=split, streaming=True)


def iter_chexpert(
    split: str = "train",
    sample_ids: set | None = None,
) -> Iterable[dict]:
    """Stream CheXpert one item at a time as a plain dict.
    """
    ds = _load_hf_stream(split)

    for example in ds:
        item_id = example.get("Path")

        if sample_ids is not None and item_id not in sample_ids:
            continue

        labels = {p: example.get(p, -1) for p in PATHOLOGIES}

        yield {
            "item_id": item_id,
            "image": example.get("image"),
            "labels": labels,
            "metadata": {
                "age": example.get("Age"),
                "sex": example.get("Sex"),
                "view": example.get("Frontal/Lateral"),
            },
        }


def iter_batched(batch_size: int = 8, **kwargs):
    """Yield lists of items for batched inference."""
    batch = []
    for item in iter_chexpert(**kwargs):
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
