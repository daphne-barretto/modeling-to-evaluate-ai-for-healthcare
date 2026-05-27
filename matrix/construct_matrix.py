"""Construct the J × I response matrix from per-model GPT inference outputs.

Cells are coded:
    1  = model got all 14 pathologies correct (i.e. entry's `correct` field)
    0  = model produced a parseable prediction but it was not fully correct
    "" = MISSING (model refused / filter blocked / unparseable / exception)

Missing items are intentionally left blank so downstream IRT and factor models
can treat them as missing data rather than as wrong answers. A per-model
summary (counts and reasons) is printed so missingness is acknowledged.

NOTE: this script only handles the dict-keyed-by-image Daphne-GPT output
shape (``data/inference/gpt-5.4.json``, ``data/inference/gpt-4o.json``).
For the full 15-model long-form response set used by the IRT pipeline,
use ``analysis/data_loader.load_all()`` instead.
"""

import csv
import json
import os
from collections import Counter

OUTPUT_FILES = {
    "GPT-5.4": "data/inference/gpt-5.4.json",
    "GPT-4o": "data/inference/gpt-4o.json",
}


def _correct_or_missing(entry):
    """Return 1, 0, or None (missing) for a single item entry."""
    if entry is None:
        return None
    if entry.get("error") or entry.get("missing_reason"):
        return None
    # Backward compatibility with outputs written before the diagnostics patch:
    # refusals previously surfaced as empty predictions + empty raw text.
    if entry.get("predictions") in (None, {}) and entry.get("raw") in (None, ""):
        return None
    c = entry.get("correct")
    if c is None:
        return None
    return 1 if int(c) == 1 else 0


def _missing_reason(entry):
    if entry is None:
        return "absent"
    if entry.get("missing_reason"):
        return entry["missing_reason"]
    if entry.get("error"):
        return "exception"
    if entry.get("predictions") in (None, {}) and entry.get("raw") in (None, ""):
        return "legacy_empty"
    return "absent"


all_outputs = {}
for model_name, path in OUTPUT_FILES.items():
    if not os.path.exists(path):
        print(f"  skip {model_name}: {path} not yet produced")
        continue
    with open(path) as f:
        all_outputs[model_name] = json.load(f)

if not all_outputs:
    raise SystemExit("No output files available; nothing to do.")

item_ids = sorted(set.intersection(*[set(v.keys()) for v in all_outputs.values()]))
print(f"\n{len(item_ids)} items present in all available output files\n")

print(f"  {'Model':<14s} {'observed':>9s} {'missing':>8s}  reasons")
for model_name, outputs in all_outputs.items():
    observed = 0
    missing = 0
    reasons: Counter = Counter()
    for iid in item_ids:
        entry = outputs.get(iid)
        if _correct_or_missing(entry) is None:
            missing += 1
            reasons[_missing_reason(entry)] += 1
        else:
            observed += 1
    print(f"  {model_name:<14s} {observed:>9d} {missing:>8d}  {dict(reasons) or '-'}")

out_path = "outputs/response_matrix.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model"] + item_ids)
    for model_name, outputs in all_outputs.items():
        row = [model_name]
        for iid in item_ids:
            v = _correct_or_missing(outputs.get(iid))
            row.append("" if v is None else v)
        writer.writerow(row)

print(f"\nResponse matrix written to {out_path}")
print("Missing cells are left blank so IRT/factor fits treat them as missing,")
print("not as incorrect.")
