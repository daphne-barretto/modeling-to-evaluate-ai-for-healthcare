import json, csv

OUTPUT_FILES = {
    "GPT-4o": "outputs/daphne_outputs.json",
    "Qwen-large": "outputs/izhan_outputs.json",
    "LLaVA-Med": "outputs/shannon_outputs.json",
}

all_outputs = {}
for model_name, path in OUTPUT_FILES.items():
    with open(path) as f:
        all_outputs[model_name] = json.load(f)

item_ids = set.intersection(*[set(v.keys()) for v in all_outputs.values()])
print(f"{len(item_ids)} items present in all output files")

with open("outputs/response_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model"] + sorted(item_ids))
    for model_name, outputs in all_outputs.items():
        row = [model_name]
        for item_id in sorted(item_ids):
            row.append(outputs.get(item_id, {}).get("correct", 0))
        writer.writerow(row)

print("Response matrix written to outputs/response_matrix.csv")
