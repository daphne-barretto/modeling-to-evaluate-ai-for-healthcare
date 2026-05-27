# Modal CheXpert pipeline (GPT-5.4 + GPT-4o via Azure)

All scripts assume **`MODAL_PROFILE=daphne-personal`** (set per-terminal so other
sessions are unaffected).

## One-time setup
```bash
export MODAL_PROFILE=daphne-personal

# 1) Stanford AIMI SAS URL  → secret `chexpert-secret`
#    (get URL from https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-...)
modal secret create chexpert-secret CHEXPERT_SAS_URL="https://..."

# 2) Azure OpenAI credentials  → secret `azure-openai-creds`
modal secret create azure-openai-creds \
    AZURE_OPENAI_ENDPOINT="https://...openai.azure.com" \
    AZURE_OPENAI_API_KEY="..."
```

## Pipeline
```bash
export MODAL_PROFILE=daphne-personal

# Step 1: download CheXpert  (~2-6 hours, ~471 GB) → volume `chexpert-vol-v2`
modal run modal_runs/download_chexpert.py

# Step 2: unzip train1 only  (~30 min)
modal run modal_runs/unzip_chexpert.py
# (for train2/train3 later:)
# modal run modal_runs/unzip_chexpert.py --batch "CheXpert-v1.0 batch 3 (train 2).zip"

# Step 3: run inference  (parallel, ~hours; resumable)
modal run modal_runs/run_gpt5_modal.py   --n-images 10000
modal run modal_runs/run_gpt4o_modal.py  --n-images 10000

# Step 4: pull results
# Step 4: pull results into local data/inference/<canonical>.json
python modal_runs/download_results.py
```

## Volume layout (inside `chexpert-vol-v2`)
```
/data/
├── CheXpert/chexpertchestxrays-u20210408/
│   ├── CheXpert-v1.0 batch 2 (train 1).zip        # raw download
│   ├── train_visualCheXbert.csv                   # labels (read by inference)
│   ├── ...other CSVs / batches...
│   └── extracted/
│       └── CheXpert-v1.0 batch 2 (train 1)/...    # unzipped images
└── outputs/
    ├── gpt-5.4.json          # canonical filenames matching local data/inference/
    ├── gpt-4o.json
    ├── biomedclip.json
    ├── chexagent-8b.json
    ├── chexagent-2-3b.json
    ├── internvl3-8b.json
    ├── llava-med-7b.json
    ├── llava-1.5-7b.json
    ├── medgemma-4b.json
    ├── phi-3.5-vision.json
    ├── qwen2.5-vl-32b.json
    └── qwen2.5-vl-7b.json
```

## Inference selection rule
`select_frontal_train_rows(n)` in `_helpers.py` walks `train_visualCheXbert.csv`
in CSV order, keeps rows where `Frontal/Lateral == "Frontal"` AND the image
resolves to a file in `extracted/`, and returns the first **n** rows. This is
deterministic across reruns — both models score the exact same images.

## Output format
Resumable JSON dict keyed by CheXpert `Path`. Each value:
```
{
  "deployment": "GPT-5.4" | "gpt-4o",
  "api_type":   "responses" | "chat",
  "predictions": {"No Finding": 0|1, ...},     # null if missing
  "labels":      {"No Finding": "1.0"|...},    # raw CSV strings
  "raw":         "<model JSON>",
  "diagnostics": {...},                         # finish_reason, refusal, usage
  "missing_reason": null | "model_refusal" | "incomplete" | ...,
  "correct":          0 | 1 | null,
  "per_label_correct": {...},
  "binarized_labels":  {...}
}
```
