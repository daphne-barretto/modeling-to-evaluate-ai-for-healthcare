# Probabilistic Modeling for Healthcare AI Evaluation

Daphne Barretto · Izhan Hamza · Shannon Komguem

Stanford University

---

## Repository structure

```
.
├── analysis/                 # IRT / factor fitting, baselines, DIF, figures
│   ├── amortized_irt.py
│   ├── baseline_nll.py
│   ├── baselines.py          # non-IRT reference metrics (acc, F1, AUC, etc.)
│   ├── bifactor_mirt.py      # 2-D MIRT (medical vs. general factor)
│   ├── data_loader.py        # data/inference/*.json -> long-form observations
│   ├── dif_analysis.py       # subgroup gaps (sex, age, view, AP/PA, …)
│   ├── export_manuscript_numbers.py
│   ├── figures.py
│   ├── fit_irt.py            # Rasch / 2PL / 3PL + factor (1F, 2F, 3F)
│   ├── irt_vs_baseline.py
│   ├── item_metadata.py
│   ├── model_fit_metrics.py
│   ├── reliability.py        # bootstrap CI on θ̂
│   ├── scaling_law.py        # within-family θ̂ ≈ α·log10(params) + β
│   └── tinybenchmark.py
│
├── data/
│   ├── chexpert_dataset.py   # CSV-streaming helpers + canonical PATHOLOGIES list
│   ├── chexpert_testset/     # 500-image CheXpert test split + metadata
│   ├── generate_sample.py    # write data/sample_ids.json
│   ├── inference/            # one JSON per test-taker (see below)
│   ├── sample_ids.json
│   └── train_visualCheXbert.csv     # (gitignored; ~8M-row labels file)
│
├── inference/                # Azure-OpenAI entry points (run locally)
│   ├── gpt5_local.py
│   ├── gpt4o_local.py
│   ├── openai_inference.py   # shared Azure helper
│   ├── inference_pixtral.py        # Modal: Pixtral-12B (HF, gated)
│   └── inference_llama_vision.py   # Modal: Llama-3.2-Vision-11B (HF, gated)
│
├── matrix/
│   └── construct_matrix.py   # outputs/response_matrix.csv from data/inference/*.json
│
├── modal_runs/               # Modal launchers for the remaining 13 test-takers
│   ├── README.md
│   ├── _helpers.py
│   ├── _open_vlm_helpers.py
│   ├── download_chexpert.py        # one-time CheXpert download from Stanford AIMI
│   ├── unzip_chexpert.py           # one-time train1 unzip on the volume
│   ├── download_chexpert_testset.py
│   ├── download_results.py         # pulls pixtral / llama outputs locally
│   ├── verify_volume.py
│   ├── run_gpt5_modal.py           # GPT-5.4 (Azure)
│   ├── run_gpt4o_modal.py          # GPT-4o  (Azure)
│   ├── run_gpt5_topup_modal.py     # GPT-5.4 lateral-view + val top-up
│   ├── run_gpt4o_topup_modal.py    # GPT-4o  lateral-view + val top-up
│   ├── run_biomedclip_modal.py     # BiomedCLIP zero-shot (floor anchor)
│   ├── run_chexagent_modal.py      # CheXagent-8B
│   ├── run_chexagent3b_modal.py    # CheXagent-2-3B
│   ├── run_internvl3_modal.py      # InternVL3-8B
│   ├── run_llava15_modal.py        # LLaVA-1.5-7B
│   ├── run_llava_med_modal.py      # LLaVA-Med-7B
│   ├── run_medgemma_modal.py       # MedGemma-4B
│   ├── run_phi35_vision_modal.py   # Phi-3.5-Vision
│   └── run_qwen25vl_32b_modal.py   # Qwen2.5-VL-32B
│
├── outputs/                  # all derived artifacts (committed)
│   ├── baselines/            # CSVs: aggregate / per-pathology / per-subgroup acc
│   ├── irt/                  # fit_table.csv, item_params_*.csv, abilities_*.csv,
│   │                         #   predictions_*.npz, dif_*.csv, headline_findings.json
│   ├── figures/              # PDF + PNG for every manuscript figure
│   ├── response_matrix.csv
│   ├── results_numbers.tex   # \newcommand{...} macros for the manuscript
│   └── tinybenchmark.csv
│
├── scripts/
│   └── refit_all.py          # one-shot driver: runs every analysis module in order
│
├── .env                      # (gitignored) Azure OpenAI creds for local GPT runs
├── .gitignore
├── requirements.txt
└── README.md
```

**`data/inference/`** holds one JSON per test-taker (15 files). Row counts
vary across models — most have ~10,000 frontal train1 images; GPT-5.4 and
GPT-4o additionally include ~5,000 lateral views and 200 val images; Qwen2.5-VL
3B / 7B include ~2,000 lateral views. The IRT loader treats absent cells as
MAR, so cells/rows that a given model never produced (refusal, content
filter, exception, unanswered pathology) are simply dropped from that
test-taker’s contribution.

---

## Setup

### 1. Local Python environment

```bash
git clone https://github.com/daphne-barretto/modeling-to-evaluate-ai-for-healthcare.git
cd modeling-to-evaluate-ai-for-healthcare

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required (we test on 3.12). `requirements.txt` covers only
the local analysis pipeline — the Modal GPU images carry their own pinned
versions of `torch`, `transformers`, `qwen-vl-utils`, `open_clip_torch`, etc.

### 2. Modal account + secrets

You will need a [Modal](https://modal.com) account for the GPU inference
jobs. After `pip install modal`, run `modal token new` once to authenticate,
then export your profile name per-terminal (do **not** use
`modal profile activate`, which rewrites the global config):

```bash
export MODAL_PROFILE=<your-modal-profile>
```

Register three secrets in your Modal workspace (replace placeholders with
your own credentials):

```bash
# 1) Stanford AIMI SAS URL — for downloading CheXpert.
#    Get one by accepting the research agreement at
#    https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
modal secret create chexpert-secret \
    CHEXPERT_SAS_URL="<paste-your-SAS-URL>"

# 2) Azure OpenAI credentials — only needed if you run GPT-5.4 / GPT-4o.
#    Provision the two deployments first in your Azure OpenAI resource.
modal secret create azure-openai-creds \
    AZURE_OPENAI_ENDPOINT="<your-endpoint-url>" \
    AZURE_OPENAI_API_KEY="<your-azure-openai-key>"

# 3) Hugging Face token — needed for gated open-weight VLMs
#    (meta-llama/Llama-3.2-11B-Vision-Instruct, etc.). Generate at
#    https://huggingface.co/settings/tokens and request access to any
#    gated repos you plan to run.
modal secret create huggingface-token \
    HF_TOKEN="<your-hf-token>"
```

### 3. Local `.env` (only for the local GPT entry points)

If you prefer to run GPT-5.4 or GPT-4o on your own machine instead of via
Modal (`inference/gpt5_local.py` / `inference/gpt4o_local.py`), drop a
`.env` file at the repo root with the same Azure credentials:

```env
AZURE_OPENAI_ENDPOINT=<your-endpoint-url>
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
```

`.env` is gitignored.

---

## Reproduce the results

`data/inference/*.json` is already committed in this repo. If you only want
to reproduce the modeling and analyses (Steps 4 and 5 below), you can skip
straight to those steps and use the committed inference outputs. Steps 1–3
re-run all 15 inferences from scratch.

### Step 1 — download and unzip CheXpert (one-time, on the Modal volume)

```bash
# ~2–6 hours, ~471 GB → Modal volume `chexpert-vol-v2`
modal run modal_runs/download_chexpert.py

# ~30 min: unzip the train1 batch we score against
modal run modal_runs/unzip_chexpert.py
```

### Step 2 — run inference (15 test-takers, all parallelisable, all resumable)

Every Modal launcher writes its output to the volume at
`outputs/<canonical-name>.json`, where `<canonical-name>` matches the
filename under `data/inference/`. All launchers default to the first
10,000 frontal train1 images selected by the deterministic rule in
`modal_runs/_helpers.py`.

| Test-taker (canonical filename)                    | Run command                                                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `gpt-5.4.json` (Azure)                             | `modal run modal_runs/run_gpt5_modal.py --n-images 10000` *or* `python inference/gpt5_local.py`      |
| `gpt-4o.json` (Azure)                              | `modal run modal_runs/run_gpt4o_modal.py --n-images 10000` *or* `python inference/gpt4o_local.py`    |
| `biomedclip.json`                                  | `modal run --detach modal_runs/run_biomedclip_modal.py::run_biomedclip`                              |
| `chexagent-8b.json`                                | `modal run --detach modal_runs/run_chexagent_modal.py`                                               |
| `chexagent-2-3b.json`                              | `modal run --detach modal_runs/run_chexagent3b_modal.py`                                             |
| `internvl3-8b.json`                                | `modal run --detach modal_runs/run_internvl3_modal.py`                                               |
| `llava-1.5-7b.json`                                | `modal run --detach modal_runs/run_llava15_modal.py::run_llava15`                                    |
| `llava-med-7b.json`                                | `modal run --detach modal_runs/run_llava_med_modal.py`                                               |
| `medgemma-4b.json`                                 | `modal run --detach modal_runs/run_medgemma_modal.py::run_medgemma`                                  |
| `phi-3.5-vision.json`                              | `modal run --detach modal_runs/run_phi35_vision_modal.py`                                            |
| `qwen2.5-vl-32b.json`                              | `modal run --detach modal_runs/run_qwen25vl_32b_modal.py::run_qwen25vl_32b`                          |
| `pixtral-12b.json` (HF, requires `HF_TOKEN`)       | `modal run inference/inference_pixtral.py`                                                           |
| `llama-3.2-vision-11b.json` (HF, gated repo)       | `modal run inference/inference_llama_vision.py`                                                      |

For long-running detached jobs, monitor via `modal app logs <app-id>`.

### Step 3 — pull the inference outputs back to `data/inference/`

```bash
# Pixtral and Llama-3.2-Vision: stitches per-image checkpoints into a
# single JSON and downloads it.
python modal_runs/download_results.py

# Every other test-taker: pull the per-model JSON from the volume.
for f in gpt-5.4 gpt-4o biomedclip chexagent-8b chexagent-2-3b \
         internvl3-8b llava-1.5-7b llava-med-7b medgemma-4b \
         phi-3.5-vision qwen2.5-vl-32b; do
    modal volume get chexpert-vol-v2 outputs/$f.json data/inference/$f.json
done
```

After this step `data/inference/` should contain 15 JSON files — one per
test-taker.

### Step 4 — construct the response matrix

```bash
python -m matrix.construct_matrix
```

Writes `outputs/response_matrix.csv` — the J × I binary matrix consumed
by the IRT fits (rows = items, columns = test-takers; cells are
1 / 0 / empty for missing).

### Step 5 — fit the full IRT / factor / baseline panel and regenerate every figure

```bash
python scripts/refit_all.py
```

`refit_all.py` is a single-process driver that pays the heavy
`torch` + `torch_measure` import cost once, then runs every analysis
module in order: `fit_irt`, `bifactor_mirt`, `baselines`, `reliability`,
`dif_analysis`, `scaling_law`, `tinybenchmark`, `irt_vs_baseline`,
`amortized_irt`, `figures`, `export_manuscript_numbers`. Each step is
isolated in its own `try`/`except` so a late failure doesn’t lose earlier
work.

To re-run a single stage, invoke it directly — e.g.
`python -m analysis.dif_analysis` or `python -m analysis.figures`.

---

## Outputs map

| Path                                 | What it contains                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `outputs/response_matrix.csv`        | J × I binary response matrix (rows = items, columns = test-takers) used by the IRT fits.           |
| `outputs/results_numbers.tex`        | Auto-generated `\newcommand{...}` macros (counts, accuracies, fit stats) for the manuscript.       |
| `outputs/tinybenchmark.csv`          | tinyBenchmarks-style sub-sampled accuracy estimates.                                               |
| `outputs/baselines/`                 | Non-IRT references: aggregate / per-pathology / per-view / per-subgroup accuracy + P/R/F1.         |
| `outputs/irt/fit_table.csv`          | Headline fit comparison (logLik, AIC, BIC, df, RMSEA, M2, held-out NLL/F1/AUC) across models.      |
| `outputs/irt/item_params_*.csv`      | Per-item parameters (difficulty β, discrimination a, factor loadings) for each fitted model.       |
| `outputs/irt/abilities_*.csv`        | Per-test-taker abilities θ̂ for each fitted model.                                                  |
| `outputs/irt/predictions_*.npz`      | Posterior predictive cell probabilities (gitignored; regenerated on each fit).                     |
| `outputs/irt/dif_*.csv`              | Differential Item Functioning by sex / age / view / AP-PA / anatomical group.                      |
| `outputs/irt/headline_findings.json` | Single-file summary of the headline numbers cited in the report.                                   |
| `outputs/figures/fig_*.{pdf,png}`    | All manuscript figures (caterpillar, ICC examples, item information, factor heatmap, scaling, …).  |

---

## Data access

Raw imaging data is **not** committed to this repo. Source source datasets
require an external research agreement:

| Dataset             | Size        | Conditions | Access                                                                         |
| ------------------- | ----------- | ---------- | ------------------------------------------------------------------------------ |
| CheXpert (Stanford) | 224K images | 14         | [Research agreement](https://stanfordmlgroup.github.io/competitions/chexpert/) |        |

After agreeing to the CheXpert terms, you receive a SAS URL — that is the
value you paste into `modal secret create chexpert-secret CHEXPERT_SAS_URL="..."`
in the Setup section. Images live on the Modal volume `chexpert-vol-v2`
and are never copied to this repository.

---

## Authors

| Name            | Email                 |
| --------------- | --------------------- |
| Daphne Barretto | daphnegb@stanford.edu |
| Izhan Hamza     | izhamza@stanford.edu  |
| Shannon Komguem | skomguem@stanford.edu |

---

## License

This repository is for academic use. Data usage is governed by the
respective dataset licenses (CheXpert research agreement; PhysioNet
Credentialed Health Data License).
