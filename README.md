# Probabilistic Modeling for Healthcare AI Evaluation

Daphne Barretto · Izhan Hamza · Shannon Komguem

Stanford University

---

## Overview

This project applies probabilistic measurement models from psychometrics to evaluate AI diagnostic systems on clinical imaging tasks. Rather than summarizing model performance with a single aggregate accuracy score, we treat AI models as _test takers_ and clinical imaging cases as _test items_, constructing a binary response matrix amenable to Item Response Theory (IRT) and factor analysis.

We fit and compare Rasch/1PL, 2PL, and Factor models to reveal item-level structure in AI performance — surfacing failure modes that AUC and F1 systematically obscure, particularly for rare but clinically critical conditions.

---

## Research Question

> Which probabilistic model (Rasch/1PL, 2PL, or Factor) best characterizes the response patterns of diagnostic AI systems on clinical imaging tasks, and what does this reveal about system capabilities that aggregate accuracy obscures?

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Raw dataset files (not committed; see Data Access)
│   ├── processed/            # Binarized response matrices (J × I)
│   ├── labels/               # Ground-truth condition labels
│   └── inference/            # Per-model inference outputs on CheXpert
│                             # train1, one canonical JSON per model
│                             # (e.g. gpt-5.4.json, pixtral-12b.json).
│                             # Row counts vary across models; the IRT
│                             # loader treats missing cells as MAR.
│
├── models/
│   ├── inference/            # Scripts to run AI model inference on CheXpert
│   └── checkpoints/          # Model weights (not committed; see Data Access)
│
├── irt/
│   ├── fit_rasch.py          # Rasch / 1PL model fitting
│   ├── fit_2pl.py            # 2PL model fitting
│   ├── fit_3pl.py            # 3PL model fitting
│   └── fit_factor.py         # Latent factor model fitting (1- and 2-factor)
│
├── analysis/
│   ├── data_loader.py        # Load data/inference/*.json into long-form IRT obs
│   ├── model_comparison.py   # AIC, BIC, LRT, M2, RMSEA
│   ├── item_fit.py           # Infit/outfit (Rasch), S-chi2 (2PL)
│   ├── ranking_stability.py  # Spearman's ρ: θ-based vs. accuracy-based rankings
│   └── validity.py           # Clinical validity checks (uncertainty labels, rare conditions)
│
├── notebooks/
│   ├── 01_response_matrix.ipynb
│   ├── 02_model_fitting.ipynb
│   ├── 03_icc_visualization.ipynb
│   └── 04_governance_analysis.ipynb
│
├── outputs/
│   ├── figures/              # ICCs, TIF curves, θ vs. accuracy scatter plots
│   ├── tables/               # Fit statistics, parameter estimates
│   └── irt/                  # IRT artifacts (e.g. dif_by_sex.csv, headline_findings.json)
│
├── requirements.txt
└── README.md
```

> **Cell-level missingness in gpt-4o.** ``data/inference/gpt-4o.json`` carries
> the same train1 images as the other models but ~93% of those rows are
> `missing_reason=model_refusal` (Azure OpenAI content filter) — only
> ~6.94% of pathology cells are populated. IRT loaders skip missing cells
> (treating them as MAR).

---

## Methods Summary

### Response Matrix Construction

We evaluate J ≥ 5 AI diagnostic models on clinical imaging items drawn from the CheXpert validation set (14 thoracic pathology conditions). Each entry $X_{ij} \in \{0, 1\}$ indicates whether model j correctly identified the pathology status of item i.

**AI model set (planned):** CheXNet, DenseNet-121 variants, TorchXRayVision ensembles, and frontier multimodal LLMs spanning a range of capability levels.

### Psychometric Models

| Model       | Parameters               | Formula                                                     |
| ----------- | ------------------------ | ----------------------------------------------------------- |
| Rasch / 1PL | $\theta_i, \beta_j$      | $P(Y_{ij} = 1) = \sigma(\theta_i - \beta_j)$                |
| 2PL         | $\theta_i, \beta_j, a_j$ | $P(Y_{ij} = 1) = \sigma\bigl(a_j(\theta_i - \beta_j)\bigr)$ |
| Factor      | $U_i^\top,V_j,Z_j$       | $P(Y_{ij} = 1) = \sigma(U_i^\top V_j + Z_j)$                |

### Model Comparison

- **Nested models:** Likelihood ratio tests (LRT)
- **All models:** AIC, BIC
- **Absolute fit:** M2 statistic, RMSEA
- **Item-level fit:** Infit/outfit mean-square (Rasch); $S\text{-}\chi^2$ (2PL)

---

## Data Access

This project uses publicly available de-identified medical imaging datasets obtained under appropriate research agreements.

| Dataset             | Size        | Conditions | Access                                                                         |
| ------------------- | ----------- | ---------- | ------------------------------------------------------------------------------ |
| CheXpert (Stanford) | 224K images | 14         | [Research agreement](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| MIMIC-CXR (MIT)     | 377K images | 14         | [PhysioNet credentialed](https://physionet.org/content/mimic-cxr/)             |

> **Note:** Raw data files are not committed to this repository. After obtaining access, place data under `data/raw/` following the structure described in `data/README.md`.

---

## Setup

```bash
git clone https://github.com/<org>/cs321m-healthcare-irt.git
cd cs321m-healthcare-irt
pip install -r requirements.txt
```

Python 3.10+ recommended. Key dependencies: `pyirt`, `factor_analyzer`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch` (for model inference).

---

## Project Timeline

| Week | Dates     | Milestones                                                                     |
| ---- | --------- | ------------------------------------------------------------------------------ |
| 1    | May 5–11  | Finalize dataset & model set; obtain data access; submit pre-analysis plan     |
| 2    | May 12–18 | Run inference; binarize predictions; construct J × I response matrix           |
| 3    | May 19–25 | Fit all models; compute fit statistics; generate visualizations; draft results |
| 4    | May 26–27 | Final manuscript polish; code cleanup; submission                              |

---

## Key Claims & Hypotheses

1. **Aggregate accuracy is insufficient** for governance of healthcare AI — AUC and F1 treat all errors as equally consequential and all items as interchangeable.
2. **IRT-derived parameters** (item difficulty $\beta_j$, discrimination $a_j$, latent ability $\theta_i$) provide actionable item-level diagnostics that scalar metrics cannot.
3. **High-risk blind spots** — rare, high-discrimination conditions (e.g., tension pneumothorax) — are precisely where AI capability varies most yet aggregate scores mask the variation.
4. **Factor models** may reveal that diagnostic AI ability is multidimensional (e.g., sensitivity to opacities vs. structural abnormalities), challenging the unidimensionality assumed by standard IRT.

---

## References

- Martínez-Plumed et al. (2019). Item response theory in AI. _Artificial Intelligence_, 271:18–42.
- Rajpurkar et al. (2017). CheXNet: Radiologist-level pneumonia detection. _arXiv:1711.05225_.
- Irvin et al. (2019). CheXpert: A large chest radiograph dataset. _AAAI_, 33:590–597.
- Johnson et al. (2019). MIMIC-CXR. _Scientific Data_, 6(1):317.
- Schilling-Wilhelmi et al. (2025). Lifting the benchmark iceberg with IRT. _ICLR 2025 Workshop_.
- Unell et al. (2025). Beyond mean scores: Factor models for AI evaluation. _OpenReview_.
- Wu et al. (2021). How medical AI devices are evaluated. _Nature Medicine_, 27:582–584.
- Yu et al. (2025). Beyond accuracy: Allocation-aware evaluation of AI in healthcare. _arXiv:2601.06161_.

---

## Authors

| Name            | Email                 |
| --------------- | --------------------- |
| Daphne Barretto | daphnegb@stanford.edu |
| Izhan Hamza     | izhamza@stanford.edu  |
| Shannon Komguem | skomguem@stanford.edu |

---

## License

This repository is for academic use. Data usage is governed by the respective dataset licenses (CheXpert research agreement; PhysioNet Credentialed Health Data License).
