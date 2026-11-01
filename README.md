# Evaluation design can decide the winner: spatial and acquisition leakage in few-label hyperspectral classification

Code, split definitions, per-seed results and manuscript for the letter submitted to *Remote Sensing Letters*.

The letter compares LightGBM, a random forest, an RBF-SVM, three transformers and the 3-D CNN HybridSN
under leaky and controlled evaluation protocols:

- **Spatial leakage.** On Indian Pines, a random pixel split puts HybridSN 17-20 percentage points ahead
  of LightGBM; on spatially disjoint tiles no deep model is significantly better. On Pavia University and
  Salinas the random split inflates accuracy by up to 11 points but keeps the ranking, and HybridSN stays
  ahead under disjoint tiles, also when early stopping uses only the labelled budget.
- **Acquisition leakage.** In laboratory soil images, 38-72% of patches show only the sample tray, and a
  classifier trained on tray patches alone reaches 87% (chance 33%). After masking the tray and normalising
  brightness, every model averages about 94% or more with one labelled sample per class.

## Repository structure

```
.
├── rsl_manuscript.tex / .pdf     # the letter (Taylor & Francis Interact template, interact.cls)
├── rsl_supplementary.tex / .pdf  # supplementary tables S1-S2
├── references.bib
├── results/figures/              # figures used by the letter (copies of revision/results/final/figures)
├── revision/                     # all code and results for the letter
│   ├── hsi_common.py             # shared protocol: data loading, splits, models, training, logging
│   ├── r*_*.py                   # one script per experiment (R1-R20, see REVISION_PLAN.md)
│   ├── build.sh                  # packs a session's scripts into one Kaggle kernel and pushes it
│   ├── analysis.py               # tables, statistics and figures from the logged results
│   ├── stats_utils.py            # paired t-test, Benjamini-Hochberg FDR, table writer
│   ├── wavelengths/              # band-centre wavelengths (soil camera, Indian Pines)
│   └── results/session-*/        # per-seed CSV/JSONL logs and run manifests from each Kaggle session
├── notebooks/, results_from_notebook/  # earlier version of the study (superseded, kept for the record)
└── REVISION_PLAN.md              # experiment plan and data-check notes
```

## Reproducing

All experiments ran on Kaggle (NVIDIA Tesla T4 for the deep models). The airborne scenes are downloaded by
`hsi_common.load_scene`; the soil images are the Kaggle dataset
[local-soil-hyperspectral-dataset](https://www.kaggle.com/datasets/dev123123456/local-soil-hyperspectral-dataset).

```bash
cd revision
bash build.sh smoke f    # 1 seed, 1 budget, 2 epochs
bash build.sh full f     # full run; GPU=false for a CPU kernel, KAGGLE_OWNER=<user> for another account
```

Download a finished kernel's output into `revision/results/session-<id>/`, then rebuild every table and figure:

```bash
python revision/analysis.py
```

Outputs go to `revision/results/final/{tables,figures}/`.
