# Evaluation design can decide the winner: spatial and acquisition leakage in few-label hyperspectral classification

Code, split definitions and per-seed results for the letter of the same title (submitted to *Remote Sensing Letters*).

The letter compares LightGBM, a random forest, an RBF-SVM, three transformers and the 3-D CNN HybridSN
under leaky and controlled evaluation protocols:

- **Spatial leakage.** On Indian Pines, a random pixel split puts HybridSN 17-20 percentage points ahead
  of LightGBM; on spatially disjoint tiles no deep model is significantly better. On Pavia University and
  Salinas the random split inflates accuracy by up to 11 points but keeps the ranking, and HybridSN stays
  ahead under disjoint tiles, also when early stopping uses only the labelled budget.
- **Acquisition leakage.** In laboratory soil images, 38-72% of patches show only the sample tray, and a
  classifier trained on tray patches alone reaches 87% (chance 33%). After masking the tray and normalising
  brightness, every model averages about 94% or more with one labelled sample per class.

## Layout

```
code/
  hsi_common.py     shared protocol: data loading, splits, models, training, metrics, logging
  stats_utils.py    paired t-test, Benjamini-Hochberg FDR, table writer
  r*_*.py           one script per experiment (table below)
  build.sh          packs a session's scripts into one Kaggle kernel and pushes it
  wait.sh           waits for Kaggle kernels to finish
  analysis.py       every table, statistic and figure in the letter
  wavelengths/      band-centre wavelengths (soil camera, Indian Pines)
results/
  runs/session-*/   per-seed CSV/JSONL logs and run manifests, one folder per Kaggle session
  tables/           tables and statistics written by analysis.py
  figures/          figures written by analysis.py
```

## Experiments

| Exp. | What it measures | Sessions |
|---|---|---|
| R1 | Indian Pines, random pixel split, deep models | a |
| R8 | Disjoint-tile benchmark on Indian Pines, Pavia University, Salinas | e, f, fc, g, gc, m |
| R12 | LightGBM with stable settings on the R8 splits | l |
| R18 | Random forest and SVM on the disjoint splits; LightGBM on the random split | k |
| R19 | Random pixel split on Pavia University and Salinas | w, x, y, z |
| R15 | Every deep model rerun with up to 100 epochs on the disjoint splits | n, o, p, s, t, u, v |
| R20 | Deep models with early stopping on 20% of the labelled budget (no extra labels) | r20a-r20d |
| R17 | Classification maps, seed 42 | n |
| R11 | Soil, raw patches | h |
| R11b | Soil, soil-only patches; tray-only leakage check | i |
| R13 | Soil, moist/dry cross-state test; water-band-removed and brightness-normalised variants | j, q |

## Reproducing

All experiments ran on Kaggle (NVIDIA Tesla T4 for the deep models). The airborne scenes are downloaded by
`hsi_common.load_scene`; the soil images are the Kaggle dataset
[local-soil-hyperspectral-dataset](https://www.kaggle.com/datasets/dev123123456/local-soil-hyperspectral-dataset).

```bash
bash code/build.sh smoke f    # 1 seed, 1 budget, 2 epochs
bash code/build.sh full f     # GPU=false for a CPU kernel, KAGGLE_OWNER=<user> for another account
```

Download a finished kernel's output into `results/runs/session-<id>/`, then rebuild every table and figure:

```bash
python code/analysis.py
```
