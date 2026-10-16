# Revision Plan: Remote Sensing Letters Submission

Target: **Remote Sensing Letters** (Taylor & Francis, ScholarOne). This is a short letter format, so check the current word, figure and table limits on the Instructions for Authors page before drafting.
GPU budget: **~18 T4 hours** (about 60% of one week of Kaggle's 30 h/week quota).

---

## 0. One claim the letter will make
> Under strict annotation budgets, gradient boosting (LightGBM) beats deep models from scratch, but cross-domain pretraining moves the crossover point lower.

Everything else (Mamba, Tinto, HyBEAR, GCN, 1D-CNN, SVM, heads/depth ablations) is cut or moved to the supplement.

---

## 1. Shared protocol (apply to every rerun)

| Item | Setting |
|---|---|
| Seeds | 5 (`42, 43, 44, 45, 46`) |
| Splits | 1%, 5%, 10%, 20% |
| Validation | Indian Pines: fixed 10% drawn from the non-training pool. Local soil: a separate set of validation **files** (see split row). **Never early-stop on the test set.** |
| Preprocessing | Per-band standardisation fitted on the training split only (same for all models) |
| Optimiser | AdamW, lr 5e-4, wd 1e-4, cosine schedule, batch 64 |
| Epochs | 50 max, early stopping patience 10 on validation loss |
| LightGBM | 100 trees, defaults, centre pixel |
| Metrics | OA, AA, Kappa, macro-F1 (all saved per seed to CSV) |
| Stats | Paired t-test + 95% CI + Cohen's d_z, Benjamini–Hochberg FDR. With n=5, Wilcoxon can't go below p = 0.0625, so don't use it. |
| Local soil split | **Image-level** 60/20/20 by file (train / validation / test), so validation never shares images with test. All three file lists saved to `splits/local_files.json` |
| Logging | Save `manifest.json` (versions, seeds, splits) + per-seed CSV in every run |

Put all of this in one shared module `hsi_common.py` (data loading, models, `train_and_eval`, stats). Every notebook imports it, so protocols can't drift.

---

## 2. Re-runs

| # | Experiment | What changes vs. old run | Grid | Est. T4 h |
|---|---|---|---|---|
| R1 | **Few-shot curve on Indian Pines** (Table II, Fig. crossover) | Proper validation split; one preprocessing; SpectralFormer and ConvStem ViT included in the same run | 4 splits × 5 seeds × {LightGBM, ViT, SpectralFormer, ConvStem ViT, HybridSN} | ~6.0 |
| R2 | **Local soil benchmark** (Table I) | Image-level split instead of dry/moist patch split; only LightGBM, SVM (scaled), HybridSN, ViT, SpectralFormer | 5 seeds × 5 models (full Local training set, ~19k patches) | ~3.5 |
| R3 | **Transfer Local → Indian Pines** (negative-transfer claim) | Never completed with logged results; rerun with the same code as the IP → Local run | 4 splits × 5 seeds × 3 models × {scratch, transfer} | ~3.5 |
| R4 | **Transfer IP → Local** | Rerun with the separate validation split (old checkpoints were picked on test loss, so they can't be reused) | same grid | ~3.5 |
| R5 | **Linear probe (ViT only)** | Fix file name `vanillavit_local.pt` → `vit_local.pt`. HybridSN probe: average-pool the `conv3` output (last layer that transfers between 168 and 200 bands) and train one new `nn.Linear` on it, for both scratch and transfer. | 3 splits × 5 seeds × 3 models × 2 | ~1.0 |
| R6 | **Timing (Table V)** | Real measurement on T4, batch size 1, 1000 test patches, median of 3 runs. LightGBM timed on CPU. | — | ~0.3 |
| | **Total** | | | **~18 h** |

Not rerun (results reused as-is): depth and heads ablations (supplement only; they already use a proper validation split). ConvStem now runs inside R1.

**Run order (Kaggle sessions ≤12 h):**
1. Session A: R1 + R6 (~6.5 h)
2. Session B: R3 + R5 (~4.5 h)
3. Session C: R4 + R2 (~7 h)

Use incremental CSV saving per seed so a timeout loses nothing.

---

## 3. CPU-only analyses (no GPU)

- **SHAP on LightGBM** (`shap.TreeExplainer`) for Local and Indian Pines, using the R1/R2 models.
- **Band → wavelength mapping:** Pika NIR-320, 168 bands, 900–1700 nm, so band *i* ≈ 900 + i·(800/167) nm. Band 96 ≈ 1360 nm, near the 1400 nm water/OH feature. Remove the "clay at 2200 nm" claim; it's outside the sensor range.
- **Crossover:** compute the piecewise-linear crossover for scratch and transfer **on the same target dataset**, with bootstrap CIs over seeds.
- **Pixel-wise map:** regenerate from the R1 HybridSN model (state the model in the caption).

---

## 4. Manuscript changes

### Methods
- [ ] Replace the hyperparameter table with Section 1's protocol (exact values).
- [ ] Software versions from `manifest.json` (PyTorch 2.10, CUDA 12.8, LightGBM 4.6). Hardware: Kaggle T4 only, so remove "RTX 4090" and "32 GB".
- [ ] Describe the image-level split for Local; remove the "15-pixel buffer" claim. Say there are 6 specimens and 47 images, and that the classes are Black/Red/Yellow soil.
- [ ] Local data is **lab-imaged** (Resonon line-scan). Remove "UAV/drone" wording. AVIRIS is **airborne**, not satellite.
- [ ] Statistics: one consistent statement (paired t-test, 95% CI, d_z, BH-FDR). Remove the Shapiro–Wilk, Friedman and "no correction" text unless actually run.
- [ ] Transfer: linear positional-embedding interpolation over the **band** dimension (not patch size, not bicubic). State which layers are frozen.
- [ ] Rename models honestly: "spectral Transformer (centre pixel)" instead of "ViT"; drop "GCN" or label it an MLP baseline.

### Results
- [ ] Table I ← R2; Table II / crossover figure ← R1; transfer table ← R3 + R4 (both directions, both reported, LightGBM row added).
- [ ] Rewrite the negative-transfer text from R3's actual numbers. Expect HybridSN to gain and the transformers to lose; describe it as **architecture-dependent asymmetry**, not universal collapse.
- [ ] Linear-probe text ← R5.
- [ ] Remove every number that has no artefact (53.92/35.90, appendix AA/F1, Kappa ± values, old Table V).
- [ ] Discuss the ConvStem ViT result from R1 openly (old ablation run: 84.65% at 5%): a local inductive bias closes the gap.

### Format for the letter
- [ ] Main text: 1 workflow figure, 1 crossover figure, 1 transfer table, 1 combined Table I/II. Everything else goes to the supplement.
- [ ] Drop the "Proposed ConvStem architecture" figure (it isn't proposed).
- [ ] Replace the AI-generated Figures 1/7 with simple vector diagrams (check the journal's generative-AI image policy).
- [ ] Fix citations: SpectralFormer = b3 (not b8/b11), HybridSN = b16 (not b15), b7 is Mamba, not CNN-Transformer. Verify b11, b12 (arXiv 2405.01234 looks like a placeholder) and b20–b22; drop anything unverifiable.
- [ ] Convert from `sn-jnl` (Springer) to the Taylor & Francis template; switch citation style to the journal's.

---

## 5. Repository (Data & Code Availability)
- [ ] Add `hsi_common.py`, notebooks `R1`–`R6`, `notebooks/exp-5-ablation.ipynb` (from Kaggle `exp_4` v6) and `notebooks/ip-shap-and-maps.ipynb`.
- [ ] Add `requirements.txt` (pinned to the manifest versions) and `splits/` (seed lists, Local file split).
- [ ] Remove stale outputs (`results_#`, old `results_exp_2`, the `drive-download…` previews) or move them to `archive/`.
- [ ] README: one table mapping each paper table/figure → notebook → output file.
- [ ] State how the Local dataset can be accessed (on request / Kaggle link).

---

## 6. Timeline

| Day | Work |
|---|---|
| 1 | Write `hsi_common.py`; dry-run each notebook on 1 seed / 1 split |
| 2–4 | Kaggle sessions A, B, C (~18 GPU-h) |
| 5 | CPU analyses: SHAP, crossover with CIs, maps, timing table |
| 6–8 | Rewrite manuscript in the T&F template; trim to the letter length |
| 9 | Repo clean-up, README mapping, final cross-check of every number against its CSV |
| 10 | Co-author read-through → submit via ScholarOne |

**Submission gate:** every number in the manuscript must trace to a CSV in the repo produced by the notebooks listed in the README.
