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

---

## 7. Round 3 (answers to the second set of five reviews)

Already done in the manuscript (no runs): AA and κ table (Appendix A1), dataset citations, Local budget curve in place of the transfer-gain figure, transfer table moved to the appendix, shorter title and abstract, 1715 nm edge band flagged.

### Data check of the Red images (done 2026-10-23, 16 Red images downloaded)
- **Same soil in `Red Soil/Moist/n` and `Red Soil/11-10-22/n`? Cannot be decided from the data.** Wetting changes the spectrum about 10x more than the difference between two dry samples, so every moist image is closest to the same dry image (6-2), not to its namesake. The preparations also look different (round wet patties vs. square dry piles). Decision: **merge them** (treat Moist/1 = dry/1, Moist/3 = dry/3) in R13/R14, so a possible leak cannot happen. Red then has 8 independent samples instead of 10. Ask the provider to confirm.
- **New, more serious finding: most patches are background, labelled as soil.** Each image shows six soil spots on a bright tray. Measured on 24 images (4 Black, 4 Yellow, all 16 Red) with a vignetting-corrected mask (band-mean image divided by each column's 90th percentile; soil where < 0.6; masks checked visually): **38–72% of the non-overlapping 15x15 patches contain <=10% soil** (Black 53–61%, Yellow 38–55%, Red dry 43–60%, Red moist 61–72%), and only 16–40% are >=90% soil. Every one of these patches carries the soil label, and the tray brightness and vignetting differ between sessions (background DN: Black 5900–7250, Yellow 6600–7460 with one image at 12600, Red dry 6000–6850, Red moist 7900–8100), so a model can learn the session, and thus the class, from the tray. All Local results (R2, R4, R11, R12) are affected. (A plain global threshold fails on the vignetted dry Red images; use the per-column correction.)
- No white/dark reference frames are in the dataset, and only the Mix folder has `.hdr` files: the data are raw digital numbers.

### New runs
| Run | What it answers | Design | Where | Est. |
|---|---|---|---|---|
| **R11b soil mask** (new, blocks everything Local) | Are the Local results driven by background? | Soil mask as above (band mean / per-column 90th percentile < 0.6); keep only patches with >=90% soil (about 90–250 per image); also train LightGBM on background-only patches as a leakage check (above-chance accuracy = the tray leaks the label); merge same-number Red samples; rerun R11 and the Local part of R12 on the same repeat design | GPU + CPU | GPU ~6 h, CPU 1 h |
| **R13 moisture test** (reviewer 3.2) | Do the models classify soil or moisture? | Red is the only class in both states. (a) Train on Black (moist), Yellow (dry) and Red dry → test on Red moist; (b) train with Red moist → test on Red dry. Report the share of Red test patches called Red / Black / Yellow. If moist Red is called Black, the models key on moisture. (c) Repeat R11 with the water bands removed (930–1010 nm, 1340–1500 nm) and (d) with each spectrum divided by its mean (removes brightness). LightGBM, SVM, HybridSN, grouped-band transformer; 5 repeats. | LightGBM/SVM on CPU; HybridSN + grouped-band on GPU | CPU 1 h, GPU ~1.5 h |
| **R14 leave-one-sample-out** (reviewer 3.3) | Accuracy per physical sample with all other samples for training | 24 folds; same models as R13; report per-sample accuracy and which samples fail | CPU + GPU | GPU ~2.5 h |
| **R15 longer training** (reviewer 1.4) | Did the 50-epoch cap favour LightGBM? | Indian Pines (the scene where LightGBM wins), HybridSN and spectral transformer, 100 epochs, same splits | GPU | ~4 h |
| **R17 maps** (reviewer 2.4) | Classification maps | Seed 42, 5% (IP) / 1% (Pavia U, Salinas): LightGBM + HybridSN over the whole scene; one figure in the supplement | GPU | ~0.5 h |
| **R18 more classical baselines** (reviewer 2.2) | Is LightGBM the right baseline? | Add RBF-SVM and random forest to R12 on the same splits; zero-pad Pavia U to 104 bands and rerun the grouped-band transformer | CPU | ~2 h |
| R16 wavelength-aligned transfer (reviewer 4.1), *optional* | Does transfer help when bands match physically? | Resample Local (168 bands) to the 82 Indian Pines band centres in 889–1715 nm; both directions; HybridSN + grouped-band | GPU | ~6 h |

Required GPU: **~14.5 h** (R11b, R13–R15, R17); with R16 **~20.5 h**. R11b goes first; R13 and R14 reuse its mask. dev123123456 used ~15 h this week, devgurucodes ~27 h, so run R13/R14/R17 first and R15/R16 after the weekly reset. Standard published disjoint splits (reviewer 2.1) are left out unless an official Pavia U training set can be obtained; the paper justifies the tile design instead.

### Code changes needed
- `hsi_common.py`: `local_moisture(path)` (Moist vs dated folder), optional band mask and per-spectrum normalisation in `load_local_files`; zero-pad bands to a multiple of 4 in `SpectralFormer`; `predict_scene` for maps.
- New scripts `r13_moisture.py`, `r14_loso.py`, `r15_long.py` (sets `EPOCHS=100`), `r17_maps.py`; extend `r12_lightgbm.py` with SVM and random forest; new sessions in `build.sh`.
- `analysis.py`: confusion of Red test patches for R13, per-sample table for R14, epochs-100 vs 50 table for R15.

### Manuscript after round 3
- Section on the moisture test in Results; rewrite the Local limitation and the SHAP paragraph from its outcome.
- If R15 changes the Indian Pines ranking, rewrite the Indian Pines paragraph, the abstract and the conclusion.
