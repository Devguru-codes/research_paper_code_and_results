Dr Swati Hira
Department of Computer Science and Engineering
Indian Institute of Information Technology, Nagpur, Maharashtra, India
shira@iiitn.ac.in

[Date]

The Editor
*Remote Sensing Letters*

Dear Editor,

We submit the letter **"Evaluation design decides the winner: spatial and acquisition leakage in few-label hyperspectral classification"** for consideration in *Remote Sensing Letters*.

Papers that compare deep models with simpler learners for hyperspectral classification with few labels often reach opposite conclusions. Our letter shows that much of this disagreement comes from the evaluation design rather than from the models. We use the same five learners, seeds and statistics throughout: LightGBM, three transformers and the 3-D CNN HybridSN, with a random forest and an SVM as further baselines. Each dataset is evaluated under a leaky and a controlled protocol.

- **Spatial leakage.** On Indian Pines, a random pixel split puts HybridSN 17–20 percentage points ahead of LightGBM. On spatially disjoint tiles, no deep model is significantly better, and the ranking reverses. On Pavia University and Salinas, the random split inflates accuracy by up to 11 points but leaves the ranking unchanged, and HybridSN stays ahead of both LightGBM and the random forest under disjoint tiles.
- **Acquisition leakage.** In laboratory soil images, 38–72% of the patches show only the sample tray, and a classifier trained on tray patches alone reaches 87% accuracy (chance is 33%). After the tray is masked and brightness normalised, every model reaches at least 94% with one labelled sample per class. A moist/dry cross-state test shows that brightness, not water absorption, links class to the imaging session.
- **Robustness.** All tests use paired seeds, 95% confidence intervals and Benjamini–Hochberg correction. Rerunning every deep model with twice the training epochs leaves the conclusions unchanged.

The letter ends with a short checklist for few-label comparisons in hyperspectral imaging. We think it suits the readers of *Remote Sensing Letters* because it is brief, quantitative and directly usable by anyone who benchmarks classifiers on hyperspectral data. The code, split definitions, soil masks, per-seed results and run manifests are publicly available, and the soil images are on Kaggle.

We confirm that:

- the manuscript is original, has not been published, and is not under consideration elsewhere;
- all authors have approved the submitted version;
- the authors have no competing interests;
- the use of generative AI is declared in the manuscript.

Thank you for considering our submission.

Yours sincerely,

Swati Hira, on behalf of all authors
(Devguru Tiwari, Nagamani Molakatala, R. Kokila, Gururaja H S and J. Rajini)
