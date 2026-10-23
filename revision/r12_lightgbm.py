from r8_disjoint import *
# R12 (CPU): every LightGBM baseline and SHAP value of R8/R11, recomputed with LGBM_STABLE on identical splits.
# The analysis takes LightGBM rows from here and ignores those logged by the R8/R11 GPU runs.

for scene in SCENES:
    cube, gt = load_scene(scene)
    for split in SCENE_SPLITS[scene]:
        for seed in SEEDS:
            tr, va, te = disjoint_split(gt, split, seed, SCENES[scene][4])
            Xtr, Xte = cube[tr][:, None, None], cube[te][:, None, None]
            log('r12', dict(exp='R12', dataset=scene, split=split, seed=seed, n_train=len(tr[0]), n_test=len(te[0]),
                            init='none', **run_classic('LightGBM', Xtr, gt[tr] - 1, Xte, gt[te] - 1, seed)))
            if scene == 'IndianPines' and split == 0.05 and seed == 42:
                shap_bands(Xtr, gt[tr] - 1, Xte, gt[te] - 1, 'indianpines_disjoint_5pct', 'r12')

# Local soil: same sample-level splits as R11 (per class: 2 test, 1 validation, first k pool samples), same patch cap.
CAP = 300
samples = local_samples()
data = {s: load_local_files(fs) for s, (c, fs) in samples.items()}
capped = {s: (X[i], y[i]) for s, (X, y) in data.items() for i in [np.random.default_rng(0).permutation(len(y))[:CAP]]}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    order = {c: list(rng.permutation(sorted(s for s, (cc, _) in samples.items() if cc == c))) for c in range(3)}
    te_ids = [s for c in range(3) for s in order[c][:2]]
    Xte, yte = np.concatenate([data[s][0] for s in te_ids]), np.concatenate([data[s][1] for s in te_ids])
    for k in [1, 2, 3, 4]:
        tr_ids = [s for c in range(3) for s in order[c][3:3 + k]]
        Xtr, ytr = np.concatenate([capped[s][0] for s in tr_ids]), np.concatenate([capped[s][1] for s in tr_ids])
        log('r12', dict(exp='R12', dataset='Local', budget_samples_per_class=k, seed=seed, n_train=len(ytr), n_test=len(yte),
                        init='none', **run_classic('LightGBM', Xtr, ytr, Xte, yte, seed)))
        if k == 4:
            shap_bands(Xtr, ytr, Xte, yte, f'local_samples_seed{seed}', 'r12')
finish('r12', lgbm=LGBM_STABLE)
