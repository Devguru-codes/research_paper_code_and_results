from r8_disjoint import *
# R18 (CPU): classical baselines on the exact splits of the paper.
# (A) Indian Pines random pixel split (R1 splits) with LightGBM at LGBM_STABLE, replacing R1's default LightGBM rows.
# (B) RBF-SVM and random forest on the spatially disjoint splits of all three scenes (R8/R12 splits).

cube, gt = load_scene('IndianPines')
rows, cols = np.nonzero(gt)  # same pixel order as load_indian_pines, so ip_split gives R1's splits
X, y = cube[rows, cols][:, None, None], (gt[rows, cols] - 1).astype(np.int64)
for split in SPLITS:
    for seed in SEEDS:
        tr, va, te = ip_split(y, split, seed)
        log('r18', dict(exp='R18', part='random', dataset='IndianPines', split=split, seed=seed, n_train=len(tr), n_test=len(te),
                        init='none', **run_classic('LightGBM', X[tr], y[tr], X[te], y[te], seed)))

for scene in SCENES:
    cube, gt = load_scene(scene)
    for split in ([SCENE_SPLITS[scene][1]] if SMOKE else SCENE_SPLITS[scene]):
        for seed in SEEDS:
            tr, va, te = disjoint_split(gt, split, seed, SCENES[scene][4])
            Xtr, Xte = cube[tr][:, None, None], cube[te][:, None, None]
            for kind in ['SVM', 'RF']:
                log('r18', dict(exp='R18', part='disjoint', dataset=scene, split=split, seed=seed, n_train=len(tr[0]), n_test=len(te[0]),
                                init='none', **run_classic(kind, Xtr, gt[tr] - 1, Xte, gt[te] - 1, seed)))
finish('r18', lgbm=LGBM_STABLE, rf_trees=500)
