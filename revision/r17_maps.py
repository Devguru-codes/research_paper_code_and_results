from r8_disjoint import *
# R17: classification maps (seed 42, disjoint split): LightGBM and HybridSN predictions for every labelled pixel,
# saved with the training and test masks; figures are drawn by analysis.py.
MAP_SPLIT = {'IndianPines': 0.05, 'PaviaU': 0.01, 'Salinas': 0.01}
for scene, split in MAP_SPLIT.items():
    cube, gt = load_scene(scene)
    tr, va, te = disjoint_split(gt, split, 42, SCENES[scene][4])
    lab = np.nonzero(gt)
    pos = np.full(gt.shape, -1); pos[lab] = np.arange(len(lab[0]))
    pred = {'LightGBM': classic('LightGBM', 42).fit(cube[tr], gt[tr] - 1).predict(cube[lab])}
    loaders = scene_loaders(cube, gt, [tr, va, lab], 42)
    set_seed(42)
    m, info = train(MODELS['HybridSN'](cube.shape[2], int(gt.max())), *loaders[:2])
    pred['HybridSN'] = evaluate(m, loaders[2])[2]
    free(m)
    np.savez_compressed(f'{OUT}/map_{scene}.npz', gt=gt, rows=lab[0], cols=lab[1], train_r=tr[0], train_c=tr[1],
                        test_r=te[0], test_c=te[1], **pred)
    for k, p in pred.items():
        log('r17', dict(exp='R17', dataset=scene, split=split, seed=42, model=k, **metrics(gt[te] - 1, p[pos[te]])))
finish('r17', map_split=MAP_SPLIT)
