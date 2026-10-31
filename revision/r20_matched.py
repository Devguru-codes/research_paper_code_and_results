from r8_disjoint import *
# R20: deep models under a matched label budget. R8 let deep models use up to 3,000 extra training-area pixels for
# early stopping, while LightGBM/RF train on the p labelled pixels alone. Here 20% of those p pixels (per class,
# rounded down) are held out for early stopping, so every model sees p labels in total. Same splits and seeds as R8/R12,
# so R12 LightGBM and R18 RF rows are the paired baselines. Entry: r20_<scene>*.py.


def matched(tr, gt, seed, frac=0.2):
    """Split the p training pixels into train/val per class; classes with < 5 pixels stay entirely in train."""
    y, rng, va = gt[tr], np.random.default_rng(seed), []
    for c in np.unique(y):
        i = rng.permutation(np.flatnonzero(y == c))
        va += list(i[:int(frac * len(i))])
    va = np.array(sorted(va), int)
    keep = np.setdiff1d(np.arange(len(y)), va)
    assert len(keep) + len(va) == len(y) and not np.intersect1d(keep, va).size
    return tuple(a[keep] for a in tr), tuple(a[va] for a in tr)


def run_matched(scene, nets, part=''):
    log_name = f'r20_{scene.lower()}{part}'
    cube, gt = load_scene(scene)
    for split in ([SCENE_SPLITS[scene][1]] if SMOKE else SCENE_SPLITS[scene]):
        for seed in SEEDS:
            tr0, _, te = disjoint_split(gt, split, seed, SCENES[scene][4])
            tr, va = matched(tr0, gt, seed)
            base = dict(exp='R20', protocol='matched_budget', dataset=scene, split=split, seed=seed,
                        n_labelled=len(tr0[0]), n_train=len(tr[0]), n_val=len(va[0]), n_test=len(te[0]))
            loaders = scene_loaders(cube, gt, [tr, va, te], seed)
            for name in nets:
                set_seed(seed)
                m, info = train(MODELS[name](cube.shape[2], int(gt.max())), *loaders[:2])
                log(log_name, {**base, 'model': name, 'init': 'scratch', **info, **test(m, loaders[2])})
                free(m)
    finish(log_name, protocol='matched_budget', val_fraction=0.2, splits_scene=SCENE_SPLITS[scene])
