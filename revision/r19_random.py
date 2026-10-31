from r8_disjoint import *
# R19: random pixel split (the leaky protocol of R1) on Pavia U and Salinas, with the budgets, seeds and models of the
# disjoint runs, to test whether the spatial-leakage effect seen on Indian Pines generalises. Entry: r19_<scene>_*.py.


def random_split(gt, frac, seed, val_cap=3000):
    """Stratified random pixels: train = frac of labelled pixels; val = 10% of the rest (at most val_cap); test = rest."""
    rc, y = np.argwhere(gt > 0), gt[gt > 0]  # same row-major order
    tr, rest = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=min(val_cap, len(rest) // 10), stratify=y[rest], random_state=seed)
    return [tuple(rc[i].T) for i in (tr, va, te)]


def run_random(scene, nets, lightgbm, part):
    log_name = f'r19_{scene.lower()}{part}'
    cube, gt = load_scene(scene)
    for split in ([SCENE_SPLITS[scene][1]] if SMOKE else SCENE_SPLITS[scene]):
        for seed in SEEDS:
            tr, va, te = random_split(gt, split, seed)
            base = dict(exp='R19', protocol='random', dataset=scene, split=split, seed=seed, n_train=len(tr[0]), n_test=len(te[0]))
            if lightgbm:
                log(log_name, {**base, 'init': 'none', **run_classic('LightGBM', cube[tr][:, None, None], gt[tr] - 1,
                                                                      cube[te][:, None, None], gt[te] - 1, seed)})
            loaders = scene_loaders(cube, gt, [tr, va, te], seed)
            for name in nets:
                set_seed(seed)
                m, info = train(MODELS[name](cube.shape[2], int(gt.max())), *loaders[:2])
                log(log_name, {**base, 'model': name, 'init': 'scratch', **info, **test(m, loaders[2])})
                free(m)
    finish(log_name, protocol='random', splits_scene=SCENE_SPLITS[scene])
