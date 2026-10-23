from hsi_common import *
# R8: annotation-budget benchmark on spatially disjoint splits (Indian Pines, Pavia University, Salinas).
# Called by r8_<scene>.py. Indian Pines also gets the Local -> Indian Pines transfer arms (replaces R3).

SCENE_SPLITS = {'IndianPines': [0.01, 0.05, 0.10, 0.20], 'PaviaU': [0.005, 0.01, 0.02, 0.05],
                'Salinas': [0.005, 0.01, 0.02, 0.05]}
NETS = ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN']
TRANSFER_NETS = ['HybridSN', 'SpectralFormer', 'ViT']


def pretrain_on_local(log_name):
    """Source models for Local -> scene transfer: all soil samples, one sample per class held out for early stopping."""
    samples = local_samples()
    val = {c: sorted(s for s, (cc, _) in samples.items() if cc == c)[0] for c in range(3)}
    tr_files = [f for s, (c, fs) in samples.items() if val[c] != s for f in fs]
    va_files = [f for s, (c, fs) in samples.items() if val[c] == s for f in fs]
    Ltr, ytr = load_local_files(tr_files)
    Lva, yva = load_local_files(va_files)
    src = make_loaders([(Ltr, ytr, None), (Lva, yva, None), (Lva, yva, None)], 0)
    pre = {}
    for name in TRANSFER_NETS:
        set_seed(0)
        pre[name], info = train(MODELS[name](168, 3), *src[:2])
        log(log_name, dict(exp='R8', stage='pretrain_local', model=name, **info, **test(pre[name], src[1])))
    return pre


def run_disjoint(scene, with_transfer=False, nets=NETS, lightgbm=True, part=''):
    """`nets`/`lightgbm` select what runs, so cheap models can go to a CPU kernel; `part` names its log."""
    log_name = f'r8_{scene.lower()}{part}'
    cube, gt = load_scene(scene)
    bands, classes, tile = cube.shape[2], int(gt.max()), SCENES[scene][4]
    pre = pretrain_on_local(log_name) if with_transfer else {}
    for split in ([SCENE_SPLITS[scene][1]] if SMOKE else SCENE_SPLITS[scene]):
        for seed in SEEDS:
            tr, va, te = disjoint_split(gt, split, seed, tile)
            base = dict(exp='R8', dataset=scene, split=split, seed=seed, n_train=len(tr[0]), n_val=len(va[0]),
                        n_test=len(te[0]), train_classes=len(np.unique(gt[tr])), test_classes=len(np.unique(gt[te])))
            Xtr, Xte = cube[tr][:, None, None], cube[te][:, None, None]
            if lightgbm:
                log(log_name, {**base, 'init': 'none', **run_classic('LightGBM', Xtr, gt[tr] - 1, Xte, gt[te] - 1, seed)})
            if lightgbm and scene == 'IndianPines' and split == 0.05 and seed == 42:
                shap_bands(Xtr, gt[tr] - 1, Xte, gt[te] - 1, 'indianpines_disjoint_5pct', log_name)
            loaders = scene_loaders(cube, gt, [tr, va, te], seed)
            for name in nets:
                for init in ['scratch'] + (['transfer'] if name in pre else []):
                    set_seed(seed)
                    m = MODELS[name](bands, classes)
                    copied = transfer(pre[name], m) if init == 'transfer' else []
                    m, info = train(m, *loaders[:2], frozen=copied, freeze_epochs=FREEZE_EPOCHS if copied else 0)
                    log(log_name, {**base, 'model': name, 'init': init, 'n_copied': len(copied), **info, **test(m, loaders[2])})
                    free(m)
    finish(log_name, tile=tile, splits_scene=SCENE_SPLITS[scene])
