from hsi_common import *
# R11b: R11 on soil pixels only. 38-72% of the Local patches show bare tray but carry the soil label, and the
# tray differs between sessions, so R11 may partly classify the tray. Changes from R11: only patches with
# >= 90% soil (soil_mask); Red Moist/n and Red 11-10-22/n count as one sample (local_soil_id); plus a leakage
# check that trains LightGBM/SVM on background-only patches (<= 10% soil) of the same samples.

CAP, BUDGETS = 300, ([2] if SMOKE else [1, 2, 3, 4])
NETS, TRANSFER_NETS = ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN'], ['HybridSN', 'SpectralFormer', 'ViT']

samples = local_samples(key=local_soil_id)
data = {s: load_local_files(fs, soil=(0.9, 1)) for s, (c, fs) in samples.items()}
bg = {s: load_local_files(fs, soil=(0, 0.1)) for s, (c, fs) in samples.items()}
for s in samples:
    print(s.split('hyperspectral-dataset/')[-1], 'soil patches', len(data[s][1]), 'background patches', len(bg[s][1]), flush=True)
    assert len(data[s][1]) and len(bg[s][1]), f'no patches for {s}'
cap = lambda src: {s: (X[i], y[i]) for s, (X, y) in src.items() for i in [np.random.default_rng(0).permutation(len(y))[:CAP]]}
capped, bg_capped = cap(data), cap(bg)
print('samples per class', np.bincount([c for c, _ in samples.values()]).tolist(), flush=True)

# Source models for Indian Pines -> Local, as in R11.
cube, gt = load_scene('IndianPines')
rc = np.argwhere(gt > 0)
ptr, pva = train_test_split(np.arange(len(rc)), test_size=0.1, stratify=gt[gt > 0], random_state=0)
src = scene_loaders(cube, gt, [tuple(rc[ptr].T), tuple(rc[pva].T), tuple(rc[pva].T)], 0)
pre = {}
for name in TRANSFER_NETS:
    set_seed(0)
    pre[name], info = train(MODELS[name](200, 16), *src[:2])
    log('r11b', dict(exp='R11b', stage='pretrain_ip', model=name, **info, **test(pre[name], src[1])))
del src, cube


def stack(ids, source):
    return np.concatenate([source[s][0] for s in ids]), np.concatenate([source[s][1] for s in ids])


for seed in SEEDS:
    rng = np.random.default_rng(seed)
    order = {c: list(rng.permutation(sorted(s for s, (cc, _) in samples.items() if cc == c))) for c in range(3)}
    test_ids = [s for c in range(3) for s in order[c][:2]]
    val_ids = [s for c in range(3) for s in order[c][2:3]]
    Xte, yte = stack(test_ids, data)
    Xva, yva = stack(val_ids, capped)
    for k in BUDGETS:
        tr_ids = [s for c in range(3) for s in order[c][3:3 + k]]
        Xtr, ytr = stack(tr_ids, capped)
        base = dict(exp='R11b', dataset='Local', budget_samples_per_class=k, seed=seed, n_train=len(ytr),
                    n_test=len(yte), test_samples=';'.join(s.split('hyperspectral-dataset/')[-1] for s in test_ids))
        for kind in ['LightGBM', 'SVM']:
            log('r11b', {**base, 'init': 'none', **run_classic(kind, Xtr, ytr, Xte, yte, seed)})
        if k == BUDGETS[-1]:
            shap_bands(Xtr, ytr, Xte, yte, f'local_masked_seed{seed}', 'r11b')
            Btr, btr = stack(tr_ids, bg_capped)
            Bte, bte = stack(test_ids, bg)
            for kind in ['LightGBM', 'SVM']:  # leakage check: tray only, same samples
                log('r11b', {**base, 'n_train': len(btr), 'n_test': len(bte), 'init': 'none',
                             **run_classic(kind, Btr, btr, Bte, bte, seed), 'model': f'{kind}_background'})
        loaders = make_loaders([(Xtr, ytr, None), (Xva, yva, None), (Xte, yte, None)], seed)
        for name in NETS:
            for init in ['scratch'] + (['transfer'] if name in pre else []):
                set_seed(seed)
                m = MODELS[name](168, 3)
                copied = transfer(pre[name], m) if init == 'transfer' else []
                m, info = train(m, *loaders[:2], frozen=copied, freeze_epochs=FREEZE_EPOCHS if copied else 0)
                log('r11b', {**base, 'model': name, 'init': init, 'n_copied': len(copied), **info, **test(m, loaders[2])})
                free(m)
finish('r11b', cap_patches_per_sample=CAP, budgets=BUDGETS, min_soil_fraction=0.9, lgbm=LGBM_STABLE)
