from hsi_common import *
# R11: Local soil, annotation budget counted in labelled *samples* (one label labels every patch of a sample).
# Five repeated sample-level splits: per class 2 test samples, 1 validation sample, the rest form the
# training pool; budget k = the first k pool samples per class (nested). Replaces R2 and R4.
# Patches per training/validation sample are capped (they are near-duplicates); test uses every patch.

CAP, BUDGETS = 300, ([2] if SMOKE else [1, 2, 3, 4])
NETS, TRANSFER_NETS = ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN'], ['HybridSN', 'SpectralFormer', 'ViT']

samples = local_samples()
data = {s: load_local_files(fs) for s, (c, fs) in samples.items()}
capped = {s: (X[i], y[i]) for s, (X, y) in data.items()
          for i in [np.random.default_rng(0).permutation(len(y))[:CAP]]}
print('samples per class', np.bincount([c for c, _ in samples.values()]).tolist(), flush=True)

# Source models for Indian Pines -> Local: all labelled Indian Pines pixels, 10% for early stopping.
cube, gt = load_scene('IndianPines')
rc = np.argwhere(gt > 0)
ptr, pva = train_test_split(np.arange(len(rc)), test_size=0.1, stratify=gt[gt > 0], random_state=0)
src = scene_loaders(cube, gt, [tuple(rc[ptr].T), tuple(rc[pva].T), tuple(rc[pva].T)], 0)
pre = {}
for name in TRANSFER_NETS:
    set_seed(0)
    pre[name], info = train(MODELS[name](200, 16), *src[:2])
    log('r11', dict(exp='R11', stage='pretrain_ip', model=name, **info, **test(pre[name], src[1])))
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
        base = dict(exp='R11', dataset='Local', budget_samples_per_class=k, seed=seed, n_train=len(ytr),
                    n_test=len(yte), test_samples=';'.join(s.split('hyperspectral-dataset/')[-1] for s in test_ids))
        for kind in ['LightGBM', 'SVM']:
            log('r11', {**base, 'init': 'none', **run_classic(kind, Xtr, ytr, Xte, yte, seed)})
        if k == BUDGETS[-1]:
            shap_bands(Xtr, ytr, Xte, yte, f'local_samples_seed{seed}', 'r11')
        loaders = make_loaders([(Xtr, ytr, None), (Xva, yva, None), (Xte, yte, None)], seed)
        for name in NETS:
            for init in ['scratch'] + (['transfer'] if name in pre else []):
                set_seed(seed)
                m = MODELS[name](168, 3)
                copied = transfer(pre[name], m) if init == 'transfer' else []
                m, info = train(m, *loaders[:2], frozen=copied, freeze_epochs=FREEZE_EPOCHS if copied else 0)
                log('r11', {**base, 'model': name, 'init': init, 'n_copied': len(copied), **info, **test(m, loaders[2])})
                free(m)
finish('r11', cap_patches_per_sample=CAP, budgets=BUDGETS)
