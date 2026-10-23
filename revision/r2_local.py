from hsi_common import *
# R2: Local soil benchmark (Table I). Physical samples split 60/20/20 per class; mixed-soil folders excluded.

split = local_file_split()
Ltr, ytr = load_local_files(split['train'])
Lva, yva = load_local_files(split['val'])
Lte, yte = load_local_files(split['test'])
counts = {k: np.bincount(v, minlength=3).tolist() for k, v in [('train', ytr), ('val', yva), ('test', yte)]}
print('patches per class', counts, flush=True)
for seed in SEEDS:
    base = dict(exp='R2', dataset='Local', seed=seed, n_train=len(ytr))
    for kind in ['LightGBM', 'SVM']:
        log('r2', {**base, **run_classic(kind, Ltr, ytr, Lte, yte, seed)})
    loaders = make_loaders([(Ltr, ytr, None), (Lva, yva, None), (Lte, yte, None)], seed)
    for name in ['ViT', 'SpectralFormer', 'HybridSN']:
        set_seed(seed)
        m, info = train(MODELS[name](168, 3), *loaders[:2])
        log('r2', {**base, 'model': name, **info, **test(m, loaders[2])})
        free(m)
finish('r2', files={k: len(v) for k, v in split.items()}, patches_per_class=counts)
