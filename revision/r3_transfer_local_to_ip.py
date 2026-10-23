from hsi_common import *
# R3: transfer Local soil -> Indian Pines (the "negative transfer" direction), scratch vs. transfer per split/seed.

split = local_file_split()
Ltr, ytr = load_local_files(split['train'])
Lva, yva = load_local_files(split['val'])
X, y = load_indian_pines()
NETS = ['HybridSN', 'SpectralFormer', 'ViT']

pre = {}
src = make_loaders([(Ltr, ytr, None), (Lva, yva, None), (Lva, yva, None)], 0)
for name in NETS:
    set_seed(0)
    pre[name], info = train(MODELS[name](168, 3), *src[:2])
    log('r3', dict(exp='R3', stage='pretrain_local', model=name, **info, **test(pre[name], src[1])))
del src, Ltr, Lva

for split_ in SPLITS:
    for seed in SEEDS:
        tr, va, te = ip_split(y, split_, seed)
        base = dict(exp='R3', dataset='IndianPines', split=split_, seed=seed, n_train=len(tr))
        log('r3', {**base, 'init': 'none', **run_classic('LightGBM', X[tr], y[tr], X[te], y[te], seed)})
        loaders = make_loaders([(X, y, tr), (X, y, va), (X, y, te)], seed)
        for name in NETS:
            for init in ['scratch', 'transfer']:
                set_seed(seed)
                m = MODELS[name](200, 16)
                copied = transfer(pre[name], m) if init == 'transfer' else []
                m, info = train(m, *loaders[:2], frozen=copied, freeze_epochs=FREEZE_EPOCHS if copied else 0)
                log('r3', {**base, 'model': name, 'init': init, 'n_copied': len(copied), **info, **test(m, loaders[2])})
                free(m)
finish('r3')
