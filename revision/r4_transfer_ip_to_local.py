from hsi_common import *
# R4: transfer Indian Pines -> Local soil. Target train = split fraction of the train samples;
# validation and test are the held-out val/test samples (never the test set for early stopping).

X, y = load_indian_pines()
split = local_file_split()
Ltr, ytr = load_local_files(split['train'])
Lva, yva = load_local_files(split['val'])
Lte, yte = load_local_files(split['test'])
NETS = ['HybridSN', 'SpectralFormer', 'ViT']

pre = {}
ptr, rest = train_test_split(np.arange(len(y)), train_size=0.8, stratify=y, random_state=0)
pva = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=0)[0]
src = make_loaders([(X, y, ptr), (X, y, pva), (X, y, pva)], 0)
for name in NETS:
    set_seed(0)
    pre[name], info = train(MODELS[name](200, 16), *src[:2])
    log('r4', dict(exp='R4', stage='pretrain_ip', model=name, **info, **test(pre[name], src[1])))
del src, X

for split_ in SPLITS:
    for seed in SEEDS:
        tr = train_test_split(np.arange(len(ytr)), train_size=split_, stratify=ytr, random_state=seed)[0]
        base = dict(exp='R4', dataset='Local', split=split_, seed=seed, n_train=len(tr))
        log('r4', {**base, 'init': 'none', **run_classic('LightGBM', Ltr[tr], ytr[tr], Lte, yte, seed)})
        loaders = make_loaders([(Ltr, ytr, tr), (Lva, yva, None), (Lte, yte, None)], seed)
        for name in NETS:
            for init in ['scratch', 'transfer']:
                set_seed(seed)
                m = MODELS[name](168, 3)
                copied = transfer(pre[name], m) if init == 'transfer' else []
                m, info = train(m, *loaders[:2], frozen=copied, freeze_epochs=FREEZE_EPOCHS if copied else 0)
                log('r4', {**base, 'model': name, 'init': init, 'n_copied': len(copied), **info, **test(m, loaders[2])})
                free(m)
finish('r4')
