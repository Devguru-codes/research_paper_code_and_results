from hsi_common import *
# R1: few-shot curve on Indian Pines (Table II + crossover figure). All models, same splits and protocol.

X, y = load_indian_pines()
for split in SPLITS:
    for seed in SEEDS:
        tr, va, te = ip_split(y, split, seed)
        base = dict(exp='R1', dataset='IndianPines', split=split, seed=seed, n_train=len(tr))
        log('r1', {**base, **run_classic('LightGBM', X[tr], y[tr], X[te], y[te], seed)})
        loaders = make_loaders([(X, y, tr), (X, y, va), (X, y, te)], seed)
        for name in ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN']:
            set_seed(seed)
            m, info = train(MODELS[name](X.shape[-1], 16), *loaders[:2])
            log('r1', {**base, 'model': name, **info, **test(m, loaders[2])})
            free(m)
finish('r1')
