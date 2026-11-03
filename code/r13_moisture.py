from hsi_common import *
# R13: does the Local classifier see soil or moisture? Soil patches only (>= 90% soil, as in R11b).
# (A) Cross-state: Red is the only class imaged both moist and dry. Train on Black (all moist), Yellow (all dry)
#     and one Red state, test on the other Red state; moist Red called Black means the model keys on moisture.
# (B) The R11b budgets again with the water bands removed, and with each spectrum divided by its mean.

CAP, BUDGETS = 300, ([2] if SMOKE else [1, 2, 3, 4])
WATER = np.r_[9:26, 93:125]  # band centres 932-1010 nm and 1344-1498 nm (wavelengths/local_pika_nir320.txt)
VARIANTS = {'raw': lambda X: X, 'nowater': lambda X: np.delete(X, WATER, axis=-1),
            'norm': lambda X: X / X.mean(-1, keepdims=True)}
NETS = globals().get('R13_NETS', ['HybridSN', 'SpectralFormer'])  # r13b_setup.py narrows the rerun
CLASSIC13 = globals().get('R13_CLASSIC', ['LightGBM', 'SVM'])
ONLY = globals().get('R13_VARIANTS')
LOG13 = globals().get('R13_LOG', 'r13')

files = local_files()
patches = {f: load_local_files([f], soil=(0.9, 1)) for f in files}


def group(key):
    """Soil patches per sample id, files in sorted order (so key=local_soil_id reproduces R11b exactly)."""
    out = {}
    for f in files:
        if len(patches[f][1]):
            out.setdefault(key(f), []).append(f)
    return {s: (np.concatenate([patches[f][0] for f in fs]), np.concatenate([patches[f][1] for f in fs])) for s, fs in out.items()}


def cap(src):
    return {s: (X[i], y[i]) for s, (X, y) in src.items() for i in [np.random.default_rng(0).permutation(len(y))[:CAP]]}


def stack(ids, source, f):
    return f(np.concatenate([source[s][0] for s in ids])), np.concatenate([source[s][1] for s in ids])


def shares(p):
    return {f'pred_{n}': float(v) for n, v in zip(LOCAL_CLASSES, np.bincount(p, minlength=3) / len(p))}


def run(tr, va, te, seed, base):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tr, va, te
    base = {**base, 'seed': seed, 'n_train': len(ytr), 'n_test': len(yte), 'bands': Xtr.shape[-1]}
    for kind in CLASSIC13:
        p = classic(kind, seed).fit(centre(Xtr), ytr).predict(centre(Xte))
        log(LOG13, {**base, 'model': kind, **metrics(yte, p), **shares(p)})
    loaders = make_loaders([(Xtr, ytr, None), (Xva, yva, None), (Xte, yte, None)], seed)
    for name in NETS:
        set_seed(seed)
        m, info = train(MODELS[name](Xtr.shape[-1], 3), *loaders[:2])
        _, y, p = evaluate(m, loaders[2])
        log(LOG13, {**base, 'model': name, **info, **metrics(y, p), **shares(p)})
        free(m)


# (A) cross-state, per image folder (Red Moist/n and Red 11-10-22/n stay apart here on purpose)
state = group(local_sample_id)
state_cap = cap(state)
cls = {s: int(y[0]) for s, (_, y) in state.items()}
red_moist = lambda s: cls[s] == 1 and re.search(r'[\\/]moist[\\/]', s, re.I) is not None
for variant, f in VARIANTS.items():
    if ONLY and variant not in ONLY:
        continue
    for test_moist in [True, False]:
        te = sorted(s for s in state if cls[s] == 1 and red_moist(s) == test_moist)
        pool = sorted(s for s in state if s not in te)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            va = [rng.choice(sorted(s for s in pool if cls[s] == c)) for c in range(3)]  # early stopping only
            tr = [s for s in pool if s not in va]
            run(stack(tr, state_cap, f), stack(va, state_cap, f), stack(te, state, f), seed,
                dict(exp='R13', part='cross_state', variant=variant, test_state='moist' if test_moist else 'dry'))

# (B) R11b budgets (same merged samples, same splits) without water bands / brightness-normalised
merged = group(local_soil_id)
merged_cap = cap(merged)
mcls = {s: int(y[0]) for s, (_, y) in merged.items()}
for variant in [v for v in ['nowater', 'norm'] if not ONLY or v in ONLY]:
    f = VARIANTS[variant]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = {c: list(rng.permutation(sorted(s for s in merged if mcls[s] == c))) for c in range(3)}
        te = [s for c in range(3) for s in order[c][:2]]
        va = [s for c in range(3) for s in order[c][2:3]]
        for k in BUDGETS:
            tr = [s for c in range(3) for s in order[c][3:3 + k]]
            run(stack(tr, merged_cap, f), stack(va, merged_cap, f), stack(te, merged, f), seed,
                dict(exp='R13', part='budget', variant=variant, budget_samples_per_class=k))
finish(LOG13, cap_patches_per_sample=CAP, budgets=BUDGETS, water_bands=WATER.tolist(), min_soil_fraction=0.9)
