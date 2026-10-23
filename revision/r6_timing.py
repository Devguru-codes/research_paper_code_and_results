from hsi_common import *
# R6: Table V timing. Batch size 1, 1000 Indian Pines test patches, median of 3 passes.
# Deep models on the GPU (untrained weights: speed does not depend on the values); LightGBM on the CPU.

X, y = load_indian_pines()
tr, va, te = ip_split(y, 0.05, 42)
te = te[:50 if SMOKE else 1000]
mean, std = band_stats(X, tr)
xs = [torch.from_numpy(np.ascontiguousarray(((X[i] - mean) / std).transpose(2, 0, 1))[None, None]) for i in te]


def median_time(fn, reps=3):
    times = []
    for _ in range(reps):
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


for name in ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN']:
    m = MODELS[name](200, 16).to(DEVICE).eval()
    with torch.no_grad():
        for x in xs[:20]: m(x.to(DEVICE))  # warm-up
        t = median_time(lambda: [m(x.to(DEVICE)) for x in xs])
    log('r6', dict(exp='R6', model=name, platform=torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
                   params=sum(p.numel() for p in m.parameters()), n=len(xs), total_s=t, ms_per_patch=1000 * t / len(xs)))
    free(m)

clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(centre(X[tr]), y[tr])
rows = [r[None] for r in centre(X[te])]
t = median_time(lambda: [clf.predict(r) for r in rows])
log('r6', dict(exp='R6', model='LightGBM', platform=f'cpu x{os.cpu_count()}', params='100 trees', n=len(rows),
               total_s=t, ms_per_patch=1000 * t / len(rows)))
finish('r6')
