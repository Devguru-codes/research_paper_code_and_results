"""Analysis for the letter: how evaluation design changes the model ranking.

Airborne scenes: random pixel split (R1/R18 Indian Pines, R19 Pavia U and Salinas) vs spatially disjoint tiles (R8/R12).
Local soil: raw patches (R11 + R12 LightGBM) vs soil-only patches (R11b) vs soil-only + brightness-normalised (R13),
plus the tray-only leakage check (R11b) and the moist/dry cross-state test (R13).
Reads revision/results/session-{a,e,f,fc,g,gc,h,i,j,k,l,m,n,o,p,q,s,t,u,v,w,x,y,z,r20a-d}/ and writes revision/results/final/{tables,figures}/.
FDR families (Benjamini-Hochberg): (1) every model-vs-LightGBM test on the airborne scenes, both split protocols;
(2) every model-vs-LightGBM test on Local soil, all three patch protocols.
Run: python revision/analysis.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stats_utils import paired, bh, grid, fmt, saver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('RSL_RESULTS', os.path.join(HERE, 'results'))
FIG = os.path.join(RES, 'final', 'figures')
os.makedirs(FIG, exist_ok=True)
save = saver(os.path.join(RES, 'final', 'tables'))

NAME = {'LightGBM': 'LightGBM', 'SVM': 'SVM', 'RF': 'Random forest', 'ViT': 'Spectral transformer', 'SpectralFormer': 'Grouped-band transformer',
        'ConvStemViT': 'Conv-stem transformer', 'HybridSN': 'HybridSN'}
NETS = ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN']
SOIL = ['LightGBM', 'SVM', 'SpectralFormer', 'HybridSN']  # models run under all three soil protocols
SCENES = {'IndianPines': ('e', 'Indian Pines'), 'PaviaU': ('f', 'Pavia University'), 'Salinas': ('g', 'Salinas')}
K = 'budget_samples_per_class'


def load(session, name):
    df = pd.read_csv(os.path.join(RES, f'session-{session}', f'{name}.csv'))
    return df[df['stage'].isna()] if 'stage' in df else df


def load_scene_parts(key, session):
    """One scene's rows: a single GPU run (Indian Pines) or a CPU part + a GPU part (Pavia U, Salinas)."""
    parts = [(session, f'r8_{key.lower()}'), (session + 'c', f'r8_{key.lower()}_cpu'), (session, f'r8_{key.lower()}_gpu')]
    found = [load(s, n) for s, n in parts if os.path.exists(os.path.join(RES, f'session-{s}', f'{n}.csv'))]
    return pd.concat(found, ignore_index=True) if found else None


# LightGBM rows of R8/R11 come from R12 (stable settings, identical splits); those logged by R8/R11 are dropped.
r12 = load('l', 'r12')


def with_r12(df, dataset):
    return pd.concat([df[df.model != 'LightGBM'], r12[r12.dataset == dataset]], ignore_index=True)


def scratch(df):
    return df[df.init.isin(['none', 'scratch'])]


# R18: stable LightGBM on the random split (replaces R1's default LightGBM rows), SVM and RF on the disjoint splits;
# session m: grouped-band transformer on Pavia U with zero-padded bands (replaces the run that dropped 3 bands).
r18 = load('k', 'r18')
sf_pad = load('m', 'r8_paviau_sfpad')


def assemble(k, df):
    df = with_r12(df, k)
    if k == 'PaviaU':
        df = pd.concat([df[df.model != 'SpectralFormer'], sf_pad], ignore_index=True)
    return scratch(pd.concat([df, r18[(r18.part == 'disjoint') & (r18.dataset == k)]], ignore_index=True))


scenes = {k: assemble(k, df) for k, (s, _) in SCENES.items() if (df := load_scene_parts(k, s)) is not None}
r1 = load('a', 'r1').assign(init='none')
ip_random = pd.concat([r1[r1.model != 'LightGBM'], r18[r18.part == 'random']], ignore_index=True)
r19 = pd.concat([load(s_, n) for s_, n in [('w', 'r19_paviau_gpu'), ('x', 'r19_salinas_gpu_a'), ('y', 'r19_salinas_gpu_b'),
                                           ('z', 'r19_paviau_cpu'), ('z', 'r19_salinas_cpu')]], ignore_index=True)
random = {'IndianPines': ip_random, **{k: r19[r19.dataset == k] for k in ['PaviaU', 'Salinas']}}
CLASSIC = ['LightGBM', 'RF', 'SVM']
r11b = load('i', 'r11b')
r13 = load('j', 'r13')
r13b = load('q', 'r13b')  # grouped-band transformer, water bands removed, last group zero-padded (R13 dropped 3 bands)
r13 = pd.concat([r13[~((r13.model == 'SpectralFormer') & (r13.variant == 'nowater'))], r13b], ignore_index=True)
soil = {'Raw patches': scratch(with_r12(load('h', 'r11'), 'Local')),
        'Soil-only patches': scratch(r11b[~r11b.model.str.endswith('_background', na=False)]),
        'Soil-only, brightness-normalised': r13[(r13.part == 'budget') & (r13.variant == 'norm')]}


def vs_lightgbm(g, models, ref='LightGBM', **key):
    """Paired over the seeds both models have (joined on seed, so a missing run cannot shift the pairing)."""
    out = []
    for m in models:
        if m != ref and (g.model == m).any():
            j = g[g.model == ref][['seed', 'OA']].merge(g[g.model == m][['seed', 'OA']], on='seed', validate='1:1')
            out.append(dict(**key, model=NAME[m], reference=NAME[ref], **paired(j.OA_x, j.OA_y)))
    return out


# ---- Airborne: random pixel split vs disjoint split on each scene, and the three disjoint scenes
rows, tests = [], []
for sc in SCENES:
    for proto, df in [('Random pixels', random[sc]), ('Disjoint tiles', scenes[sc])]:
        for s, g in df.groupby('split'):
            rows.append({'Dataset': SCENES[sc][1], 'Split': proto, 'Labelled': f'{s * 100:g}%',
                         **{NAME[m]: fmt(v.mean(), v.std()) for m in ['LightGBM'] + NETS for v in [g[g.model == m].OA]}})
            if proto == 'Random pixels' or sc == 'IndianPines':  # disjoint Pavia U / Salinas nets are tested below
                tests += vs_lightgbm(g, NETS, dataset=SCENES[sc][1], split=proto, budget=f'{s * 100:g}%')
save(pd.DataFrame(rows), 'table_protocols', 'OA (%), random pixel split vs spatially disjoint tiles (mean ± SD over 5 seeds).')

rows = []
for sc, df in scenes.items():
    for s, g in df.groupby('split'):
        rows.append({'Dataset': SCENES[sc][1], 'Labelled': f'{s * 100:g}%', 'Test px': f'{int(g.n_test.min())}-{int(g.n_test.max())}',
                     **{NAME[m]: fmt(v.mean(), v.std()) if len(v) else '–' for m in CLASSIC + NETS for v in [g[g.model == m].OA]}})
        key = dict(dataset=SCENES[sc][1], split='Disjoint tiles', budget=f'{s * 100:g}%')
        tests += vs_lightgbm(g, ['RF', 'SVM'] + ([] if sc == 'IndianPines' else NETS), **key)
        tests += vs_lightgbm(g, ['HybridSN'], ref='RF', **key)  # against the strongest classical baseline
save(pd.DataFrame(rows), 'table_scenes', 'Overall accuracy (%) on spatially disjoint test areas, mean ± SD over 5 seeds (tile splits).')
save(bh(pd.DataFrame(tests)).round(4), 'stats_airborne_vs_lightgbm', 'Model minus reference (LightGBM, or RF for HybridSN) OA (pp), paired over seeds; one BH-FDR family (both split protocols).')

# ---- Local soil: three patch protocols
rows, tests = [], []
for proto, df in soil.items():
    for k, g in df.groupby(K):
        rows.append({'Patches': proto, 'Samples per class': int(k),
                     **{NAME[m]: fmt(v.mean(), v.std()) for m in SOIL for v in [g[g.model == m].OA]}})
        tests += vs_lightgbm(g, SOIL, protocol=proto, samples_per_class=int(k))
save(pd.DataFrame(rows), 'table_soil_protocols', 'Local soil OA (%) on held-out samples under three patch protocols (mean ± SD over 5 repeated sample-level splits).')
save(bh(pd.DataFrame(tests)).round(4), 'stats_soil_vs_lightgbm', 'Local soil: model minus LightGBM OA (pp), paired over repeats; one BH-FDR family (all protocols).')

# ---- Leakage checks: tray-only patches (R11b) and Red moist/dry cross-state test (R13)
tray = r11b[r11b.model.str.endswith('_background', na=False)]
save(tray.groupby('model').OA.agg(['mean', 'std', 'min', 'max']).round(2).reset_index(), 'table_tray_only',
     'LightGBM/SVM trained and tested on tray-only patches (<= 10% soil) of the same samples, 4 samples per class; chance = 33%.')
cs = r13[r13.part == 'cross_state']
rows = []
for (state, model), g in cs.groupby(['test_state', 'model'], sort=False):
    rows.append({'Red test state': state, 'Model': NAME[model],
                 **{f'{v}: Red / Black': f"{100 * h.pred_red.mean():.0f} / {100 * h.pred_black.mean():.0f}"
                    for v in ['raw', 'nowater', 'norm'] for h in [g[g.variant == v]]}})
save(pd.DataFrame(rows), 'table_cross_state', 'Red test patches predicted Red / Black (%) when the other Red moisture state is in training (mean over 5 seeds).')

# ---- Figures
# ---- Robustness to training length: R15 reruns every deep model with up to 100 epochs on the same disjoint splits
long = pd.concat([pd.read_csv(f) for s_ in 'nopstuv' for f in [os.path.join(RES, f'session-{s_}', n) for n in
                  os.listdir(os.path.join(RES, f'session-{s_}'))] if '_long' in f and f.endswith('.csv')], ignore_index=True)
long = long[long.init == 'scratch']
rows, tests = [], []
for sc, df in scenes.items():
    for (s, m), g in long[long.dataset == sc].groupby(['split', 'model']):
        h = df[(df.split == s) & (df.model == m)]
        rows.append({'Dataset': SCENES[sc][1], 'Labelled': f'{s * 100:g}%', 'Model': NAME[m], 'OA, 50 epochs': fmt(h.OA.mean(), h.OA.std()),
                     'OA, 100 epochs': fmt(g.OA.mean(), g.OA.std()), 'At cap (50)': f'{(h.epochs >= 50).mean():.0%}',
                     'At cap (100)': f'{(g.epochs >= 100).mean():.0%}'})
        ref = df[(df.split == s)]
        key = dict(dataset=SCENES[sc][1], budget=f'{s * 100:g}%')
        tests += vs_lightgbm(pd.concat([ref[ref.model.isin(['LightGBM', 'RF'])], g]), [m], **key)
        if m == 'HybridSN':
            tests += vs_lightgbm(pd.concat([ref[ref.model == 'RF'], g]), [m], ref='RF', **key)
save(pd.DataFrame(rows), 'table_long_training', 'Disjoint splits: OA (%) with at most 50 vs 100 epochs (same splits and seeds), and share of runs stopped by the epoch cap.')
save(bh(pd.DataFrame(tests)).round(4), 'stats_long_vs_baselines', '100-epoch models minus LightGBM (or RF for HybridSN), OA (pp); one BH-FDR family.')

# ---- Matched label budget: R20 reruns HybridSN and the conv-stem transformer with early stopping on 20% of the p
# training pixels (no extra validation labels), on the R8 splits, so R12 LightGBM and R18 RF stay paired baselines
r20 = pd.concat([pd.read_csv(f) for s_ in 'abcd' for f in [os.path.join(RES, f'session-r20{s_}', n) for n in
                 os.listdir(os.path.join(RES, f'session-r20{s_}'))] if f.endswith('.csv')], ignore_index=True)
rows, tests = [], []
for sc, df in scenes.items():
    for (s, m), g in r20[r20.dataset == sc].groupby(['split', 'model']):
        h, ref = df[(df.split == s) & (df.model == m)], df[df.split == s]
        rows.append({'Dataset': SCENES[sc][1], 'Labelled': f'{s * 100:g}%', 'Model': NAME[m], 'OA, extra validation': fmt(h.OA.mean(), h.OA.std()),
                     'OA, matched budget': fmt(g.OA.mean(), g.OA.std()), 'LightGBM': fmt(*(lambda v: (v.mean(), v.std()))(ref[ref.model == 'LightGBM'].OA)),
                     'RF': fmt(*(lambda v: (v.mean(), v.std()))(ref[ref.model == 'RF'].OA))})
        key = dict(dataset=SCENES[sc][1], budget=f'{s * 100:g}%')
        tests += vs_lightgbm(pd.concat([ref[ref.model == 'LightGBM'], g]), [m], **key)
        tests += vs_lightgbm(pd.concat([ref[ref.model == 'RF'], g]), [m], ref='RF', **key)
save(pd.DataFrame(rows), 'table_matched_budget', 'Disjoint splits: deep models with up to 3,000 extra validation pixels (R8) vs early stopping on 20% of the p labelled pixels (R20); LightGBM and RF use the p pixels.')
save(bh(pd.DataFrame(tests)).round(4), 'stats_matched_vs_baselines', 'Matched-budget models minus LightGBM or RF, OA (pp); one BH-FDR family.')

STY = {'LightGBM': 'k-o', 'SVM': '-x', 'ViT': '-v', 'SpectralFormer': '-d', 'ConvStemViT': '-^', 'HybridSN': '-s'}


def curves(ax, df, models, title, x='split', scale=100, log=True):
    for m in models:
        if (df.model == m).any():
            b, oa = grid(df, m, x=x, scale=scale)
            ax.errorbar(b, oa.mean(1), yerr=oa.std(1, ddof=1), fmt=STY[m], capsize=3, label=NAME[m])
    if log:
        ax.set_xscale('log')
    ax.set_xticks(b); ax.set_xticklabels([f'{v:g}' for v in b])
    ax.set_title(title, fontsize=10); ax.grid(alpha=.3)


fig, axes = plt.subplots(1, len(scenes), figsize=(4.4 * len(scenes), 3.8), squeeze=False)
for ax, (sc, df) in zip(axes[0], scenes.items()):
    curves(ax, df, ['LightGBM'] + NETS, SCENES[sc][1]); ax.set_xlabel('Labelled pixels (%)')
axes[0][0].set_ylabel('Overall accuracy (%)'); axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_scenes.png'), dpi=300); plt.close(fig)

panels = [('Indian Pines, random pixels', ip_random, 'split', 100, True), ('Indian Pines, disjoint tiles', scenes['IndianPines'], 'split', 100, True)] + \
         [(f'Local soil, {p[0].lower() + p[1:]}', d, K, 1, False) for p, d in soil.items()]
fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), sharey=True)
for ax, (title, df, x, scale, lg) in zip(axes, panels):
    curves(ax, df, ['LightGBM', 'SpectralFormer', 'HybridSN'], title, x=x, scale=scale, log=lg)
    ax.set_xlabel('Labelled pixels (%)' if x == 'split' else 'Labelled samples per class')
axes[0].set_ylabel('Overall accuracy (%)'); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_protocols.png'), dpi=300); plt.close(fig)

# ---- Classification maps (R17, seed 42): ground truth, LightGBM and HybridSN on every labelled pixel
r17 = load('n', 'r17')
fig, axes = plt.subplots(3, 3, figsize=(10, 11))
for row, (sc, (_, title)) in zip(axes, SCENES.items()):
    z = np.load(os.path.join(RES, 'session-n', f'map_{sc}.npz'))
    for ax, (lab, key) in zip(row, [('Ground truth', 'gt'), ('LightGBM', 'LightGBM'), ('HybridSN', 'HybridSN')]):
        img = np.full(z['gt'].shape, np.nan)
        img[z['rows'], z['cols']] = z['gt'][z['rows'], z['cols']] if key == 'gt' else z[key] + 1
        ax.imshow(img, cmap='tab20', vmin=1, vmax=20, interpolation='nearest')
        if key == 'gt':
            ax.scatter(z['train_c'], z['train_r'], s=1, c='k', marker='.', linewidths=0)
            ax.set_title(f'{title}: ground truth', fontsize=9)
        else:
            oa = r17[(r17.dataset == sc) & (r17.model == key)].OA.iloc[0]
            ax.set_title(f'{lab} (test OA {oa:.1f}%)', fontsize=9)
        ax.axis('off')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_maps.png'), dpi=200); plt.close(fig)
save(r17[['dataset', 'split', 'model', 'OA', 'AA', 'Kappa']].round(2), 'table_maps', 'Seed-42 map runs: accuracy on the disjoint test area.')

# ---- SHAP against true band-centre wavelengths (soil-only patches; Indian Pines disjoint 5%)
WL = {k: np.loadtxt(os.path.join(HERE, 'wavelengths', f), comments='#')
      for k, f in [('local', 'local_pika_nir320.txt'), ('ip', 'indianpines_aviris1992.txt')]}
shap_local = [pd.read_csv(os.path.join(RES, 'session-i', f)).total.to_numpy()
              for f in sorted(os.listdir(os.path.join(RES, 'session-i'))) if f.startswith('shap_local_masked')]
panels = [('local', np.mean(shap_local, 0), f'Local soil, soil-only patches (LightGBM, 4 samples/class; mean of {len(shap_local)} splits)'),
          ('ip', pd.read_csv(os.path.join(RES, 'session-l', 'shap_indianpines_disjoint_5pct.csv')).total.to_numpy(),
           'Indian Pines (LightGBM, 5% labelled, disjoint split)')]
fig, axes = plt.subplots(2, 1, figsize=(7, 5.6))
top = []
for ax, (key, v, title) in zip(axes, panels):
    o = np.argsort(WL[key]); x, y = WL[key][o], v[o]
    gap = np.nonzero(np.diff(x) > 3 * np.median(np.diff(x)))[0] + 1  # removed water bands: break the line
    ax.plot(np.insert(x, gap, np.nan), np.insert(y, gap, np.nan), lw=1.2, color='k')
    peaks = []
    for i in np.argsort(-y):
        if all(abs(x[i] - x[p]) >= 30 for p in peaks):
            peaks.append(i)
        if len(peaks) == 3:
            break
    for i in peaks:
        ax.annotate(f'{x[i]:.0f} nm', (x[i], y[i]), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
        top.append(dict(dataset=key, band=int(o[i]), wavelength_nm=round(float(x[i]), 1), mean_abs_shap=round(float(y[i]), 4)))
    ax.set_title(title, fontsize=10); ax.set_ylabel('mean |SHAP|'); ax.grid(alpha=.3)
axes[-1].set_xlabel('Band-centre wavelength (nm)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_shap.png'), dpi=300); plt.close(fig)
save(pd.DataFrame(top), 'shap_top_bands', 'Top-3 LightGBM bands by mean |SHAP| (summed over classes), true band centres.')
print('\nwrote', sorted(os.listdir(FIG)))
