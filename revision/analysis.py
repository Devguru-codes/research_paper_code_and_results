"""Analysis of the second-round reruns (R8: spatially disjoint scenes, R11: Local sample-level budgets).

Reads revision/results/session-{e,f,fc,g,gc,h,l}/ (LightGBM and SHAP from l) and writes revision/results/final/{tables,figures}/.
FDR families (Benjamini-Hochberg): (1) every model-vs-LightGBM test on the three scenes, (2) every
model-vs-LightGBM test on Local soil, (3) every transfer-vs-scratch test in both directions.
Run: python revision/analysis.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stats_utils import paired, bh, crossover_ci, grid, fmt, saver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('RSL_RESULTS', os.path.join(HERE, 'results'))
FIG = os.path.join(RES, 'final', 'figures')
os.makedirs(FIG, exist_ok=True)
save = saver(os.path.join(RES, 'final', 'tables'))

NAME = {'LightGBM': 'LightGBM', 'SVM': 'SVM', 'ViT': 'Spectral transformer', 'SpectralFormer': 'Grouped-band transformer',
        'ConvStemViT': 'Conv-stem transformer', 'HybridSN': 'HybridSN'}
NETS, TNETS = ['ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN'], ['HybridSN', 'SpectralFormer', 'ViT']
SCENES = {'IndianPines': ('e', 'Indian Pines'), 'PaviaU': ('f', 'Pavia University'), 'Salinas': ('g', 'Salinas')}


def load(session, name):
    df = pd.read_csv(os.path.join(RES, f'session-{session}', f'{name}.csv'))
    return df[df['stage'].isna()] if 'stage' in df else df


def load_scene_parts(key, session):
    """One scene's rows: a single GPU run (Indian Pines) or a CPU part + a GPU part (Pavia U, Salinas)."""
    parts = [(session, f'r8_{key.lower()}'), (session + 'c', f'r8_{key.lower()}_cpu'), (session, f'r8_{key.lower()}_gpu')]
    found = [load(s, n) for s, n in parts if os.path.exists(os.path.join(RES, f'session-{s}', f'{n}.csv'))]
    return pd.concat(found, ignore_index=True) if found else None


# LightGBM rows come from R12 (stable settings, identical splits); those logged by R8/R11 are dropped.
r12 = load('l', 'r12')


def with_r12(df, dataset):
    return pd.concat([df[df.model != 'LightGBM'], r12[r12.dataset == dataset]], ignore_index=True)


scenes = {k: with_r12(df, k) for k, (s, _) in SCENES.items() if (df := load_scene_parts(k, s)) is not None}
local = with_r12(load('h', 'r11'), 'Local')
K = 'budget_samples_per_class'

# ---- Scenes: accuracy by budget (scratch models) and tests vs LightGBM
rows, tests = [], []
for sc, df in scenes.items():
    for s, g in df.groupby('split'):
        row = {'Dataset': SCENES[sc][1], 'Labelled': f'{s * 100:g}%', 'Test px': int(g.n_test.iloc[0]),
               'Classes (train/test)': f"{int(g.train_classes.min())}/{int(g.test_classes.min())}"}
        lg = g[g.model == 'LightGBM'].sort_values('seed').OA
        for m in ['LightGBM'] + NETS:
            v = g[(g.model == m) & (g.init.isin(['none', 'scratch']))].sort_values('seed').OA
            row[NAME[m]] = fmt(v.mean(), v.std()) if len(v) else '–'  # GPU part of Pavia U / Salinas may be missing
            if m != 'LightGBM' and len(v):
                tests.append(dict(dataset=SCENES[sc][1], model=NAME[m], split=f'{s * 100:g}%', **paired(lg, v)))
        rows.append(row)
save(pd.DataFrame(rows), 'table_scenes', 'Overall accuracy (%) on spatially disjoint test areas, mean ± SD over 5 seeds (tile splits).')
save(bh(pd.DataFrame(tests)).round(4), 'stats_scenes_vs_lightgbm', 'Model minus LightGBM OA (pp), paired over seeds; one BH-FDR family.')

# ---- Local: accuracy by labelled samples per class, tests vs LightGBM
rows, tests = [], []
for k, g in local.groupby(K):
    row = {'Samples per class': int(k), 'Train patches': int(g.n_train.iloc[0])}
    lg = g[g.model == 'LightGBM'].sort_values('seed').OA
    for m in ['LightGBM', 'SVM'] + NETS:
        v = g[(g.model == m) & (g.init.isin(['none', 'scratch']))].sort_values('seed').OA
        row[NAME[m]] = fmt(v.mean(), v.std())
        if m != 'LightGBM':
            tests.append(dict(model=NAME[m], samples_per_class=int(k), **paired(lg, v)))
    rows.append(row)
save(pd.DataFrame(rows), 'table_local_samples', 'Local soil OA (%) by labelled samples per class; 5 repeated sample-level splits (2 test samples per class).')
save(bh(pd.DataFrame(tests)).round(4), 'stats_local_vs_lightgbm', 'Local soil: model minus LightGBM OA (pp), paired over repeats; one BH-FDR family.')

# ---- Transfer: Local -> Indian Pines (R8) and Indian Pines -> Local (R11)
tests = []
for label, df, x in [('Local -> Indian Pines', scenes.get('IndianPines'), 'split'), ('Indian Pines -> Local', local, K)]:
    if df is None:
        continue
    for m in TNETS:
        for b, g in df[df.model == m].groupby(x):
            a = g[g.init == 'scratch'].sort_values('seed').OA.to_numpy()
            t = g[g.init == 'transfer'].sort_values('seed').OA.to_numpy()
            lg = df[(df.model == 'LightGBM') & (df[x] == b)].OA.mean()
            tests.append(dict(direction=label, model=NAME[m], budget=f'{b * 100:g}%' if x == 'split' else f'{int(b)} samples/class',
                              lightgbm=lg, scratch=a.mean(), transfer=t.mean(), **paired(a, t)))
save(bh(pd.DataFrame(tests)).round(4), 'stats_transfer', 'Transfer minus scratch OA (pp), paired over seeds; one BH-FDR family over both directions.')

# ---- Crossover with LightGBM
cross = []
for sc, df in scenes.items():
    sp, ref = grid(df, 'LightGBM', 'none')
    for m in [m for m in NETS if (df.model == m).any()]:
        cross.append(dict(dataset=SCENES[sc][1], model=NAME[m], init='scratch', unit='% labelled', **crossover_ci(sp, ref, grid(df, m, 'scratch')[1])))
    if sc == 'IndianPines':
        for m in TNETS:
            cross.append(dict(dataset=SCENES[sc][1], model=NAME[m], init='Local-pretrained', unit='% labelled', **crossover_ci(sp, ref, grid(df, m, 'transfer')[1])))
sp, ref = grid(local, 'LightGBM', 'none', x=K, scale=1)
for m in NETS:
    for init in ['scratch'] + (['transfer'] if m in TNETS else []):
        cross.append(dict(dataset='Local soil', model=NAME[m], init='IP-pretrained' if init == 'transfer' else init,
                          unit='samples/class', **crossover_ci(sp, ref, grid(local, m, init, x=K, scale=1)[1])))
save(pd.DataFrame(cross).round(3), 'crossover', 'Budget after which a model stays ahead of LightGBM; 95% CI from 5000 seed resamples. '
     'below_first / never = share of resamples where it already wins at the smallest budget / never wins.')


# ---- Figures
def curves(ax, df, series, title, x='split', scale=100, log=True):
    for label, m, init, style in series:
        b, oa = grid(df, m, init, x=x, scale=scale)
        ax.errorbar(b, oa.mean(1), yerr=oa.std(1, ddof=1), fmt=style, capsize=3, label=label)
    if log:
        ax.set_xscale('log')
    ax.set_xticks(b); ax.set_xticklabels([f'{v:g}' for v in b])
    ax.set_title(title, fontsize=10); ax.set_ylabel('Overall accuracy (%)'); ax.grid(alpha=.3)


STY = {'LightGBM': 'k-o', 'ViT': '-v', 'SpectralFormer': '-d', 'ConvStemViT': '-^', 'HybridSN': '-s'}
fig, axes = plt.subplots(1, len(scenes), figsize=(4.4 * len(scenes), 3.8), squeeze=False)
for ax, (sc, df) in zip(axes[0], scenes.items()):
    curves(ax, df, [(NAME[m], m, 'none' if m == 'LightGBM' else 'scratch', STY[m]) for m in ['LightGBM'] + NETS if (df.model == m).any()], SCENES[sc][1])
    ax.set_xlabel('Labelled pixels (%)')
axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_scenes.png'), dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, m in zip(axes, TNETS):
    curves(ax, local, [('LightGBM', 'LightGBM', 'none', 'k-o'), (f'{NAME[m]}, scratch', m, 'scratch', '--s'),
                       (f'{NAME[m]}, IP-pretrained', m, 'transfer', '-s')], f'Local soil: {NAME[m]}', x=K, scale=1, log=False)
    ax.set_xlabel('Labelled samples per class'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_local_samples.png'), dpi=300); plt.close(fig)

tr = pd.read_csv(os.path.join(RES, 'final', 'tables', 'stats_transfer.csv'))
fig, axes = plt.subplots(1, tr.direction.nunique(), figsize=(11, 3.8), sharey=True, squeeze=False)
for ax, (d, g) in zip(axes[0], tr.groupby('direction', sort=False)):
    budgets = list(dict.fromkeys(g.budget))
    for j, m in enumerate(TNETS):
        v = g[g.model == NAME[m]]
        x = np.arange(len(budgets)) + (j - 1) * 0.25
        ax.bar(x, v['diff'], 0.25, yerr=[v['diff'] - v.ci_low, v.ci_high - v['diff']], capsize=3, label=NAME[m])
    ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([b.replace(' samples/class', '') for b in budgets])
    ax.set_xlabel('Labelled target pixels (%)' if '%' in budgets[0] else 'Labelled target samples per class')
    ax.set_title(d, fontsize=10); ax.grid(axis='y', alpha=.3)
axes[0][0].set_ylabel('Transfer gain, OA (pp, 95% CI)'); axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_transfer_gain.png'), dpi=300); plt.close(fig)

# ---- SHAP against true band-centre wavelengths
WL = {k: np.loadtxt(os.path.join(HERE, 'wavelengths', f), comments='#')
      for k, f in [('local', 'local_pika_nir320.txt'), ('ip', 'indianpines_aviris1992.txt')]}
shap_local = [pd.read_csv(os.path.join(RES, 'session-l', f)).total.to_numpy()
              for f in sorted(os.listdir(os.path.join(RES, 'session-l'))) if f.startswith('shap_local_samples')]
shap_ip = os.path.join(RES, 'session-l', 'shap_indianpines_disjoint_5pct.csv')
panels = [('local', np.mean(shap_local, 0), f'Local soil (LightGBM, all training samples; mean of {len(shap_local)} splits)')]
if os.path.exists(shap_ip):
    panels.append(('ip', pd.read_csv(shap_ip).total.to_numpy(), 'Indian Pines (LightGBM, 5% labelled, disjoint split)'))
fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.8 * len(panels)), squeeze=False)
top = []
for ax, (key, v, title) in zip(axes[:, 0], panels):
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
axes[-1, 0].set_xlabel('Band-centre wavelength (nm)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_shap.png'), dpi=300); plt.close(fig)
save(pd.DataFrame(top), 'shap_top_bands', 'Top-3 LightGBM bands by mean |SHAP| (summed over classes), true band centres.')
print('\nwrote', sorted(os.listdir(FIG)))
