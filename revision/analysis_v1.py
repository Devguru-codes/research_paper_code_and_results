"""CPU analysis of the rerun results (REVISION_PLAN.md section 3).

Reads revision/results/session-*/r*.csv and writes revision/results/tables/*.csv|.tex and
revision/results/figures/*.png. Stats: paired t-test (seeds paired), 95% CI of the mean
difference, Cohen's d_z, Benjamini-Hochberg FDR within each family of tests.
Crossover: piecewise-linear interpolation of mean OA curves; 95% CI by resampling seeds.
Run: python revision/analysis_v1.py  (first-round results R1-R7; superseded by analysis.py)
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stats_utils import paired, bh, crossover_ci, grid, fmt, saver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TAB, FIG = os.path.join(RES, 'tables'), os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)
save = saver(TAB)


def load(name, session):
    df = pd.read_csv(os.path.join(RES, f'session-{session}', f'{name}.csv'))
    return df[df['stage'].isna()] if 'stage' in df else df


r1, r2, r3, r4, r5, r6 = load('r1', 'a'), load('r2', 'c'), load('r3', 'b'), load('r4', 'c'), load('r5', 'b'), load('r6', 'a')

# ---- Table: Local benchmark (R2)
t = r2.groupby('model').agg(OA=('OA', 'mean'), OA_sd=('OA', 'std'), AA=('AA', 'mean'), AA_sd=('AA', 'std'),
                            Kappa=('Kappa', 'mean'), F1=('F1', 'mean'), seeds=('seed', 'nunique')).reset_index()
t = t.sort_values('OA')
save(pd.DataFrame({'Model': t['model'], 'OA (%)': [fmt(a, b) for a, b in zip(t.OA, t.OA_sd)],
                   'AA (%)': [fmt(a, b) for a, b in zip(t.AA, t.AA_sd)], 'Kappa': t.Kappa.round(3),
                   'Macro F1': t.F1.round(3)}), 'table_local_benchmark',
     'Local soil benchmark, held-out physical samples (5 seeds; LightGBM and SVM are deterministic).')

# ---- Table: Indian Pines few-shot (R1), plus each model vs LightGBM
models = ['LightGBM', 'ViT', 'SpectralFormer', 'ConvStemViT', 'HybridSN']
rows = []
for s, g in r1.groupby('split'):
    row = {'Split': f'{s * 100:g}%'}
    for m in models:
        v = g[g.model == m]
        row[m] = fmt(v.OA.mean(), v.OA.std())
    rows.append(row)
save(pd.DataFrame(rows), 'table_ip_fewshot', 'Indian Pines overall accuracy (%) by training split, mean ± SD over 5 seeds.')

tests = []
for m in models[1:]:
    for s, g in r1.groupby('split'):
        a = g[g.model == 'LightGBM'].sort_values('seed').OA
        b = g[g.model == m].sort_values('seed').OA
        tests.append(dict(model=m, split=f'{s * 100:g}%', **paired(a, b)))
save(bh(pd.DataFrame(tests)).round(4), 'stats_ip_vs_lightgbm',
     'Indian Pines: model minus LightGBM OA (pp), paired over seeds; BH-FDR within this table.')

# ---- Transfer tables (R3: Local->IP, R4: IP->Local)
tests = []
for name, df in [('Local -> Indian Pines', r3), ('Indian Pines -> Local', r4)]:
    for m in ['HybridSN', 'SpectralFormer', 'ViT']:
        for s, g in df[df.model == m].groupby('split'):
            a = g[g.init == 'scratch'].sort_values('seed').OA.to_numpy()
            b = g[g.init == 'transfer'].sort_values('seed').OA.to_numpy()
            lg = df[(df.model == 'LightGBM') & (df.split == s)].OA
            tests.append(dict(direction=name, model=m, split=f'{s * 100:g}%', lightgbm=lg.mean(),
                              scratch=a.mean(), transfer=b.mean(), **paired(a, b)))
save(bh(pd.DataFrame(tests)).round(4), 'stats_transfer',
     'Transfer minus scratch OA (pp), paired over seeds; BH-FDR over both directions.')

# ---- Linear probe (R5)
tests = []
for m in ['HybridSN', 'SpectralFormer', 'SpatialViT']:
    for s, g in r5[r5.model == m].groupby('split'):
        a = g[g.init == 'scratch'].sort_values('seed').OA
        b = g[g.init == 'transfer'].sort_values('seed').OA
        tests.append(dict(model=m, split=f'{s * 100:g}%', scratch=a.mean(), transfer=b.mean(), **paired(a, b)))
save(bh(pd.DataFrame(tests)).round(4), 'stats_linear_probe',
     'Linear probe on Indian Pines after Local-soil pretraining: transfer minus random-init features (pp).')

# ---- Timing (R6)
save(r6[['model', 'platform', 'params', 'ms_per_patch']].assign(ms_per_patch=lambda d: d.ms_per_patch.round(2)),
     'table_timing', 'Inference time per 15×15 patch, batch size 1, median of 3 passes over 1000 patches.')

# ---- Crossover points vs LightGBM
cross = []
for label, df, m, init in [('Indian Pines, scratch', r1, 'HybridSN', None), ('Indian Pines, scratch', r1, 'ConvStemViT', None),
                           ('Indian Pines, scratch (R3)', r3, 'HybridSN', 'scratch'), ('Indian Pines, transfer (R3)', r3, 'HybridSN', 'transfer'),
                           ('Local, scratch', r4, 'HybridSN', 'scratch'), ('Local, transfer', r4, 'HybridSN', 'transfer'),
                           ('Local, scratch', r4, 'SpectralFormer', 'scratch'), ('Local, transfer', r4, 'SpectralFormer', 'transfer'),
                           ('Local, scratch', r4, 'ViT', 'scratch'), ('Local, transfer', r4, 'ViT', 'transfer')]:
    sp, ref = grid(df, 'LightGBM', 'none' if init else None)
    _, oth = grid(df, m, init)
    cross.append(dict(setting=label, model=m, **crossover_ci(sp, ref, oth)))
save(pd.DataFrame(cross).round(3), 'crossover',
     'Training share (%) where the model overtakes LightGBM; 95% CI from 5000 seed resamples. '
     'below_first_split / never = share of resamples where the model already wins at 1% / never wins.')

# ---- Figures
def curves(ax, df, series, title):
    for label, m, init, style in series:
        sp, oa = grid(df, m, init)
        ax.errorbar(sp, oa.mean(1), yerr=oa.std(1, ddof=1), fmt=style, capsize=3, label=label)
    ax.set_xscale('log'); ax.set_xticks([1, 5, 10, 20]); ax.set_xticklabels(['1', '5', '10', '20'])
    ax.set_xlabel('Labelled training data (%)'); ax.set_ylabel('Overall accuracy (%)'); ax.set_title(title)
    ax.grid(alpha=.3); ax.legend(fontsize=8)


fig, ax = plt.subplots(figsize=(6, 4.2))
curves(ax, r1, [('LightGBM', 'LightGBM', None, 'k-o'), ('HybridSN', 'HybridSN', None, '-s'), ('ConvStem ViT', 'ConvStemViT', None, '-^'),
                ('ViT (spectral)', 'ViT', None, '-v'), ('SpectralFormer', 'SpectralFormer', None, '-d')],
       'Indian Pines: accuracy vs. annotation budget')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_ip_fewshot.png'), dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, m in zip(axes, ['HybridSN', 'SpectralFormer', 'ViT']):
    curves(ax, r4, [('LightGBM', 'LightGBM', 'none', 'k-o'), (f'{m} scratch', m, 'scratch', '--s'), (f'{m} transfer', m, 'transfer', '-s')],
           f'Local soil: {m}')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_local_transfer.png'), dpi=300); plt.close(fig)

tr = pd.read_csv(os.path.join(TAB, 'stats_transfer.csv'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (d, g) in zip(axes, tr.groupby('direction', sort=False)):
    splits = list(dict.fromkeys(g.split))
    for k, m in enumerate(['HybridSN', 'SpectralFormer', 'ViT']):
        v = g[g.model == m]
        x = np.arange(len(splits)) + (k - 1) * 0.25
        ax.bar(x, v['diff'], 0.25, yerr=[v['diff'] - v.ci_low, v.ci_high - v['diff']], capsize=3, label=m)
    ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(len(splits))); ax.set_xticklabels(splits)
    ax.set_title(d); ax.set_xlabel('Labelled target data'); ax.grid(axis='y', alpha=.3)
axes[0].set_ylabel('Transfer gain, OA (pp, 95% CI)'); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_transfer_gain.png'), dpi=300); plt.close(fig)

# ---- SHAP vs wavelength (R7), if that run has been downloaded
SHAP = os.path.join(RES, 'session-d')
if os.path.exists(os.path.join(SHAP, 'shap_local.csv')):
    # Pika NIR-320: 168 evenly spaced bands, 900-1700 nm.
    # Indian Pines: 220 AVIRIS bands over ~400-2500 nm; 'corrected' drops 1-based bands 104-108, 150-163 and 220.
    kept = [k for k in range(220) if not (103 <= k <= 107 or 149 <= k <= 162 or k == 219)]
    assert len(kept) == 200
    wl = {'local': 900 + np.arange(168) * 800 / 167, 'indianpines_5pct': 400 + np.array(kept) * 2100 / 219}
    titles = {'local': 'Local soil (LightGBM, held-out samples)', 'indianpines_5pct': 'Indian Pines (LightGBM, 5% training)'}
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5))
    rows = []
    for ax, name in zip(axes, ['local', 'indianpines_5pct']):
        s = pd.read_csv(os.path.join(SHAP, f'shap_{name}.csv'))
        x = wl[name]
        gap = np.diff(x) > 1.5 * np.median(np.diff(x))  # removed water bands: break the line, don't bridge it
        xs, ys = np.insert(x, np.nonzero(gap)[0] + 1, np.nan), np.insert(s.total.to_numpy(), np.nonzero(gap)[0] + 1, np.nan)
        ax.plot(xs, ys, lw=1.2, color='k')
        peaks = []  # top-3 bands at least 30 nm apart, so one peak is labelled once
        for b in s.total.sort_values(ascending=False).index:
            if all(abs(x[b] - x[p]) >= 30 for p in peaks):
                peaks.append(b)
            if len(peaks) == 3:
                break
        for b in peaks:
            ax.annotate(f'{x[b]:.0f} nm', (x[b], s.total[b]), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
            rows.append(dict(dataset=name, band=int(b), wavelength_nm=round(float(x[b])), mean_abs_shap=float(s.total[b])))
        ax.set_title(titles[name], fontsize=10); ax.set_ylabel('mean |SHAP|'); ax.grid(alpha=.3)
    axes[1].set_xlabel('Wavelength (nm, approximate band centre)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_shap.png'), dpi=300); plt.close(fig)
    save(pd.DataFrame(rows), 'shap_top_bands', 'Top-3 LightGBM bands by mean |SHAP| (summed over classes).')
print('\nwrote', sorted(os.listdir(TAB)), sorted(os.listdir(FIG)))
