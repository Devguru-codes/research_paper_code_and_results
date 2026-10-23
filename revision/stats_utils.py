"""Statistics and table helpers shared by analysis_v1.py and analysis.py.

Paired t-test over matched seeds with 95% CI and Cohen's d_z; Benjamini-Hochberg FDR within a family;
crossover = budget after which a model stays above a reference, with a seed-resampling 95% interval.
"""
import os
import numpy as np, pandas as pd
from scipy import stats

BOOT = 5000
rng = np.random.default_rng(0)


def paired(a, b):
    """b - a over matched seeds: mean, 95% CI, Cohen's d_z, paired t-test p."""
    d = np.asarray(b, float) - np.asarray(a, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = np.nan if n < 2 else stats.ttest_rel(b, a).pvalue if sd > 0 else (0.0 if m else 1.0)
    return dict(n=n, diff=m, ci_low=m - h, ci_high=m + h, d_z=m / sd if sd > 0 else np.nan, p=p)


def bh(df):
    """Benjamini-Hochberg over every row of df: one table = one family of tests."""
    df['p_fdr'] = stats.false_discovery_control(df['p'].to_numpy(), method='bh')
    return df


def crossover(splits, ref, other):
    """Budget after which mean(other) stays above mean(ref), linearly interpolated.
    splits[0] if `other` wins at every budget; nan if it is not ahead at the largest budget."""
    d = other - ref
    if d[-1] <= 0:
        return np.nan
    behind = np.nonzero(d <= 0)[0]
    if not len(behind):
        return splits[0]
    i = behind[-1] + 1  # first budget of the final winning run
    return splits[i - 1] + (splits[i] - splits[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1])


def crossover_ci(splits, ref, other):
    """ref/other: arrays (budgets x seeds), seeds paired. Resample seeds jointly per budget."""
    point = crossover(splits, ref.mean(1), other.mean(1))
    n = ref.shape[1]
    boots = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=(len(splits), n))
        boots.append(crossover(splits, np.take_along_axis(ref, idx, 1).mean(1), np.take_along_axis(other, idx, 1).mean(1)))
    boots = np.array(boots)
    ok = boots[~np.isnan(boots)]
    return dict(crossover=point, ci_low=np.percentile(ok, 2.5) if len(ok) else np.nan,
                ci_high=np.percentile(ok, 97.5) if len(ok) else np.nan,
                below_first=np.mean(boots == splits[0]), never=np.mean(np.isnan(boots)))


def grid(df, model, init=None, x='split', scale=100):
    """OA as (budgets x seeds): rows sorted by budget column `x` (times `scale`), columns by seed."""
    sub = df[df['model'] == model] if init is None else df[(df['model'] == model) & (df['init'] == init)]
    t = sub.pivot_table(index=x, columns='seed', values='OA')
    return t.index.to_numpy() * scale, t.to_numpy()


def fmt(m, s):
    return f'{m:.2f} ± {s:.2f}'


def saver(tab_dir):
    os.makedirs(tab_dir, exist_ok=True)

    def save(df, name, caption):
        df.to_csv(os.path.join(tab_dir, f'{name}.csv'), index=False)
        with open(os.path.join(tab_dir, f'{name}.tex'), 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, caption=caption, label=f'tab:{name}', escape=True,
                                float_format=lambda v: f'{v:.3g}').replace('±', r'$\pm$'))
        print(f'\n== {name}\n{df.to_string(index=False)}')
    return save
