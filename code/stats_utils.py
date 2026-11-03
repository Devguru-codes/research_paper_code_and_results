"""Statistics and table helpers for analysis.py.

Paired t-test over matched seeds with 95% CI and Cohen's d_z; Benjamini-Hochberg FDR within a family.
"""
import os
import numpy as np, pandas as pd
from scipy import stats


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
