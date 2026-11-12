"""Statistics logger for Supplementary Table 1.

Each figure notebook creates one table in its setup cell and logs the arrays it plots, at the moment it plots them:

    from paper_stats import StatsTable
    ST = StatsTable('figure_s1')          # notebook stem; sets the sheet name and the output CSV

and ends with `ST.save()`, which writes `analysis/tables/<file>.csv` (full float precision). Rows are keyed on
(panel, dataset, measure, group / comparison): logging the same key again overwrites the row in place, so re-running a cell
never duplicates it. `tables/build_supp_table.py` assembles the CSVs into the workbook.
"""
import json, os, re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(HERE, 'tables')
COLUMNS = ['Panel', 'Dataset', 'Measure', 'Group / comparison', 'n', 'Unit of n', 'Mean', 's.e.m.', 'Test', 'Statistic', 'P',
           'Notebook location']
TEST_NAME = {'wilcoxon': 'Two-sided Wilcoxon signed-rank', 'ttest_rel': 'Two-sided paired t-test',
             'ttest_1samp': 'Two-sided one-sample t-test', 'pearson': 'Pearson correlation', 'spearman': 'Spearman correlation'}


def sheet_of(stem):
    """figure_3 -> ('Fig. 3', 'fig3.csv'); figure_s3 -> ('ED Fig. 3', 'ed_fig3.csv'); extra_noise_correlations ->
    ('extra_noise_correlations', 'extra_noise_correlations.csv') (extra figures: own table, not in Supplementary Table 1)."""
    if re.fullmatch(r'extra_\w+', stem):
        return stem, stem + '.csv'
    m = re.fullmatch(r'figure_(s?)(\d+)', stem)
    if not m:
        raise ValueError(f'not a figure notebook stem: {stem}')
    return (f'ED Fig. {m[2]}', f'ed_fig{m[2]}.csv') if m[1] else (f'Fig. {m[2]}', f'fig{m[2]}.csv')


def _finite(x):
    x = np.asarray(x, float).ravel()
    return x[np.isfinite(x)]


def _paired_finite(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def _check(p, expect_p):
    # guard: the logged test must be the one the notebook already ran on the same values
    if expect_p is not None and not np.isclose(p, expect_p, rtol=1e-9, atol=0):
        raise AssertionError(f'logged P {p!r} differs from the notebook P {expect_p!r}')


def panel_locations(nb_path):
    """Panel letter -> (code-cell index, heading) from the notebook's markdown headings such as '## Fig S1 a · ...',
    '## Fig 2 c, g · ...', '## Fig 5 d–g · ...' or figure_s8's '## a · ...'. The first heading naming a letter wins."""
    nb = json.load(open(nb_path))
    cells, out = nb['cells'], {}
    for i, c in enumerate(cells):
        if c['cell_type'] != 'markdown':
            continue
        for line in ''.join(c['source']).splitlines():
            m = re.match(r'#{2,3}\s*(?:Fig\.?\s*\S+\s+)?(.+?)\s*·', line)
            if not m:
                continue
            spec = re.sub(r'\([^)]*\)', '', m[1])
            letters = []
            for tok in spec.split(','):
                tok = tok.strip()
                r = re.fullmatch(r'([a-z])\s*[–-]\s*([a-z])', tok)
                if r:
                    letters += [chr(k) for k in range(ord(r[1]), ord(r[2]) + 1)]
                elif re.fullmatch(r'[a-z]', tok):
                    letters.append(tok)
                else:
                    letters = []
                    break
            code = next((j for j in range(i + 1, len(cells)) if cells[j]['cell_type'] == 'code'), None)
            for L in letters:
                if code is not None and L not in out:
                    out[L] = (code, line.lstrip('#').strip())
    return out


class StatsTable:
    def __init__(self, stem):
        self.stem = stem
        self.sheet, self.fname = sheet_of(stem)
        self.rows = {}                                  # key -> row dict; dict order = first time each key was logged

    def _log(self, panel, dataset, measure, label, **cols):
        row = dict.fromkeys(COLUMNS)
        row.update({'Panel': panel, 'Dataset': dataset, 'Measure': measure, 'Group / comparison': label}, **cols)
        self.rows[(panel, dataset, measure, label)] = row

    # ---- group rows -------------------------------------------------------------------------------------------------
    def group(self, panel, values, *, dataset, measure, group, unit):
        v = _finite(values); n = len(v)
        mean = float(v.mean()) if n else np.nan
        sem = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        self._log(panel, dataset, measure, group, **{'n': n, 'Unit of n': unit, 'Mean': mean, 's.e.m.': sem})
        return mean, sem, n

    def value(self, panel, value, *, dataset, measure, group, n, unit):
        self._log(panel, dataset, measure, group, **{'n': n, 'Unit of n': unit, 'Mean': float(value)})
        return value

    def schematic(self, panel, what='schematic'):
        self._log(panel, None, what, None)

    # ---- comparison rows --------------------------------------------------------------------------------------------
    def paired(self, panel, a, b, *, test, dataset, measure, comparison, unit, expect_p=None):
        a, b = _paired_finite(a, b)
        res = {'wilcoxon': stats.wilcoxon, 'ttest_rel': stats.ttest_rel}[test](a, b)
        _check(res.pvalue, expect_p)
        self._log(panel, dataset, measure, comparison, **{'n': len(a), 'Unit of n': unit, 'Test': TEST_NAME[test],
                                                          'P': float(res.pvalue)})
        return res

    def one_sample(self, panel, a, *, test, popmean=0, dataset, measure, comparison, unit, expect_p=None):
        a = _finite(a)
        res = stats.wilcoxon(a - popmean) if test == 'wilcoxon' else stats.ttest_1samp(a, popmean)
        _check(res.pvalue, expect_p)
        name = 'Two-sided Wilcoxon signed-rank' if test == 'wilcoxon' else TEST_NAME['ttest_1samp']
        self._log(panel, dataset, measure, comparison, **{'n': len(a), 'Unit of n': unit, 'Test': f'{name} vs {popmean:g}',
                                                          'P': float(res.pvalue)})
        return res

    def corr(self, panel, x, y, *, method, dataset, measure, comparison, unit, slope_per=None, expect_p=None):
        x, y = _paired_finite(x, y)
        if method == 'pearson':
            res = stats.pearsonr(x, y); r = res.statistic
            stat = f'r = {r:+.2f}'
            if slope_per is not None:
                lr = stats.linregress(x, y)
                stat = f'slope = {slope_per * lr.slope:+.3f} per {slope_per:g}%, r = {r:+.2f}'
        else:
            res = stats.spearmanr(x, y); stat = f'ρ = {res.statistic:+.2f}'
        _check(res.pvalue, expect_p)
        self._log(panel, dataset, measure, comparison, **{'n': len(x), 'Unit of n': unit, 'Test': TEST_NAME[method],
                                                          'Statistic': stat, 'P': float(res.pvalue)})
        return res

    def test(self, panel, *, test, p, n, unit, dataset, measure, comparison, statistic=None):
        """Escape hatch for a test the notebook already ran (MixedLM, permutation null, per-session slopes, cached)."""
        self._log(panel, dataset, measure, comparison, **{'n': n, 'Unit of n': unit, 'Test': test, 'Statistic': statistic,
                                                          'P': None if p is None else float(p)})

    # ---- output -----------------------------------------------------------------------------------------------------
    def save(self):
        loc = panel_locations(os.path.join(HERE, f'{self.stem}.ipynb'))
        df = pd.DataFrame(list(self.rows.values()), columns=COLUMNS)
        unmatched = sorted({p for p in df.Panel if p not in loc})
        df['Notebook location'] = [f'{self.stem}.ipynb, cell {loc[p][0]} ({loc[p][1]})' if p in loc else f'{self.stem}.ipynb'
                                   for p in df.Panel]
        df['n'] = df['n'].astype('Int64')
        os.makedirs(TABLE_DIR, exist_ok=True)
        path = os.path.join(TABLE_DIR, self.fname)
        df.to_csv(path, index=False, encoding='utf-8', float_format='%.17g')
        print(f'{self.sheet}: {len(df)} rows -> {path}')
        print('  rows per panel: ' + ', '.join(f'{p} {k}' for p, k in df.groupby('Panel', sort=True).size().items()))
        if unmatched:
            print('  no heading found for panel(s): ' + ', '.join(unmatched))
        return df
