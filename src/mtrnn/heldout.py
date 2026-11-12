"""Held-out TI / TD analysis of every task pair, the paper's Figure 4e/h and Figure 5 measures with task pairs in place of sessions.

For every pair and network (primary window, readout features, network's own choice as label):
    halves     the paper's interleaved split (make_halves, seed 0), both directions: label half -> decode half
    labels     snCCGP and within-task accuracy of every unit on the label half; group = active and within >= 0.55 in either task;
               TI / TD = top / bottom 20 % of the label-half index, pooled over all pairs and networks of the condition (one cut
               per direction, the analogue of pooling the sessions of a species)
    decoding   on the decode half, the paper's cd_decode on per-task z-scored activity:
                 full group within / across; % TI and % TD of the group
                 size-matched TI-only / TD-only / random-group subsets (n = min(#TI, #TD) >= 6; 5 draws)
                 removal of the TI (TD) units vs removal of the same number of random group units (10 draws)
    Values are averaged over the two directions.

    python -m mtrnn.heldout main_relu [--min_perf 0.9]   -> results/readout/<run>/heldout_pairs[_p90].parquet

    --min_perf: a network is left out of every pair containing a task it performs below this (default readout.MIN_TASK_PERF = 0.80;
    the paper figures use 0.90)
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from . import readout as Q

from .lib import index as sn
from .lib.population import make_halves, zctx, cd_decode

PROP = 0.20
MIN_N = 6


def _label_pass(Fp, a, b, keep):
    """Per network and direction: label-half index, group mask."""
    w = Q.primary_window(a, b); out = []
    for k in keep:
        X, y, ctx = Q.session(Fp, a, b, w, k)
        A, B = make_halves(y, ctx, seed=0)
        active = (X[ctx == 1].mean(0) > 1e-4) | (X[ctx == 2].mean(0) > 1e-4)
        for d, (L, Dh) in enumerate(((A, B), (B, A))):
            r = sn._summarise(sn._decode_matrix(X[L], y[L], ctx[L], method='fast'))
            g = active & np.isfinite(r['snCCGP']) & (np.fmax(r['sn_within_1'], r['sn_within_2']) >= Q.GROUP_MIN)
            out.append(dict(k=k, dir=d, idx=np.asarray(r['snCCGP']), group=g))
    return out


def _acc(Z, y, ctx, cols, seed=42):
    r = cd_decode(Z[:, cols], y, ctx, seed=seed) if len(cols) >= 2 else None
    return (np.nan, np.nan) if r is None else (r['within_cd'], r['across_cd'])


def _decode_pass(Fp, a, b, labels, cuts, seeds):
    w = Q.primary_window(a, b); rows = []
    for lab in labels:
        k, d = lab['k'], lab['dir']
        X, y, ctx = Q.session(Fp, a, b, w, k)
        A, B = make_halves(y, ctx, seed=0); Dh = B if d == 0 else A
        Xd, yd, cd = X[Dh], y[Dh], ctx[Dh]; Z = zctx(Xd, cd)
        lo, hi = cuts[d]; g = np.flatnonzero(lab['group']); v = lab['idx'][g]
        ti, td = g[v >= hi], g[v <= lo]
        rng = np.random.default_rng([int(seeds[k]), d, 7])
        fw, fa = _acc(Z, yd, cd, g)
        row = dict(task_a=a, task_b=b, seed=int(seeds[k]), dir=d, n_group=len(g), n_ti=len(ti), n_td=len(td),
                   pct_TI=100 * len(ti) / max(len(g), 1), pct_TD=100 * len(td) / max(len(g), 1), within=fw, across=fa)
        n = min(len(ti), len(td))
        if n >= MIN_N:
            sub = lambda pool: np.nanmean([_acc(Z, yd, cd, rng.choice(pool, n, replace=False), seed=42 + i)[1] for i in range(5)])
            row.update(TI_across=sub(ti), TD_across=sub(td), random_across=sub(g), n_subset=n)
        for cls, rem in (('TI', ti), ('TD', td)):
            if 0 < len(rem) and len(g) - len(rem) >= MIN_N:
                cut = _acc(Z, yd, cd, np.setdiff1d(g, rem))[1]
                rnd = np.nanmean([_acc(Z, yd, cd, np.setdiff1d(g, rng.choice(g, len(rem), replace=False)), seed=50 + i)[1] for i in range(10)])
                row[f'rem_{cls}'] = cut - rnd
        rows.append(row)
    return rows


def run(run_name, n_jobs=16, min_perf=None):
    from joblib import Parallel, delayed
    F = Q.load_features(run_name); names = list(F['names']); seeds = list(F['seeds'])
    pairs = list(itertools.combinations(names, 2))
    sub = lambda a, b: {key: F[key] for key in F if key.split('|')[0] in (a, b)}
    if min_perf is None:
        keep = {p: Q.ok_seeds(run_name, p[0], p[1], seeds) for p in pairs}
    else:
        tp = Q.task_perf(run_name)
        keep = {p: [k for k, s in enumerate(seeds) if tp[(int(s), p[0])] >= min_perf and tp[(int(s), p[1])] >= min_perf] for p in pairs}
    labels = Parallel(n_jobs=n_jobs)(delayed(_label_pass)(sub(a, b), a, b, keep[(a, b)]) for a, b in pairs)
    labels = [L if L else [] for L in labels]
    cuts = {}
    for d in (0, 1):
        v = np.concatenate([l['idx'][l['group']] for L in labels for l in L if l['dir'] == d])
        cuts[d] = (np.quantile(v, PROP), np.quantile(v, 1 - PROP))
    rows = Parallel(n_jobs=n_jobs)(delayed(_decode_pass)(sub(a, b), a, b, L, cuts, seeds) for (a, b), L in zip(pairs, labels) if L)
    T = pd.DataFrame([r for R in rows for r in R])
    T = T.groupby(['task_a', 'task_b', 'seed']).mean(numeric_only=True).drop(columns='dir').reset_index()
    T.attrs['cuts'] = cuts
    T.to_parquet(os.path.join(Q.OUT, run_name, 'heldout_pairs.parquet' if min_perf is None else f'heldout_pairs_p{int(round(min_perf * 100))}.parquet'))
    print(f'{run_name}: cuts {cuts}; {len(T)} rows', flush=True)
    return T


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--min_perf', type=float, default=None)
    a = ap.parse_args()
    run(a.run, min_perf=a.min_perf)
