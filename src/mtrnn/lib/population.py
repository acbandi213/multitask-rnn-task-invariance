"""Population decoding and TI / TD labels (the functions used for the recordings, copied unchanged).

    make_halves, zctx, cd_decode, _mean_res     trial halves, per-task z-score, coding-direction decoder (within / across task)
    unit_index, pooled_cuts, apply_cuts         per-unit index on every trial split; TI / TD tails pooled over networks
    decode_job                                  every decoding row of one network: size-matched TI-only / TD-only / random subsets,
                                                removal of TI / TD vs the same number of random units, the four-condition decoder

A "session" is a dict with X (trials x units), y (choice 0 / 1), ctx (task 1 / 2) and mean_rate (units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import index as sn
from . import decoders as D

MIN_UNITS, N_DRAWS_CLS, N_DRAWS_FIXED, MIN_PER = 6, 5, 10, 8


PROP_TI, PROP_TD = 0.25, 0.25   # TI = top quartile, TD = bottom quartile of the index within the neuron group, both species


def make_halves(y, ctx, seed=7, mode='interleaved'):
    """Two disjoint trial halves A, B (boolean masks).
    interleaved: random, balanced within every (context x label) cell — the split used everywhere unless stated
    blocked:     within each context, the first half of that context's trials in time -> A, the second half -> B,
                 so slow drift, arousal and satiety differ between the halves instead of being shared"""
    A = np.zeros(len(y), bool)
    if mode == 'interleaved':
        rng = np.random.default_rng(seed)
        for c in (1, 2):
            for k in (0, 1):
                idx = np.flatnonzero((ctx == c) & (y == k)); rng.shuffle(idx); A[idx[: len(idx) // 2]] = True
    elif mode == 'blocked':
        for c in (1, 2):
            idx = np.flatnonzero(ctx == c); A[idx[: len(idx) // 2]] = True
    else: raise ValueError(mode)
    return A, ~A


def zctx(X, ctx):
    # z-score every neuron within each context over the trials given (label-agnostic: mean and SD over all of that context's trials)
    Z = np.array(X, float, copy=True)
    for c in (1, 2):
        m = ctx == c; mu = Z[m].mean(0); sd = Z[m].std(0); sd[sd == 0] = 1.0; Z[m] = (Z[m] - mu) / sd
    return Z


def cd_decode(X, y, ctx, min_per=MIN_PER, seed=42):
    """Coding-direction decoder on one neuron set (the Figure 3 recipe: difference of class means, unit norm,
    midpoint threshold on training trials). Returns within (5-fold, both contexts averaged), across (train one
    context / test the other, both directions averaged) and the cosine between the two contexts' CDs.
    None if a class is too rare in a context."""
    Xs, ys = {}, {}
    for c in (1, 2):
        m = ctx == c; yy = y[m]
        if len(np.unique(yy)) < 2 or np.bincount(yy).min() < min_per: return None
        Xs[c], ys[c] = X[m], yy
    def fit(Xt, yt):
        cd = Xt[yt == 1].mean(0) - Xt[yt == 0].mean(0); n = np.linalg.norm(cd)
        if n == 0: return None
        cd = cd / n; pr = Xt @ cd; thr = .5 * (pr[yt == 1].mean() + pr[yt == 0].mean()); return cd, thr
    def score(model, Xe, ye):
        cd, thr = model; return float(((Xe @ cd > thr).astype(int) == ye).mean())
    within = []
    for c in (1, 2):
        acc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs[c], ys[c]):
            mdl = fit(Xs[c][tr], ys[c][tr])
            if mdl is not None: acc.append(score(mdl, Xs[c][te], ys[c][te]))
        if not acc: return None
        within.append(np.mean(acc))
    m1, m2 = fit(Xs[1], ys[1]), fit(Xs[2], ys[2])
    if m1 is None or m2 is None: return None
    return dict(within_cd=float(np.mean(within)), across_cd=.5 * (score(m1, Xs[2], ys[2]) + score(m2, Xs[1], ys[1])), cos_cd=float(m1[0] @ m2[0]))


def _mean_res(accs):
    accs = [a for a in accs if a is not None]
    return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]} if accs else None


SPLITS = [('interleaved', s) for s in range(10)]        # the paper's 10 interleaved splits (its blocked split tests drift; a network has none)


DIRS = (('A', 'B'), ('B', 'A'))                        # (label half, decode half)


GROUP_MIN_WITHIN = 0.55


N_DRAWS_XOR = 50


def _dprime(X, y, m):
    """Signed d' per unit on trials m: (mean R - mean L) / pooled SD."""
    a, b = X[m & (y == 1)], X[m & (y == 0)]
    sd = np.sqrt(0.5 * (a.var(0) + b.var(0))); sd[sd == 0] = np.nan
    return (a.mean(0) - b.mean(0)) / sd


def unit_index(sess, pool, splits=SPLITS, pref=None, gmul=None):
    """One row per (split, half, unit): snCCGP, sn_within (both tasks), cross accuracies, signed d' per task, pool,
    group membership (from the whole session: half = 'all')."""
    X, y, ctx = sess['X'], sess['y'], sess['ctx']
    N = X.shape[1]
    rows = []

    def block(mask, mode, seed, half):
        M = sn._decode_matrix(X[mask], y[mask], ctx[mask], method='fast'); r = sn._summarise(M)
        d1, d2 = _dprime(X, y, mask & (ctx == 1)), _dprime(X, y, mask & (ctx == 2))
        return pd.DataFrame(dict(mode=mode, split=seed, half=half, unit=np.arange(N), snCCGP=r['snCCGP'], sn_within=r['sn_within'],
                                 sn_within_1=r['sn_within_1'], sn_within_2=r['sn_within_2'], acc_1_2=M['1_2'], acc_2_1=M['2_1'],
                                 d1=d1, d2=d2, pool=pool, pref=(np.zeros(N, int) if pref is None else pref), gmul=(np.ones(N) if gmul is None else gmul), mean_rate=sess['mean_rate']))

    allrows = block(np.ones(len(y), bool), 'all', -1, 'all')
    active = sess['mean_rate'] > 1e-4
    group = active & np.isfinite(allrows.snCCGP.to_numpy()) & (allrows.sn_within.to_numpy() >= GROUP_MIN_WITHIN)
    rows.append(allrows)
    for mode, seed in splits:
        A, B = make_halves(y, ctx, seed=seed, mode=mode)
        rows.append(block(A, mode, seed, 'A')); rows.append(block(B, mode, seed, 'B'))
    T = pd.concat(rows, ignore_index=True)
    T['group'] = group[T.unit.to_numpy()]
    T['active'] = active[T.unit.to_numpy()]
    return T


def pooled_cuts(index_tables, prop_ti=None, prop_td=None):
    """Tail cuts per (mode, split, half) pooled over the networks of one condition, group units with finite snCCGP only
    (comp_labels' rule). Default = the setup's PROP_TI / PROP_TD (quartiles); the paper figures use 0.20 / 0.20.
    Returns a DataFrame with hi / lo cuts and n."""
    prop_ti = PROP_TI if prop_ti is None else prop_ti
    prop_td = PROP_TD if prop_td is None else prop_td
    I = pd.concat(index_tables, ignore_index=True)
    I = I[I.group & np.isfinite(I.snCCGP)]
    C = I.groupby(['mode', 'split', 'half']).snCCGP.agg(hi=lambda s: s.quantile(1 - prop_ti), lo=lambda s: s.quantile(prop_td), n='size')
    return C.reset_index()


def apply_cuts(I, cuts):
    """Add cls (TI / TD / Int / none) to one network's index table from the condition's pooled cuts."""
    I = I.drop(columns=[c for c in ('hi', 'lo', 'cls') if c in I.columns])
    I = I.merge(cuts[['mode', 'split', 'half', 'hi', 'lo']], on=['mode', 'split', 'half'], how='left')
    ok = I.group & np.isfinite(I.snCCGP)
    I['cls'] = np.where(~ok, 'none', np.where(I.snCCGP >= I.hi, 'TI', np.where(I.snCCGP <= I.lo, 'TD', 'Int')))
    return I


def _sets(rng, g, ti, td, pool_ti, pool_td):
    """{set: (n,) or (draws, n) unit indices}. Three-point subsets, class removals, pool removals, random controls."""
    sets = {}
    allg = np.arange(len(g))
    n = min(len(ti), len(td))
    # 2-D sets carry N_DRAWS_XOR unit draws (one per balanced trial draw of the XOR decoder); cd_decode uses the first
    # N_DRAWS_CLS (subsets) / N_DRAWS_FIXED (random removals) of them, the paper's counts.
    if n >= MIN_UNITS:
        sets['TI'] = np.stack([ti if len(ti) == n else rng.choice(ti, n, replace=False) for _ in range(N_DRAWS_XOR)])
        sets['TD'] = np.stack([td if len(td) == n else rng.choice(td, n, replace=False) for _ in range(N_DRAWS_XOR)])
        sets['random'] = np.stack([rng.choice(g, n, replace=False) for _ in range(N_DRAWS_XOR)])
    if len(g) >= 2 * MIN_UNITS:
        sets['full'] = g
        for cls, rem in (('TI', ti), ('TD', td), ('poolTI', pool_ti), ('poolTD', pool_td)):
            k = len(rem)
            if k < 1 or len(g) - k < MIN_UNITS:
                continue
            sets[f'-{cls}'] = np.setdiff1d(g, rem)
            sets[f'-r{cls}'] = np.stack([np.setdiff1d(g, rng.choice(g, k, replace=False)) for _ in range(N_DRAWS_XOR)])
    return sets


def decode_job(sess, I, net_seed, splits=SPLITS):
    """All decoding rows of one network: for every split and direction, every unit set, the paper's cd_decode
    (within / across / cosine, per-task z) and the balanced four-condition decoder (choice / task / XOR / CCGP, pooled z)."""
    X, y, ctx = sess['X'], sess['y'], sess['ctx']
    pool = I[I.half == 'all'].set_index('unit').pool
    group = np.flatnonzero(I[I.half == 'all'].set_index('unit').group.to_numpy())
    g = group
    pool_ti = g[pool.loc[g].to_numpy() == 1]; pool_td = g[pool.loc[g].to_numpy() == 0]
    rows = []
    for mode, seed in splits:
        A, B = make_halves(y, ctx, seed=seed, mode=mode)
        for di, (lh, dh) in enumerate(DIRS):
            lab = I[(I['mode'] == mode) & (I.split == seed) & (I.half == lh)].set_index('unit').cls
            ti = np.array([u for u in g if lab.get(u) == 'TI'], int); td = np.array([u for u in g if lab.get(u) == 'TD'], int)
            m = A if dh == 'A' else B
            Xd, yd, cd = X[m], y[m], ctx[m]
            cond = D.conditions(yd, cd); mmin = int(np.bincount(cond, minlength=4).min())
            if mmin < MIN_PER or len(g) < 2 * MIN_UNITS:
                continue
            Zc, Zp = zctx(Xd, cd), D.zpool(Xd)
            rng = np.random.default_rng([20261007, net_seed, seed, di])
            sets = _sets(rng, g, ti, td, pool_ti, pool_td)
            for name, cols in sets.items():
                cl = np.atleast_2d(cols)
                k = 1 if cl.shape[0] == 1 else (N_DRAWS_FIXED if name.startswith('-r') else N_DRAWS_CLS)
                r = _mean_res([cd_decode(Zc[:, cl[i]], yd, cd, seed=42 + i) for i in range(k)]) or {}
                idx = D.balanced_index(cond, mmin, rng, N_DRAWS_XOR)
                q = {k: float(np.nanmean(v)) for k, v in D.geometry(D.gather(Zp, idx, cols)).items()}
                rows.append(dict(mode=mode, split=seed, dir=f'{lh}>{dh}', set=name, n_units=int(cl.shape[1]), n_group=len(g),
                                 n_ti=len(ti), n_td=len(td), m_trials=mmin, within_cd=r.get('within_cd', np.nan),
                                 across_cd=r.get('across_cd', np.nan), cos_cd=r.get('cos_cd', np.nan), **q))
    return pd.DataFrame(rows)
