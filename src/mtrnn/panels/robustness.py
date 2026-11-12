"""Extended Data Figure 7: robustness of the multitask-RNN results. One function per panel; each writes
results/panels/<name>.pkl, which analysis/figure_s7.ipynb reads.
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd

from .. import readout as Q
from .. import heldout as PH
from ..lib import index as sn
from ..lib import decoders as DEC
from ..lib.population import make_halves, zctx, cd_decode
from ..paths import MODELS, PERFORMANCE, MEASURED, PANELS, REFERENCE, pick_device
from .common import (ANALOG, CACHE, MIN_N, MIN_PER, MIN_PERF, N_JOBS, RUN, TYPES, _simulate, feats, heldout_cuts, heldout_labels, heldout_table, load, ok_fn, pairs_of, role, save, sub_feats, unit_table, J_BIN, J_PRE, J_POST, J_TRIALS, RUN_SP, PAPER_WINDOW)


# ═══ a · task performance ═══════════════════════════════════════════════════════════════════════════════════════════════
def panel_performance():
    rows = []
    for act, run in (('ReLU', RUN), ('Softplus', RUN_SP)):
        d = pd.read_csv(os.path.join(PERFORMANCE, run, 'perf.csv')); d['act'] = act; rows.append(d)
    cfg = json.load(open(os.path.join(MODELS, RUN, 'config.json')))
    perf = pd.concat(rows, ignore_index=True); drop = {}
    for act, run in (('ReLU', RUN), ('Softplus', RUN_SP)):
        F = feats(run); names = list(F['names']); seeds = list(F['seeds']); n_all = len(pairs_of(names)) * len(seeds)
        kept = {thr: sum(ok_fn(run, thr)(s, a, b) for a, b in pairs_of(names) for s in seeds) for thr in (Q.MIN_TASK_PERF, MIN_PERF)}
        drop[act] = dict(combos=perf[(perf.act == act) & (perf.perf < MIN_PERF)][['seed', 'task', 'perf']].to_dict('records'),
                         pair_rows=n_all, kept_090=kept[MIN_PERF], kept_080=kept[Q.MIN_TASK_PERF])
    save('performance', dict(perf=perf, criterion=cfg['criterion'], min_task_perf=MIN_PERF, dropped=drop))


# ═══ b · split-half reliability of the index, with a label-shuffle null ═════════════════════════════════════════════════
def _b_keys():
    U, P, _ = unit_table(RUN); seeds = list(feats()['seeds'])
    keys = []
    for (a, b, s), g in U.groupby(['task_a', 'task_b', 'seed']):
        keys.append((a, b, seeds.index(int(s)), int(s), g.unit.to_numpy(), g.type.iloc[0], g.within_1.to_numpy(), g.cls_ti.to_numpy()))
    return keys


def _sign(X, y, ctx, c):
    m = ctx == c
    return np.sign(X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0))


def _b_job(rep, keys):
    F = feats(); rng = np.random.default_rng([int(rep) + 1000, 5]); hA, hB = [], []; ex = {k_: [] for k_ in ('s1A', 's1B', 's2A', 's2B')}
    for a, b, k, s, units, t, w1, cls in keys:
        X, y, ctx = Q.session(F, a, b, Q.primary_window(a, b), k)
        if rep >= 0:                                                    # null: choice labels shuffled within each task
            y = y.copy()
            for c in (1, 2):
                m = ctx == c; y[m] = rng.permutation(y[m])
        A, B = make_halves(y, ctx, seed=0)
        for H, out, tag in ((A, hA, 'A'), (B, hB, 'B')):
            Xh = X[H][:, units]
            out.append(np.asarray(sn._summarise(sn._decode_matrix(Xh, y[H], ctx[H], method='fast'))['snCCGP']))
            if rep < 0:                                                 # preference sign in each task, per half
                ex['s1' + tag].append(_sign(Xh, y[H], ctx[H], 1)); ex['s2' + tag].append(_sign(Xh, y[H], ctx[H], 2))
    hA, hB = np.concatenate(hA), np.concatenate(hB)
    if rep < 0:
        return hA, hB, {k_: np.concatenate(v) for k_, v in ex.items()}
    from scipy.stats import spearmanr
    ok = np.isfinite(hA) & np.isfinite(hB)
    return float(spearmanr(hA[ok], hB[ok])[0])


def panel_index_reliability(n_null=100):
    from joblib import Parallel, delayed
    keys = _b_keys(); t0 = time.time()
    hA, hB, ex = _b_job(-1, keys)
    meta = pd.DataFrame([(a, b, s, t, u, w, c) for a, b, k, s, units, t, w1, cls in keys for u, w, c in zip(units, w1, cls)],
                        columns=['task_a', 'task_b', 'seed', 'type', 'unit', 'within_1', 'cls_ti'])
    for k_, v in ex.items():
        meta[k_] = v
    null = Parallel(n_jobs=N_JOBS)(delayed(_b_job)(r, keys) for r in range(n_null))
    save('index_reliability', dict(hA=hA, hB=hB, meta=meta, null_r=np.array(null)))
    print(f'b: {len(hA):,} unit-pairs, {n_null} null reps in {time.time() - t0:.0f}s', flush=True)


def panel_tail_size(props=(0.10, 0.20, 0.30)):
    out = {}
    for prop in props:
        t0 = time.time(); T, cuts = heldout_table(RUN, prop)
        out[prop] = dict(table=T, cuts=cuts)
        print(f'c: tails {prop:.0%} cuts {cuts} in {time.time() - t0:.0f}s', flush=True)
    save('tail_size', out)


# ═══ d / e · Softplus tables ════════════════════════════════════════════════════════════════════════════════════════════
def panel_softplus():
    U, P, cuts = unit_table(RUN_SP)
    H, _ = heldout_table(RUN_SP, .2)
    pct = U.groupby(['seed', 'task_a', 'task_b', 'type']).cls_ti.agg(TI=lambda s: 100 * (s == 'TI').mean(), TD=lambda s: 100 * (s == 'TD').mean()).reset_index()
    save('softplus', dict(pct=pct, heldout=H, cuts=cuts))


# ═══ f · 20-unit decoders within each category ══════════════════════════════════════════════════════════════════════════
def _f_job(Fp, a, b, labels, cuts, seeds, n_sub=50, size=20):
    w = Q.primary_window(a, b); rows = []
    for lab in labels:
        k, d = lab['k'], lab['dir']
        X, y, ctx = Q.session(Fp, a, b, w, k)
        A, B = make_halves(y, ctx, seed=0); Dh = B if d == 0 else A
        Z = zctx(X[Dh], ctx[Dh]); yd, cd = y[Dh], ctx[Dh]
        g = np.flatnonzero(lab['group']); v = lab['idx'][g]; lo, hi = cuts[d]
        if len(g) < size:
            continue
        rng = np.random.default_rng([int(seeds[k]), d, 11])
        acc = []
        for i in range(n_sub):
            r = cd_decode(Z[:, rng.choice(g, size, replace=False)], yd, cd, seed=100 + i)
            if r is not None:
                acc.append(r['across_cd'])
        rows.append(dict(task_a=a, task_b=b, seed=int(seeds[k]), dir=d, n_group=len(g), pct_TI=100 * np.mean(v >= hi),
                         pct_TD=100 * np.mean(v <= lo), across20=float(np.mean(acc)) if acc else np.nan))
    return rows


def panel_pct_ti_within_type():
    from joblib import Parallel, delayed
    H = heldout_labels(); F = feats(); cuts = heldout_cuts(H['labels'], .2); t0 = time.time()
    rows = Parallel(n_jobs=N_JOBS)(delayed(_f_job)(sub_feats(F, a, b), a, b, L, cuts, H['seeds']) for (a, b), L in zip(H['pairs'], H['labels']))
    T = pd.DataFrame([r for R in rows for r in R]).groupby(['task_a', 'task_b', 'seed']).mean(numeric_only=True).drop(columns='dir').reset_index()
    T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    # pairs share units and tasks, so (pair, network) rows are not independent: fit the slope inside each network and test the
    # 10 network slopes against zero (Wilcoxon), as for every other panel
    import statsmodels.formula.api as smf
    from scipy.stats import wilcoxon
    fits = {}
    for t in TYPES:
        g = T[T.type == t].dropna(subset=['across20', 'pct_TI'])
        per = pd.DataFrame([dict(seed=s, slope=np.polyfit(gs.pct_TI, gs.across20, 1)[0], intercept=np.polyfit(gs.pct_TI, gs.across20, 1)[1], n=len(gs))
                            for s, gs in g.groupby('seed') if len(gs) >= 5 and gs.pct_TI.nunique() > 2])
        fits[t] = dict(per_net=per, slope=float(per.slope.mean()), sem=float(per.slope.sem()), intercept=float(per.intercept.mean()),
                       p=float(wilcoxon(per.slope, method='exact').pvalue), n_rows=len(g), n_nets=len(per))
    g = T.dropna(subset=['across20', 'pct_TI'])
    per = pd.DataFrame([dict(seed=s, slope=float(smf.ols('across20 ~ pct_TI + C(type)', gs).fit().params['pct_TI'])) for s, gs in g.groupby('seed')])
    fits['pooled'] = dict(per_net=per, slope=float(per.slope.mean()), sem=float(per.slope.sem()), p=float(wilcoxon(per.slope, method='exact').pvalue), n_rows=len(g), n_nets=len(per))
    save('pct_ti_within_type', dict(table=T, fits=fits, cuts=cuts))
    print(f'f: {len(T)} rows in {time.time() - t0:.0f}s', flush=True)


# ═══ g · noise-matched analog pairs ═════════════════════════════════════════════════════════════════════════════════════
def _noisy(F, a, b, kk, rep=0):
    """Independent Gaussian noise on every unit and trial: kk x the unit's within-condition SD (task x choice), per network."""
    w = Q.primary_window(a, b); Fn = dict(sub_feats(F, a, b))
    if kk == 0:
        return Fn
    Xa, Xb = F[f'{a}|{w}'].astype(np.float32), F[f'{b}|{w}'].astype(np.float32)
    ya, yb = F[f'{a}|y'], F[f'{b}|y']; rng = np.random.default_rng([int(round(kk * 1000)), rep, 3])
    na, nb = np.empty_like(Xa), np.empty_like(Xb)
    for k in range(Xa.shape[0]):
        var = [X[k][yy[k] == c].var(0) for X, yy in ((Xa, ya), (Xb, yb)) for c in (0, 1) if (yy[k] == c).sum() > 1]
        sd = np.sqrt(np.mean(var, 0))
        na[k] = Xa[k] + kk * sd * rng.standard_normal(Xa[k].shape)
        nb[k] = Xb[k] + kk * sd * rng.standard_normal(Xb[k].shape)
    Fn[f'{a}|{w}'], Fn[f'{b}|{w}'] = na, nb
    return Fn


def _g_level(F, a, b, keep, seeds, kk):
    Fn = _noisy(F, a, b, kk)
    L = PH._label_pass(Fn, a, b, keep)
    cuts = heldout_cuts([L], .2)                                         # the 20 % rule inside the noisy condition (pooled over networks)
    T = pd.DataFrame(PH._decode_pass(Fn, a, b, L, cuts, seeds))
    T = T.groupby('seed').mean(numeric_only=True).reset_index(); T['sigma'] = kk
    return T


# the recordings' three-point values (the paper's Figure 4 h, n: across-task decoding of size-matched TD-only / random / TI-only
# populations, mean over sessions) that the noise sweep is compared with, for two mouse analysis windows: 'fig3f' (the earlier window)
# and 'pre_2_1' (-2 to -1 s before turn onset, the paper's window; see panel_noise_sweep_retarget). reference/three_point.csv
REF_THREE_POINT = os.path.join(REFERENCE, 'three_point.csv')

def _three_point_data(window='fig3f'):
    ref = pd.read_csv(REF_THREE_POINT); ref = ref[ref.window == window].set_index(['species', 'pop']).across_cd
    return {sp: {p: float(ref[key, p]) for p in ('TD', 'random', 'TI')} for sp, key in (('mouse', 'mouse'), ('macaque', 'monkey'))}

def _g_fit(v):
    # best noise level against the data values, from the stored sweep table (no network is run)
    m = v['table'].groupby('sigma')[['TD_across', 'random_across', 'TI_across']].mean(); fine = np.linspace(0, v['kmax'], 201)
    curves = {p: np.interp(fine, m.index.values, m[f'{p}_across'].values) for p in ('TD', 'random', 'TI')}
    sse = sum((curves[p] - v['data'][p]) ** 2 for p in ('TD', 'random', 'TI'))
    v.update(best_sigma=float(fine[np.argmin(sse)]), best_sse=float(sse.min()),
             best_level=float(v['levels'][np.argmin([sum((m.loc[l, f'{p}_across'] - v['data'][p]) ** 2 for p in ('TD', 'random', 'TI')) for l in v['levels']])]))
    return v


def panel_noise_sweep_retarget(window):
    """Panel g against the data three-point of another mouse window: the network noise sweep (results/panels/g.pkl) is reused unchanged; only
    the data values and the best-fitting noise level change. Saved as results/panels/g_<window>.pkl."""
    out = load('noise_sweep'); data = _three_point_data(window)
    for sp in ('mouse', 'macaque'):
        out[sp] = _g_fit(dict(out[sp], data=data[sp]))
    out['data_source'] = f'reference/three_point.csv (paper Figure 4 three-point, full setting, mouse window {window}; session means)'
    save(f'noise_sweep_{window}', out)


def panel_noise_sweep(n_levels=6, target_within=0.75):
    F = feats(); seeds = list(F['seeds']); out = {}
    data = _three_point_data('fig3f')
    for sp, (a, b) in ANALOG.items():
        keep = Q.ok_seeds(RUN, a, b, seeds); t0 = time.time()
        # coarse grid to find the noise at which within-task decoding of the full choice population falls to ~0.75
        coarse = [0, 1, 2, 4, 8, 16, 32, 64]; cw = []
        for kk in coarse:
            cw.append(float(_g_level(F, a, b, keep, seeds, kk).within.mean()))
            if cw[-1] < target_within - .03:
                break
        cg = np.array(coarse[:len(cw)]); cw = np.array(cw)
        j = int(np.argmax(cw < target_within)) if (cw < target_within).any() else len(cw) - 1
        kmax = float(np.exp(np.interp(target_within, cw[[j, j - 1]], np.log(np.maximum(cg[[j, j - 1]], .5))))) if j > 0 else float(cg[j])
        levels = np.round(np.linspace(0, kmax, n_levels), 2)
        T = pd.concat([_g_level(F, a, b, keep, seeds, kk) for kk in levels], ignore_index=True)
        m = T.groupby('sigma')[['TD_across', 'random_across', 'TI_across']].mean()
        fine = np.linspace(0, kmax, 201)
        curves = {p: np.interp(fine, m.index.values, m[f'{p}_across'].values) for p in ('TD', 'random', 'TI')}
        sse = sum((curves[p] - data[sp][p]) ** 2 for p in ('TD', 'random', 'TI'))
        out[sp] = dict(pair=(a, b), table=T, levels=levels, coarse=dict(sigma=cg, within=cw), kmax=kmax, data=data[sp],
                       best_sigma=float(fine[np.argmin(sse)]), best_sse=float(sse.min()),
                       best_level=float(levels[np.argmin([sum((m.loc[l, f'{p}_across'] - data[sp][p]) ** 2 for p in ('TD', 'random', 'TI')) for l in levels])]))
        print(f'g {sp}: kmax {kmax:.2f}, best sigma {out[sp]["best_sigma"]:.2f} ({time.time() - t0:.0f}s)', flush=True)
    out['data_source'] = 'reference/three_point.csv, window fig3f (paper Figure 4 three-point, full setting; session means)'
    save('noise_sweep', out)


# ═══ g (right) · the pursuit-output dial ══════════════════════════════════════════════════════════════════════════════════
DIAL_G = (0.0, 0.25, 0.5, 0.75, 1.0)


def panel_pursuit_dial(window, prefix='pursuit_dial_g', name='pursuit_dial'):
    """Sac vs Pursuit in networks whose pursuit answer is split between the shared eye ring (share g) and its own ring (1 - g); the 27
    Fig. 5 tasks, trained by scripts/train_all.sh. Read from the stored measured read-out (measured.py monkey_like: FEF trial counts,
    noise calibrated per g on FEF within-task accuracy, 20 % tails pooled over the networks of a g); no network is run. Data values are
    those of panel g (same window). Saved as results/panels/<name>.pkl."""
    rows, k = [], {}
    for g in DIAL_G:
        run = f'{prefix}{int(round(g * 100)):03d}'
        D = pd.read_parquet(os.path.join(MEASURED, f'{run}__monkey_like', 'decode.parquet'))
        D = D[D['mode'] == 'interleaved']
        m = D.groupby(['seed', 'set']).across_cd.mean().unstack(); w = D[D.set == 'full'].groupby('seed').within_cd.mean()
        rows.append(pd.DataFrame(dict(g=g, seed=m.index, within=w.reindex(m.index).values, full_across=m['full'].values,
                                      TD_across=m['TD'].values, random_across=m['random'].values, TI_across=m['TI'].values)))
        M = pd.read_parquet(os.path.join(Q.OUT, run, 'measured.parquet')); k[g] = float(M[M.row == 'FEF'].obs_k.iloc[0])
    T = pd.concat(rows, ignore_index=True)
    data = _three_point_data(window)['macaque']
    m = T.groupby('g')[['TD_across', 'random_across', 'TI_across']].mean(); fine = np.linspace(0, 1, 401)
    curves = {p: np.interp(fine, m.index.values, m[f'{p}_across'].values) for p in ('TD', 'random', 'TI')}
    sse = sum((curves[p] - data[p]) ** 2 for p in ('TD', 'random', 'TI'))
    # g at which each curve equals the data value (first crossing on the fine grid; nan if never within 0.01)
    match = {p: (float(fine[np.argmin(np.abs(curves[p] - data[p]))]) if np.min(np.abs(curves[p] - data[p])) < .01 else np.nan) for p in curves}
    save(name, dict(table=T, runs=prefix, obs_k=k, data=data, best_g=float(fine[np.argmin(sse)]), best_sse=float(sse.min()), match_g=match,
                    data_source=f'reference/three_point.csv (as the noise sweep, window {window})'))


def simulate_training_trajectory(device=None, iters=(1000, 2000, 5000, 10000, 20000)):
    device = pick_device(device)
    from ..train import RUNS, load_nets
    for it in iters:
        path = os.path.join(CACHE, f'trajectory_features_{it}.npz')
        if os.path.exists(path):
            continue
        net, bat, cfg, cks = load_nets(os.path.join(RUNS, RUN), tag=f'ckpt_{it:06d}', device=device); t0 = time.time(); out = {}
        for t in bat.names:
            o, _ = _simulate(net, bat, cfg, device, t, 480, seed=11)
            out.update(o)
        prov = np.stack([np.asarray(c['prov_perf']) for c in cks])
        np.savez(path, names=np.array(bat.names), seeds=np.array(net.seeds), prov_perf=prov, **out)
        print(f'  traj {it}: {time.time() - t0:.0f}s, noise-free perf {prov.mean():.3f}', flush=True)
        del net


# ═══ training trajectory ════════════════════════════════════════════════════════════════════════════════════════════════
def _traj_job(Fp, a, b, ks):
    w = Q.primary_window(a, b); out = []
    for k in ks:
        X, y, ctx = Q.session(Fp, a, b, w, k)
        if len(np.unique(y[ctx == 1])) < 2 or len(np.unique(y[ctx == 2])) < 2:
            continue
        r = sn._summarise(sn._decode_matrix(X, y, ctx, method='fast'))
        active = (X[ctx == 1].mean(0) > 1e-4) | (X[ctx == 2].mean(0) > 1e-4)
        g = active & np.isfinite(r['snCCGP']) & (np.fmax(r['sn_within_1'], r['sn_within_2']) >= Q.GROUP_MIN)
        out.append((k, np.asarray(r['snCCGP'])[g]))
    return a, b, out


def panel_training_trajectory(iters=(1000, 2000, 5000, 10000, 20000)):
    from joblib import Parallel, delayed
    _, _, cuts = unit_table(RUN); lo, hi = cuts; rows, perf = [], {}
    for it in iters:
        z = np.load(os.path.join(CACHE, f'trajectory_features_{it}.npz')); F = {k: z[k] for k in z.files}
        names = list(F['names']); seeds = list(F['seeds']); perf[it] = F['prov_perf']; pp = F['prov_perf']
        # exclusion at each checkpoint: a network leaves a pair if its noise-free performance on either task is < 0.90 there
        okk = lambda k, a, b: pp[k][names.index(a)] >= MIN_PERF and pp[k][names.index(b)] >= MIN_PERF
        res = Parallel(n_jobs=N_JOBS)(delayed(_traj_job)(sub_feats(F, a, b), a, b, [k for k in range(len(seeds)) if okk(k, a, b)])
                                      for a, b in pairs_of(names))
        for a, b, out in res:
            for k, v in out:
                if len(v):
                    rows.append(dict(iter=it, task_a=a, task_b=b, seed=seeds[k], n_group=len(v), pct_TI=100 * np.mean(v >= hi), pct_TD=100 * np.mean(v <= lo),
                                     perf_a=float(F['prov_perf'][k][names.index(a)]), perf_b=float(F['prov_perf'][k][names.index(b)])))
        print(f'traj {it}: done', flush=True)
    T = pd.DataFrame(rows); T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    save('training_trajectory', dict(table=T, cuts=cuts, prov_perf=perf, names=names))
