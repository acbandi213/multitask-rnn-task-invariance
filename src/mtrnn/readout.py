"""Q1 read-out: every unit of every network on two axes for every task pair — where its choice tuning lives (gain-free FTV) and
whether it keeps the same choice code across the two tasks (the paper's snCCGP) — plus the population consequences.

Choice = hemifield of the network's own response (right = cos(direction) > 0), errors included; no-go trials are left out.

Stages (`python -m mtrnn.readout <stage> <run>`):
    features   analysis-setting trials of every task (grid directions, fixed timing, noise on), window-mean rates per unit
               and trial in two windows: 'pre' = last 300 ms before the go cue, 'resp' = 100-400 ms after it.
               -> results/readout/<run>/features.npz
    pairs      for all 351 task pairs, every network, both windows:
                 per unit   snCCGP (lib/index.py fast estimator, whole session), within-task accuracy per task, signed d'
                            per task, gain-free FTV_d (lib.measures.dprime2, bias-corrected), active, group
                 per pair   held-out single-cell share (group decided on one half of the trials, snCCGP and within scored on the
                            other; 5 interleaved splits x 2 directions; the form used for the data), population within / across
                            (the paper's cd_decode on per-task z-scored activity of all active units)
               -> results/readout/<run>/units.parquet, pairs.parquet
    measured   the four anchor pairs read like the data: data trial counts per task x choice cell, measurement noise
               k x each unit's within-condition SD, k calibrated (bisection, pooled over networks) ONLY so the mean within-task
               accuracy of held-out-gated units equals the data's; then the single-cell and population shares.
               -> results/readout/<run>/measured.parquet

Group = active units (mean rate > 1e-4 in either task) with within-task accuracy >= 0.55 in either task.
Classes of group units: one-task |FTV_d| >= 0.5; shared-same |FTV_d| < 0.5 and d'1 d'2 > 0; shared-reversed |FTV_d| < 0.5 and d'1 d'2 < 0.
Primary window of a pair: 'pre' when both tasks carry the choice before go (PRE_OK), else 'resp'.
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .tasks import ANCHOR_PAIRS, MOUSE_PAIR, hemifield, popvec, ring_slice, resp_ring_rates, GRACE
from .train import RUNS, load_nets, run_trials

from .lib import index as sn
from .lib.population import make_halves, zctx, cd_decode   # the functions used for the recordings
from .lib.measures import dprime2, add_obs_noise
from .paths import PERFORMANCE, READOUT, pick_device

OUT = READOUT                          # results/readout/<run>/
WINDOWS = {'pre': (-15, 0), 'resp': (GRACE, GRACE + 15)}               # steps relative to the response cue
PRE_OK = {'Go', 'Dly Go', 'Anti', 'Dly Anti', 'DM 1', 'DM 2', 'Ctx DM 1', 'Ctx DM 2', 'MultSen DM', 'Dly DM 1', 'Dly DM 2',
          'Ctx Dly DM 1', 'Ctx Dly DM 2', 'MultSen Dly DM', 'Go Mod1', 'Go Mod2', 'Sac', 'Cat', 'Match', 'Cat-B', 'T-maze Vis', 'T-maze Aud'}
NOGO = {'DMS', 'DNMS', 'DMC', 'DNMC'}
GROUP_MIN = 0.55
MIN_TASK_PERF = 0.80     # a network is left out of a pair when it performs either task below this (evaluate.py perf.csv)
ONE_TASK = 0.5
N_SPLITS = 5
# data the anchors are compared with (analysis/transfer_extremes/cache/survival.csv, held-out; population shares from the
# transfer-extremes report): within = mean within-task accuracy of the held-out-gated cells (the calibration target)
DATA = pd.DataFrame([
    dict(row='M2', anchor='modality', n_cell=50, within=0.567, share=0.856, share_lo=0.638, share_hi=0.987, pop_share=0.77),
    dict(row='PPC', anchor='modality', n_cell=50, within=0.600, share=0.780, share_lo=0.740, share_hi=0.820, pop_share=0.88),
    dict(row='FEF', anchor='movement', n_cell=150, within=0.565, share=0.230, share_lo=0.171, share_hi=0.280, pop_share=0.39),
    dict(row='human task', anchor='task', n_cell=80, within=0.542, share=0.398, share_lo=-0.182, share_hi=0.684, pop_share=0.19),
    dict(row='human effector', anchor='effector', n_cell=80, within=0.535, share=1.087, share_lo=0.862, share_hi=1.617, pop_share=1.16),
])
# the faithful T-maze pair (2026-10-26) is calibrated against PPC too; measured only in runs that have it
DATA_TMAZE = pd.DataFrame([dict(row='PPC T-maze', anchor='tmaze', n_cell=50, within=0.600, share=0.780, share_lo=0.740, share_hi=0.820,
                                pop_share=0.88)])
MEASURE_PAIRS = {**ANCHOR_PAIRS, 'tmaze': MOUSE_PAIR}


def primary_window(a, b):
    return 'pre' if (a in PRE_OK and b in PRE_OK) else 'resp'


# ═══ features ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
def features(run, device=None, n_trials=480, seed=11):
    device = pick_device(device)
    net, bat, cfg, _ = load_nets(os.path.join(RUNS, run), device=device)
    rng = np.random.default_rng([seed, 17])
    out = {}
    for t in bat.names:
        P = bat.sample(rng, n_trials * (2 if t in NOGO else 1), task=t, test=True)
        keep = np.flatnonzero(P['respond'])
        P = {k: (v[keep] if k != 'meta' else {m: a[keep] for m, a in v.items()}) for k, v in P.items()}
        assert len(np.unique(P['resp_on'])) == 1 and len(np.unique(P['T'])) == 1
        r0, T = int(P['resp_on'][0]), int(P['T'][0])
        Xw = {w: [] for w in WINDOWS}; Y = []
        for s in range(0, len(keep), 160):
            sub = {k: (v[s:s + 160] if k != 'meta' else None) for k, v in P.items()}
            Z, R = run_trials(net, bat, sub, device, noise=True, keep_rates=True, sigma_in=cfg['sigma_in'], chunk=160)
            for w, (a, b) in WINDOWS.items():
                Xw[w].append(R[:, :, r0 + a:r0 + b].mean(2))
            ring = resp_ring_rates(Z[:, :, r0 + GRACE:T], P['resp_ring'][0], P['resp_share'][0]).mean(2)   # (K, n, 32)
            Y.append(hemifield(popvec(ring)))
        for w in WINDOWS:
            out[f'{t}|{w}'] = np.concatenate(Xw[w], 1).astype(np.float16)
        out[f'{t}|y'] = np.concatenate(Y, 1).astype(np.int8)
        out[f'{t}|truth'] = hemifield(P['resp_dir']).astype(np.int8)
        print(f'  {t}: {len(keep)} trials, choice accuracy {np.mean(out[f"{t}|y"] == out[f"{t}|truth"][None]):.3f}', flush=True)
    od = os.path.join(OUT, run); os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, 'features.npz'), names=np.array(bat.names), seeds=np.array(net.seeds), **out)
    return out


def load_features(run):
    z = np.load(os.path.join(OUT, run, 'features.npz'))
    return {k: z[k] for k in z.files}


# ═══ one pair ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
def session(F, a, b, w, k):
    Xa, Xb = F[f'{a}|{w}'][k].astype(float), F[f'{b}|{w}'][k].astype(float)
    X = np.vstack([Xa, Xb]); y = np.concatenate([F[f'{a}|y'][k], F[f'{b}|y'][k]]).astype(int)
    ctx = np.concatenate([np.ones(len(Xa), int), np.full(len(Xb), 2)])
    return X, y, ctx


def unit_measures(X, y, ctx):
    r = sn._summarise(sn._decode_matrix(X, y, ctx, method='fast'))
    q1, d1 = dprime2(X, y, ctx, 1); q2, d2 = dprime2(X, y, ctx, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        ftv_d = np.where(q1 + q2 > 0, (q1 - q2) / (q1 + q2), np.nan)
    active = (X[ctx == 1].mean(0) > 1e-4) | (X[ctx == 2].mean(0) > 1e-4)
    w1, w2 = np.asarray(r['sn_within_1']), np.asarray(r['sn_within_2'])
    group = active & np.isfinite(r['snCCGP']) & (np.fmax(w1, w2) >= GROUP_MIN)
    return dict(snCCGP=np.asarray(r['snCCGP']), within=np.asarray(r['sn_within']), within_1=w1, within_2=w2,
                d1=d1, d2=d2, ftv_d=ftv_d, active=active, group=group)


def heldout_share(X, y, ctx, active, n_splits=N_SPLITS):
    """(mean snCCGP - .5) / (mean within - .5) over (unit, split, direction) rows of units gated on the other half."""
    c_sum = w_sum = n = 0.0
    for s in range(n_splits):
        A, B = make_halves(y, ctx, seed=s)
        for g, h in ((A, B), (B, A)):
            rg = sn._summarise(sn._decode_matrix(X[g], y[g], ctx[g], method='fast'))
            gate = active & (np.fmax(rg['sn_within_1'], rg['sn_within_2']) >= GROUP_MIN)
            if not gate.any():
                continue
            rh = sn._summarise(sn._decode_matrix(X[h][:, gate], y[h], ctx[h], method='fast'))
            ok = np.isfinite(rh['snCCGP']) & np.isfinite(rh['sn_within'])
            c_sum += rh['snCCGP'][ok].sum(); w_sum += rh['sn_within'][ok].sum(); n += ok.sum()
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    c, w = c_sum / n, w_sum / n
    return (c - .5) / (w - .5), c, w, int(n)


def population(X, y, ctx, active):
    if active.sum() < 2:
        return dict(pop_within=np.nan, pop_across=np.nan, pop_cos=np.nan)
    r = cd_decode(zctx(X[:, active], ctx), y, ctx)
    if r is None:
        return dict(pop_within=np.nan, pop_across=np.nan, pop_cos=np.nan)
    return dict(pop_within=r['within_cd'], pop_across=r['across_cd'], pop_cos=r['cos_cd'])


def pair_job(Fpair, a, b, seeds):
    units, pairs = [], []
    for w in WINDOWS:
        for k, s in enumerate(seeds):
            X, y, ctx = session(Fpair, a, b, w, k)
            U = unit_measures(X, y, ctx)
            share, c, wi, n = heldout_share(X, y, ctx, U['active'])
            g = U['group']
            cls = np.where(np.abs(U['ftv_d']) >= ONE_TASK, 'one-task', np.where(U['d1'] * U['d2'] > 0, 'shared-same', 'shared-reversed'))
            pairs.append(dict(task_a=a, task_b=b, window=w, primary=w == primary_window(a, b), seed=s, n_trials=len(y),
                              n_active=int(U['active'].sum()), n_group=int(g.sum()),
                              median_snccgp=float(np.median(U['snCCGP'][g])) if g.any() else np.nan,
                              mean_snccgp=float(np.mean(U['snCCGP'][g])) if g.any() else np.nan,
                              frac_one_task=float(np.mean(cls[g] == 'one-task')) if g.any() else np.nan,
                              frac_same=float(np.mean(cls[g] == 'shared-same')) if g.any() else np.nan,
                              frac_reversed=float(np.mean(cls[g] == 'shared-reversed')) if g.any() else np.nan,
                              share=share, share_snccgp=c, share_within=wi, share_rows=n,
                              choice_acc_a=float(np.mean(y[ctx == 1] == Fpair[f'{a}|truth'])), choice_acc_b=float(np.mean(y[ctx == 2] == Fpair[f'{b}|truth'])),
                              **population(X, y, ctx, U['active'])))
            keep = U['active']
            units.append(pd.DataFrame(dict(task_a=a, task_b=b, window=w, seed=s, unit=np.flatnonzero(keep), cls=cls[keep],
                                           **{key: U[key][keep] for key in ('snCCGP', 'within_1', 'within_2', 'd1', 'd2', 'ftv_d', 'group')})))
    return pd.concat(units, ignore_index=True), pd.DataFrame(pairs)


def run_pairs(run, n_jobs=16, pairs=None):
    from joblib import Parallel, delayed
    F = load_features(run)
    names = list(F['names']); seeds = list(F['seeds'])
    pairs = pairs or list(itertools.combinations(names, 2))
    sub = lambda a, b: {key: F[key] for key in F if key.split('|')[0] in (a, b)}
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(pair_job)(sub(a, b), a, b, seeds) for a, b in pairs)
    U = pd.concat([r[0] for r in res], ignore_index=True); P = pd.concat([r[1] for r in res], ignore_index=True)
    od = os.path.join(OUT, run)
    U.to_parquet(os.path.join(od, 'units.parquet')); P.to_parquet(os.path.join(od, 'pairs.parquet'))
    print(f'{run}: {len(pairs)} pairs in {time.time() - t0:.0f}s', flush=True)
    return U, P


# ═══ measured read-out for the anchors ══════════════════════════════════════════════════════════════════════════════════
def _subsample(F, a, b, w, k, n_cell, seed):
    X, y, ctx = session(F, a, b, w, k)
    rng = np.random.default_rng([seed, k])
    idx = []
    for c in (1, 2):
        for v in (0, 1):
            m = np.flatnonzero((ctx == c) & (y == v))
            if len(m):
                idx.append(rng.choice(m, min(n_cell, len(m)), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx], ctx[idx]


def measured_session(F, a, b, w, k, n_cell, obs_k, seed):
    X, y, ctx = _subsample(F, a, b, w, k, n_cell, seed)
    active = (X[ctx == 1].mean(0) > 1e-4) | (X[ctx == 2].mean(0) > 1e-4)
    return add_obs_noise(X, y, ctx, obs_k, seed * 1000 + k), y, ctx, active


def measured_within(F, a, b, w, seeds, n_cell, obs_k, reps=2):
    vals = []
    for rep in range(reps):
        for k in range(len(seeds)):
            X, y, ctx, act = measured_session(F, a, b, w, k, n_cell, obs_k, rep)
            _, _, wi, n = heldout_share(X, y, ctx, act, n_splits=2)
            if n:
                vals.append(wi)
    return float(np.mean(vals))


def calibrate(F, a, b, w, seeds, n_cell, target, lo=0.0, hi=24.0, tol=0.002):
    m = np.nan
    for _ in range(14):
        mid = 0.5 * (lo + hi); m = measured_within(F, a, b, w, seeds, n_cell, mid)
        if abs(m - target) < tol:
            break
        lo, hi = (mid, hi) if m > target else (lo, mid)
    return mid, m


def task_perf(run):
    """(seed, task) -> Yang performance, from the evaluate.py read-out of this run."""
    P = pd.read_csv(os.path.join(PERFORMANCE, run, 'perf.csv'))
    return {(int(r.seed), r.task): float(r.perf) for r in P.itertuples()}


def ok_seeds(run, a, b, seeds):
    tp = task_perf(run)
    return [k for k, s in enumerate(seeds) if tp[(int(s), a)] >= MIN_TASK_PERF and tp[(int(s), b)] >= MIN_TASK_PERF]


def run_measured(run, reps=5):
    F = load_features(run); seeds = list(F['seeds'])
    rows = []
    names = list(F['names'])
    for _, d in pd.concat([DATA, DATA_TMAZE], ignore_index=True).iterrows():
        a, b = MEASURE_PAIRS[d.anchor]; w = primary_window(a, b)
        if a not in names or b not in names:
            continue
        keep = ok_seeds(run, a, b, seeds)
        Fk = {key: (v[keep] if ('|' in key and not key.endswith('|truth')) else v) for key, v in F.items()}
        k_obs, wcal = calibrate(Fk, a, b, w, [seeds[k] for k in keep], d.n_cell, d.within)
        for rep in range(reps):
            for k, s in ((k, seeds[k]) for k in keep):
                X, y, ctx, act = measured_session(F, a, b, w, k, d.n_cell, k_obs, 100 + rep)       # k indexes the full feature arrays
                share, c, wi, n = heldout_share(X, y, ctx, act)
                pop = population(X, y, ctx, act)
                rows.append(dict(row=d.row, anchor=d.anchor, task_a=a, task_b=b, window=w, seed=s, rep=rep, obs_k=k_obs,
                                 calibrated_within=wcal, share=share, share_snccgp=c, share_within=wi, **pop))
        print(f'  {run} {d.row}: k = {k_obs:.2f} (within {wcal:.3f} vs data {d.within}), share {np.nanmean([r["share"] for r in rows if r["row"] == d.row]):.2f}', flush=True)
    M = pd.DataFrame(rows)
    M['pop_share'] = (M.pop_across - .5) / (M.pop_within - .5)
    M.to_parquet(os.path.join(OUT, run, 'measured.parquet'))
    return M


# ═══ relationship of two tasks (for the map) ═══════════════════════════════════════════════════════════════════════════
FAMILY = {**{t: 'Go' for t in ('Go', 'RT Go', 'Dly Go', 'Go Mod1', 'Go Mod2', 'Sac')}, **{t: 'Anti' for t in ('Anti', 'RT Anti', 'Dly Anti')},
          **{t: 'DM' for t in ('DM 1', 'DM 2', 'Ctx DM 1', 'Ctx DM 2', 'MultSen DM')},
          **{t: 'Dly DM' for t in ('Dly DM 1', 'Dly DM 2', 'Ctx Dly DM 1', 'Ctx Dly DM 2', 'MultSen Dly DM')},
          **{t: 'Matching' for t in ('DMS', 'DNMS', 'DMC', 'DNMC')}, 'Pursuit': 'Pursuit', 'Cat': 'Cat', 'Cat-B': 'Cat', 'Match': 'Match',
          'T-maze Vis': 'T-maze', 'T-maze Aud': 'T-maze'}
# response mapping: 'pro' = answer toward a stimulus direction (Go, DM, matching tasks, Sac, Pursuit), 'anti' = away from it,
# 'abstract' = a fixed yes / no target unrelated to the stimulus direction (Cat, Match, Cat-B)
MAPPING = {t: ('anti' if f == 'Anti' else 'abstract' if f in ('Cat', 'Match') else 'pro') for t, f in FAMILY.items()}
MAPPING['T-maze Vis'] = 'abstract'      # grating orientation -> side: the answer is unrelated to any stimulus direction
RELATIONS = ['same operation', 'different operation', 'spatial vs abstract', 'reversed mapping', 'different output']
REL_DESC = {'same operation': 'same operation; timing, modality or context differ',
            'different operation': 'different operation, same response mapping and output',
            'spatial vs abstract': 'spatial answer vs fixed yes / no answer',
            'reversed mapping': 'one task answers toward the stimulus, the other away from it',
            'different output': 'answered on different output rings'}


def out_ring(t, layout):
    return 'B' if (layout == 'separate' and t in ('Pursuit', 'Cat-B')) else 'A'


def relation(a, b, layout):
    """One class per pair, in this priority: different output ring > reversed mapping > spatial vs abstract > same operation
    (same task family) > different operation."""
    if out_ring(a, layout) != out_ring(b, layout):
        return 'different output'
    ma, mb = MAPPING[a], MAPPING[b]
    if {ma, mb} == {'pro', 'anti'}:
        return 'reversed mapping'
    if (ma == 'abstract') != (mb == 'abstract'):
        return 'spatial vs abstract'
    return 'same operation' if FAMILY[a] == FAMILY[b] else 'different operation'


# ═══ task features of a pair (what dictates reuse) ═════════════════════════════════════════════════════════════════════
# input rings a task uses ('m1', 'm2' stimulus modalities, 'mot' motion); tasks whose cue modality is random per trial use both
INPUTS = {**{t: {'m1', 'm2'} for t in ('Go', 'RT Go', 'Dly Go', 'Anti', 'RT Anti', 'Dly Anti', 'Ctx DM 1', 'Ctx DM 2', 'MultSen DM',
                                    'Ctx Dly DM 1', 'Ctx Dly DM 2', 'MultSen Dly DM', 'DMS', 'DNMS', 'DMC', 'DNMC')},
          'DM 1': {'m1'}, 'DM 2': {'m2'}, 'Dly DM 1': {'m1'}, 'Dly DM 2': {'m2'}, 'Go Mod1': {'m1'}, 'Go Mod2': {'m2'}, 'Sac': {'m1'},
          'Pursuit': {'m1', 'mot'}, 'Cat': {'m1'}, 'Match': {'m1'}, 'Cat-B': {'m1'}, 'T-maze Vis': {'ori'}, 'T-maze Aud': {'m2'}}
MEMORY = {'Dly Go', 'Dly Anti', 'Dly DM 1', 'Dly DM 2', 'Ctx Dly DM 1', 'Ctx Dly DM 2', 'MultSen Dly DM', 'DMS', 'DNMS', 'DMC', 'DNMC',
          'Sac', 'Cat', 'Match', 'Cat-B'}                                   # a delay the stimulus must be held across
AT_ONSET = {'RT Go', 'RT Anti', 'DMS', 'DNMS', 'DMC', 'DNMC', 'Pursuit'}     # the answer is due at stimulus onset (no go cue after it)
FEATURES = ['different output ring', 'reversed mapping', 'spatial vs abstract answer', 'different task family',
            'input rings not shared', 'memory delay in one task only', 'answer-at-onset in one task only']


def pair_features(a, b, layout):
    ia, ib = INPUTS[a], INPUTS[b]
    ma, mb = MAPPING[a], MAPPING[b]
    return {'different output ring': float(out_ring(a, layout) != out_ring(b, layout)),
            'reversed mapping': float({ma, mb} == {'pro', 'anti'}),
            'spatial vs abstract answer': float((ma == 'abstract') != (mb == 'abstract')),
            'different task family': float(FAMILY[a] != FAMILY[b]),
            'input rings not shared': 1.0 - len(ia & ib) / len(ia | ib),
            'memory delay in one task only': float((a in MEMORY) != (b in MEMORY)),
            'answer-at-onset in one task only': float((a in AT_ONSET) != (b in AT_ONSET))}


def feature_regression(G, layout, target='share', n_boot=1000, seed=0):
    """OLS of a pair-level measure (one row per pair, mean over networks) on the pair features, all at once; coefficient =
    change in the measure when the feature is present (or fully unshared), holding the others fixed. 95 % CI by bootstrap over
    pairs. Constant features (e.g. 'different output ring' in the shared layout) are dropped."""
    Xf = pd.DataFrame([pair_features(a, b, layout) for a, b in zip(G.task_a, G.task_b)])
    cols = [c for c in FEATURES if Xf[c].std() > 0]
    X = np.column_stack([np.ones(len(G))] + [Xf[c].to_numpy() for c in cols]); y = G[target].to_numpy()
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y)); bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bs = np.array(bs); r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return pd.DataFrame(dict(feature=['intercept'] + cols, coef=beta, lo=np.percentile(bs, 2.5, 0), hi=np.percentile(bs, 97.5, 0))), r2


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['features', 'pairs', 'measured', 'all']); p.add_argument('run')
    p.add_argument('--device', default=None); p.add_argument('--n_jobs', type=int, default=16)
    a = p.parse_args()
    if a.stage in ('features', 'all'):
        features(a.run, a.device)
    if a.stage in ('pairs', 'all'):
        run_pairs(a.run, a.n_jobs)
    if a.stage in ('measured', 'all'):
        run_measured(a.run)
