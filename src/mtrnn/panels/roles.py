"""Extended Data Figure 8: what TI and TD units do in the multitask RNN. One function per panel; each writes
results/panels/<name>.pkl, which analysis/figure_s8.ipynb reads.
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


# ═══ i · removal and decoding: within, across, XOR ══════════════════════════════════════════════════════════════════════
def _i_job(Fp, a, b, labels, cuts, seeds, n_cls=5, n_rand=10, n_xor=5):
    w = Q.primary_window(a, b); rows = []
    for lab in labels:
        k, d = lab['k'], lab['dir']
        X, y, ctx = Q.session(Fp, a, b, w, k)
        A, B = make_halves(y, ctx, seed=0); Dh = B if d == 0 else A
        Xd, yd, cd = X[Dh], y[Dh], ctx[Dh]
        cond = DEC.conditions(yd, cd); mmin = int(np.bincount(cond, minlength=4).min())
        g = np.flatnonzero(lab['group']); v = lab['idx'][g]; lo, hi = cuts[d]
        ti, td = g[v >= hi], g[v <= lo]; n = min(len(ti), len(td))
        if n < MIN_N or mmin < MIN_PER or len(g) - n < MIN_N:
            continue
        Zc, Zp = zctx(Xd, cd), DEC.zpool(Xd)
        rng = np.random.default_rng([int(seeds[k]), d, 23])
        idx = DEC.balanced_index(cond, mmin, rng, n_xor)

        def measure(cols, i):
            r = cd_decode(Zc[:, cols], yd, cd, seed=42 + i)
            xor = float(np.nanmean(DEC.within_dichotomies(DEC.gather(Zp, idx, cols), dich=('xor',))['xor']))
            return (r['within_cd'], r['across_cd'], xor) if r is not None else (np.nan, np.nan, xor)

        full = np.array(measure(g, 0))
        res = {}
        for name, pool, nd in (('-TI', ti, n_cls), ('-TD', td, n_cls), ('-random', g, n_rand)):
            res[name] = np.nanmean([measure(np.setdiff1d(g, rng.choice(pool, n, replace=False)), i) for i in range(nd)], 0) - full
        row = dict(task_a=a, task_b=b, seed=int(seeds[k]), dir=d, n_removed=n, n_group=len(g),
                   full_within=full[0], full_across=full[1], full_xor=full[2])
        for name, r in res.items():
            for j, m in enumerate(('within', 'across', 'xor')):
                row[f'{name}_{m}'] = r[j]
        rows.append(row)
    return rows


def panel_unit_removal():
    from joblib import Parallel, delayed
    H = heldout_labels(); F = feats(); cuts = heldout_cuts(H['labels'], .2); t0 = time.time()
    rows = Parallel(n_jobs=N_JOBS)(delayed(_i_job)(sub_feats(F, a, b), a, b, L, cuts, H['seeds']) for (a, b), L in zip(H['pairs'], H['labels']))
    T = pd.DataFrame([r for R in rows for r in R]).groupby(['task_a', 'task_b', 'seed']).mean(numeric_only=True).drop(columns='dir').reset_index()
    T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    save('unit_removal', dict(table=T, cuts=cuts))
    print(f'i: {len(T)} rows in {time.time() - t0:.0f}s', flush=True)


# ═══ k · stimulus vs response coding in the reversed-mapping pairs ══════════════════════════════════════════════════════
def stimulus_sides(run=RUN, n_trials=480, seed=11):
    """Replays the trial parameters of readout.features (same generator and order) -> per task the stimulus hemifield of every kept
    trial: the cue's side for single-stimulus tasks (Anti: opposite to the response), else the correct target's side."""
    from ..train import RUNS, load_nets
    from ..tasks import hemifield
    _, bat, cfg, _ = load_nets(os.path.join(RUNS, run), device='cpu')
    rng = np.random.default_rng([seed, 17]); F = feats(run); out = {}
    for t in bat.names:
        P = bat.sample(rng, n_trials * (2 if t in Q.NOGO else 1), task=t, test=True)
        keep = np.flatnonzero(P['respond'])
        truth = hemifield(P['resp_dir'][keep]); assert np.array_equal(truth, F[f'{t}|truth']), t
        out[t] = hemifield(np.asarray(P['meta']['stim_dir'])[keep]) if 'stim_dir' in P['meta'] else truth
    return out


def _auc_sel(X, lab):
    from scipy.stats import rankdata
    r = rankdata(X, axis=0); n1 = lab.sum(); n0 = len(lab) - n1
    auc = (r[lab == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.abs(auc - .5) * 2


def panel_stim_vs_response():
    U, P, cuts = unit_table(RUN); F = feats(); seeds = list(F['seeds']); S = stimulus_sides(); rows = []
    R = U[U.type == 'reversed mapping']
    for (a, b, s), g in R.groupby(['task_a', 'task_b', 'seed']):
        k = seeds.index(int(s)); X, y, ctx = Q.session(F, a, b, Q.primary_window(a, b), k)
        stim = np.concatenate([S[a], S[b]]); Xg = X[:, g.unit.to_numpy()]
        rows.append(pd.DataFrame(dict(task_a=a, task_b=b, seed=int(s), unit=g.unit.to_numpy(), cls=g.cls_ti.to_numpy(),
                                      stim_sel=_auc_sel(Xg, stim), resp_sel=_auc_sel(Xg, y),
                                      stim_resp_agree=float(np.mean(stim == y)))))
    T = pd.concat(rows, ignore_index=True)
    save('stim_vs_response', dict(table=T, cuts=cuts))
    print(f'k: {len(T):,} unit-pairs, {T.groupby(["task_a", "task_b"]).ngroups} pairs', flush=True)


# ═══ l · rule-input weights ═════════════════════════════════════════════════════════════════════════════════════════════
def _weights(run=RUN):
    from ..train import RUNS, load_nets
    from ..tasks import N_IN_RINGS, N_RING
    net, bat, cfg, _ = load_nets(os.path.join(RUNS, run), device='cpu')
    rule0 = 1 + N_IN_RINGS * N_RING
    return dict(W_in=net.W_in.detach().numpy(), W_rec=net.W_rec.detach().numpy(), seeds=list(net.seeds),
                rule_col={t: rule0 + i for i, t in enumerate(bat.names)})


def panel_rule_weights():
    U, P, cuts = unit_table(RUN); W = _weights(); rows = []
    for (a, b, s), g in U.groupby(['task_a', 'task_b', 'seed']):
        k = W['seeds'].index(int(s)); u = g.unit.to_numpy()
        diff = np.abs(W['W_in'][k, u, W['rule_col'][a]] - W['W_in'][k, u, W['rule_col'][b]])
        rows.append(pd.DataFrame(dict(task_a=a, task_b=b, seed=int(s), unit=u, cls=g.cls_ti.to_numpy(), type=g.type.iloc[0], rule_diff=diff)))
    save('rule_weights', dict(table=pd.concat(rows, ignore_index=True)))


# ═══ m · recurrent connectivity onto TI and TD units ════════════════════════════════════════════════════════════════════
FIVE = ['TI same', 'TI opp', 'TD same', 'TD opp', 'control']


def _m_values(M, cls, pref, rng, n_perm=0):
    """M (n, n) W_rec among the pair's choice units (row = target, column = source, diagonal NaN). Returns {target class: [5]}, the
    mean over targets of each target's mean input weight from the five source sets; with n_perm, the same for permuted labels."""
    n = len(cls); Mz = np.nan_to_num(M); V = (~np.isnan(M)).astype(float)

    def one(c, with_ctl=True):
        out = {}
        src = np.stack([(c == 'TI') & (pref > 0), (c == 'TI') & (pref < 0), (c == 'TD') & (pref > 0), (c == 'TD') & (pref < 0)], 1).astype(float)
        num, den = Mz @ src, V @ src                                     # (n, 4): summed weight / count from each (class, sign) set
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = num / den
        for tc in ('TI', 'TD'):
            tg = np.flatnonzero(c == tc)
            if len(tg) == 0:
                out[tc] = np.full(5, np.nan); continue
            same = pref[tg] > 0
            ti_s = np.where(same, mean[tg, 0], mean[tg, 1]); ti_o = np.where(same, mean[tg, 1], mean[tg, 0])
            td_s = np.where(same, mean[tg, 2], mean[tg, 3]); td_o = np.where(same, mean[tg, 3], mean[tg, 2])
            ctl = [np.nan]
            for i in (tg if with_ctl else []):                                                 # control: random choice units, n = size of the target's class
                pool = np.setdiff1d(np.arange(n), [i]); ctl.append(np.nanmean(M[i, rng.choice(pool, min(len(tg), len(pool)), replace=False)]))
            out[tc] = np.array([np.nanmean(ti_s), np.nanmean(ti_o), np.nanmean(td_s), np.nanmean(td_o), np.nanmean(ctl)])
        return out

    obs = one(cls)
    perms = [one(rng.permutation(cls), with_ctl=False) for _ in range(n_perm)]   # control does not depend on the labels
    return obs, perms


def _m_job(Wk, items, seed, n_perm):
    rng = np.random.default_rng([seed, 41]); res = []
    for a, b, u, cls, pref in items:
        M = Wk[np.ix_(u, u)].astype(float); np.fill_diagonal(M, np.nan)
        obs, perms = _m_values(M, cls, pref, rng, n_perm)
        res.append((a, b, obs, np.stack([[p['TI'], p['TD']] for p in perms]) if perms else None))
    return seed, res


def panel_recurrent_weights(n_perm=1000):
    from joblib import Parallel, delayed
    U, P, cuts = unit_table(RUN); W = _weights(); t0 = time.time()
    U = U[np.sign(U.d1) != 0]
    jobs = []
    for s, gs in U.groupby('seed'):
        items = [(a, b, g.unit.to_numpy(), g.cls_ti.to_numpy(), np.sign(g.d1.to_numpy())) for (a, b), g in gs.groupby(['task_a', 'task_b'])]
        jobs.append((W['W_rec'][W['seeds'].index(int(s))], items, int(s)))
    res = Parallel(n_jobs=N_JOBS)(delayed(_m_job)(Wk, items, s, n_perm) for Wk, items, s in jobs)
    rows, null = [], {}
    for s, R in res:
        for a, b, obs, perm in R:
            for tc in ('TI', 'TD'):
                rows.append(dict(seed=s, task_a=a, task_b=b, target=tc, **dict(zip(FIVE, obs[tc]))))
        P_ = np.stack([perm for _, _, _, perm in R])                     # (pairs, n_perm, 2, 5)
        null[s] = np.nanmean(P_, 0)                                      # network mean over pairs, per permutation
    T = pd.DataFrame(rows); T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    nullnet = np.nanmean(np.stack([null[s] for s in sorted(null)]), 0)  # mean over networks, per permutation (n_perm, 2, 5)
    save('recurrent_weights', dict(table=T, null=nullnet, n_perm=n_perm))
    print(f'm: {len(T)} rows, {n_perm} permutations in {time.time() - t0:.0f}s', flush=True)


def simulate_task_decoding(device=None):
    device = pick_device(device)
    from ..train import RUNS, load_nets
    net, bat, cfg, _ = load_nets(os.path.join(RUNS, RUN), device=device); t0 = time.time()
    d = os.path.join(CACHE, 'task_decoding_rates'); os.makedirs(d, exist_ok=True); ep = {}
    for t in bat.names:
        out, ep[t] = _simulate(net, bat, cfg, device, t, J_TRIALS, seed=23, binned=True)
        np.savez(os.path.join(d, f'{t.replace(" ", "_")}.npz'), bins=out[f'{t}|bins'], **{w: out[f'{t}|{w}'] for w in Q.WINDOWS}, y=out[f'{t}|y'])
        print(f'  j {t}: {out[f"{t}|bins"].shape} ({time.time() - t0:.0f}s)', flush=True)
    save('task_decoding_epochs', dict(epochs=ep, bin_ms=J_BIN * 20, pre_steps=J_PRE, post_steps=J_POST))


# ═══ j · task-identity decoding over time (TI vs TD subsets) ════════════════════════════════════════════════════════════
def _cv_md(X, lab, subsets, n_folds=5, seed=0):
    """5-fold CV mean-difference decoder for every unit subset at once. X (trials, units), subsets (S, n). -> (S,) accuracy."""
    rng = np.random.default_rng(seed); fold = np.empty(len(lab), int)
    for c in (0, 1):
        ii = np.flatnonzero(lab == c); fold[ii] = rng.permutation(np.arange(len(ii)) % n_folds)
    acc = np.zeros(len(subsets))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        Xs = X[:, subsets]                                               # (trials, S, n)
        mu0, mu1 = Xs[tr & (lab == 0)].mean(0), Xs[tr & (lab == 1)].mean(0)
        w = mu1 - mu0; thr = (w * (mu0 + mu1) / 2).sum(-1)
        pred = (np.einsum('tsn,sn->ts', Xs[te], w) > thr).astype(int)
        acc += (pred == lab[te][:, None]).mean(0) * te.sum()
    return acc / len(lab)


def panel_task_decoding(n_per_type=10, n_sub=100, size=5):
    U, P, cuts = unit_table(RUN); g = load('noise_sweep'); kk = g['macaque']['best_sigma']; ep = load('task_decoding_epochs')
    d = os.path.join(CACHE, 'task_decoding_rates'); seeds = list(feats()['seeds'])
    rng = np.random.default_rng(7); chosen = []
    allp = pairs_of(list(feats()['names']))
    for t in TYPES:
        cand = [p for p in allp if Q.relation(p[0], p[1], 'shared') == t]
        chosen += [cand[i] for i in rng.choice(len(cand), min(n_per_type, len(cand)), replace=False)]
    R = {}
    load_t = lambda t: R[t] if t in R else R.setdefault(t, dict(np.load(os.path.join(d, f'{t.replace(" ", "_")}.npz'))))
    rows = []; t0 = time.time()
    for a, b in chosen:
        A, B = load_t(a), load_t(b); w = Q.primary_window(a, b)
        for s in seeds:
            u = U[(U.task_a == a) & (U.task_b == b) & (U.seed == s)]
            if not len(u):
                continue
            k = seeds.index(s); ti, td = u.unit[u.cls_ti == 'TI'].to_numpy(), u.unit[u.cls_ti == 'TD'].to_numpy()
            if len(ti) < size or len(td) < size:
                continue
            # noise: kk x each unit's within-condition SD in the readout window (as panel g), independent per bin and trial
            Xa, Xb = A[w][k].astype(float), B[w][k].astype(float); ya, yb = A['y'][k], B['y'][k]
            sd = np.sqrt(np.mean([X[yy == c].var(0) for X, yy in ((Xa, ya), (Xb, yb)) for c in (0, 1) if (yy == c).sum() > 1], 0))
            ba, bb = A['bins'][k].astype(float), B['bins'][k].astype(float)              # (trials, bins, N)
            nr = np.random.default_rng([s, zlib.crc32(f'{a}|{b}'.encode())])
            Xall = np.concatenate([ba, bb], 0); Xall = Xall + kk * sd * nr.standard_normal(Xall.shape)
            lab = np.concatenate([np.zeros(len(ba), int), np.ones(len(bb), int)])
            sub = {c: np.stack([nr.choice(pool, size, replace=False) for _ in range(n_sub)]) for c, pool in (('TI', ti), ('TD', td))}
            for i in range(Xall.shape[1]):
                Xi = Xall[:, i]
                if np.isnan(Xi).any():
                    continue
                Xi = (Xi - Xi.mean(0)) / np.where(Xi.std(0) > 0, Xi.std(0), 1)       # label-agnostic z-score per bin
                for c in ('TI', 'TD'):
                    rows.append(dict(task_a=a, task_b=b, seed=s, cls=c, bin=i, acc=float(_cv_md(Xi, lab, sub[c], seed=i).mean())))
    T = pd.DataFrame(rows); T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    save('task_decoding', dict(table=T, pairs=chosen, sigma=kk, epochs=ep))
    print(f'j: {T[["task_a", "task_b"]].drop_duplicates().shape[0]} pairs, {len(T)} rows in {time.time() - t0:.0f}s', flush=True)


# ═══ silencing: TI, TD, Int and random choice units, on the pair's own tasks and (for the panel-j pairs) on all 27 tasks ═══
def _silence_conditions(Up, seeds, K, N, rng, n_rand):
    from ..silencing import _mask
    by = {s: g for s, g in Up.groupby('seed')}
    pools = {c: [] for c in ('TI', 'TD', 'Int', 'all')}
    for s in seeds:
        g = by.get(s)
        for c in pools:
            pools[c].append(np.array([], int) if g is None else (g.unit.to_numpy() if c == 'all' else g.unit[g.cls_ti == c].to_numpy()))
    n = [min(len(a), len(b)) for a, b in zip(pools['TI'], pools['TD'])]
    pick = lambda c, k: rng.choice(pools[c][k], n[k], replace=False) if n[k] >= MIN_N and len(pools[c][k]) >= n[k] else None
    out = [('intact', 0, np.ones((K, N), np.float32))]
    for c, nd in (('TI', 1), ('TD', 1), ('Int', n_rand), ('all', n_rand)):
        for d in range(nd):
            out.append((f'{"random" if c == "all" else c}_n', d, _mask(K, N, [pick(c, k) for k in range(K)])))
    return out, n


def panel_silencing(device=None, n_rand=4, n_trials=200):
    device = pick_device(device)
    import torch
    from ..silencing import Runner
    U, P, cuts = unit_table(RUN); R = Runner(RUN, device=device, M=4, n_trials=n_trials); K, N, seeds = R.K, R.N, R.seeds
    names = R.bat.names; jp = set(map(tuple, load('task_decoding')['pairs'])); parts = os.path.join(CACHE, 'silencing_parts'); os.makedirs(parts, exist_ok=True)
    Ug = {k: g for k, g in U.groupby(['task_a', 'task_b'])}; t0 = time.time()
    for i, (a, b) in enumerate(pairs_of(names)):
        pf = os.path.join(parts, f'{i:03d}.parquet')
        if os.path.exists(pf) or (a, b) not in Ug:
            continue
        Up = Ug[(a, b)]; rng = np.random.default_rng([i, 2027])
        conds, n = _silence_conditions(Up, seeds, K, N, rng, n_rand)
        tasks = names if (a, b) in jp else [a, b]; rows = []
        for t in tasks:
            vals = np.concatenate([R.perf(t, [c[2] for c in conds[s:s + R.M]], torch_seed=s) for s in range(0, len(conds), R.M)])
            for j, (c, d, m) in enumerate(conds):
                for k, s in enumerate(seeds):
                    if s in Up.seed.values and n[k] >= MIN_N:
                        rows.append(dict(task_a=a, task_b=b, seed=s, cond=c, draw=d, task=t, own=t in (a, b), n_removed=n[k], perf=float(vals[j, k])))
        pd.DataFrame(rows).to_parquet(pf)
        if i % 20 == 0:
            print(f'  silence {i + 1}/351 {a} vs {b}: {time.time() - t0:.0f}s, MPS {torch.mps.driver_allocated_memory() / 1e9 if R.dev.type == "mps" else 0:.1f} GB', flush=True)
    T = pd.concat([pd.read_parquet(os.path.join(parts, f)) for f in sorted(os.listdir(parts)) if f.endswith('.parquet')], ignore_index=True)
    T = T.groupby(['task_a', 'task_b', 'seed', 'cond', 'task', 'own'], as_index=False)[['perf', 'n_removed']].mean()
    base = T[T.cond == 'intact'].set_index(['task_a', 'task_b', 'seed', 'task']).perf
    T['d_perf'] = T.perf.values - base.reindex(pd.MultiIndex.from_frame(T[['task_a', 'task_b', 'seed', 'task']])).values
    T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    T['role'] = [role(t, ty) if own else 'other task' for t, ty, own in zip(T.task, T.type, T.own)]
    save('silencing', dict(table=T, cuts=cuts, all27_pairs=sorted(jp), n_trials=n_trials, n_rand=n_rand))
    print(f'silence: {len(T)} rows in {time.time() - t0:.0f}s', flush=True)
