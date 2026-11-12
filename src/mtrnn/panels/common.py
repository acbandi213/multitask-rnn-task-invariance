"""Shared by the panel computations of Extended Data Figures 7 and 8: constants, the result store, unit tables,
held-out TI / TD labels and the trial simulation.

Definitions (those of the main RNN figure, analysis/figure_5.ipynb):
    choice label   hemifield of the network's own response (readout features, errors included, no-go trials left out)
    window         the pair's primary window: last 300 ms before go ('pre') when both tasks carry the choice then, else 100-400 ms
                   after go ('resp')
    index          lib/index.py fast single-unit estimator (threshold fit in one task, scored in the other, both directions)
    choice units   active units with within-task accuracy >= 0.55 in either task (readout `group`)
    exclusion      a network is left out of every pair containing a task it performs below 0.90 (Yang et al. 2019's criterion,
                   applied per network-task combination: applied per network it would remove 9 of 10 ReLU networks, all for DNMS)
    TI / TD        top / bottom 20 % of the index pooled over all pairs and the networks that perform both tasks
    held-out       labels from one interleaved half of the trials (make_halves seed 0), decoding on the other, both directions
                   averaged (heldout.py)
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

PAPER_WINDOW = 'pre_2_1'                 # mouse analysis window of the recordings' reference values (reference/three_point.csv)

CACHE = PANELS                         # results/panels/


RUN, RUN_SP = 'main_relu', 'main_softplus'


TYPES = ['same operation', 'different operation', 'spatial vs abstract', 'reversed mapping']


ANALOG = {'mouse': ('Go Mod1', 'Go Mod2'), 'macaque': ('Sac', 'Pursuit')}


N_JOBS = 8


MIN_N = PH.MIN_N                       # 6: smallest size-matched population


MIN_PER = 10                           # smallest trials per (task x choice) condition for the XOR decoder


MIN_PERF = 0.90                        # network-task exclusion (Yang et al. 2019); the main figure uses the same rule


def ok_fn(run, thr=MIN_PERF):
    tp = Q.task_perf(run)
    return lambda s, a, b: tp[(int(s), a)] >= thr and tp[(int(s), b)] >= thr


def role(t, typ):
    # which task of a changed-response pair a task is: 'anti' / 'pro' (reversed mapping), 'abstract' / 'spatial' (spatial vs abstract)
    if typ == 'reversed mapping':
        return 'anti' if Q.MAPPING[t] == 'anti' else 'pro'
    if typ == 'spatial vs abstract':
        return 'abstract' if Q.MAPPING[t] == 'abstract' else 'spatial'
    return 'task 1 or 2'


def save(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f'{name}.pkl'), 'wb') as f:
        pickle.dump(obj, f)
    print(f'  -> results/panels/{name}.pkl', flush=True)


def load(name):
    with open(os.path.join(CACHE, f'{name}.pkl'), 'rb') as f:
        return pickle.load(f)


def pairs_of(names):
    return list(itertools.combinations(names, 2))


def unit_table(run, thr=MIN_PERF):
    """Choice units of every (pair, network) with the network-exclusion rule, pair type and TI / Int / TD (whole-session index)."""
    ok = ok_fn(run, thr)
    P = pd.read_parquet(os.path.join(Q.OUT, run, 'pairs.parquet')); P = P[P.primary]
    P = P[[ok(s, a, b) for s, a, b in zip(P.seed, P.task_a, P.task_b)]].copy()
    P['type'] = [Q.relation(a, b, 'shared') for a, b in zip(P.task_a, P.task_b)]
    U = pd.read_parquet(os.path.join(Q.OUT, run, 'units.parquet'))
    U = U.merge(P[['task_a', 'task_b', 'window', 'seed', 'type']], on=['task_a', 'task_b', 'window', 'seed']); U = U[U.group].copy()
    lo, hi = U.snCCGP.quantile(.2), U.snCCGP.quantile(.8)
    U['cls_ti'] = np.where(U.snCCGP >= hi, 'TI', np.where(U.snCCGP <= lo, 'TD', 'Int'))
    return U, P, (float(lo), float(hi))


_F = {}


def feats(run=RUN):
    if run not in _F:
        _F[run] = Q.load_features(run)
    return _F[run]


def sub_feats(F, a, b):
    return {key: F[key] for key in F if key.split('|')[0] in (a, b)}


# ═══ held-out labels (shared by c, f, i) ════════════════════════════════════════════════════════════════════════════════
def heldout_labels(run=RUN):
    """heldout._label_pass for every pair: label-half index and group per (network, direction). Cached."""
    path = os.path.join(CACHE, f'heldout_labels_{run}.pkl')
    if os.path.exists(path):
        return load(f'heldout_labels_{run}')
    from joblib import Parallel, delayed
    F = feats(run); names = list(F['names']); seeds = list(F['seeds']); prs = pairs_of(names); ok = ok_fn(run)
    keep = {p: [k for k, s in enumerate(seeds) if ok(s, p[0], p[1])] for p in prs}
    L = Parallel(n_jobs=N_JOBS)(delayed(PH._label_pass)(sub_feats(F, a, b), a, b, keep[(a, b)]) for a, b in prs)
    out = dict(pairs=prs, labels=L, seeds=seeds)
    save(f'heldout_labels_{run}', out)
    return out


def heldout_cuts(L, prop):
    cuts = {}
    L = [LL for LL in L if LL]
    for d in (0, 1):
        v = np.concatenate([l['idx'][l['group']] for LL in L for l in LL if l['dir'] == d])
        cuts[d] = (float(np.quantile(v, prop)), float(np.quantile(v, 1 - prop)))
    return cuts


# ═══ c · threshold sweep (TI-only minus TD-only, size-matched, held-out) ════════════════════════════════════════════════
def heldout_table(run, prop):
    """heldout.py's decode pass on the 0.90-excluded labels -> one row per (pair, network) (both directions averaged)."""
    from joblib import Parallel, delayed
    H = heldout_labels(run); F = feats(run); cuts = heldout_cuts(H['labels'], prop)
    rows = Parallel(n_jobs=N_JOBS)(delayed(PH._decode_pass)(sub_feats(F, a, b), a, b, L, cuts, H['seeds'])
                                   for (a, b), L in zip(H['pairs'], H['labels']) if L)
    T = pd.DataFrame([r for R in rows for r in R])
    T = T.groupby(['task_a', 'task_b', 'seed']).mean(numeric_only=True).drop(columns='dir').reset_index()
    T['type'] = [Q.relation(a, b, 'shared') for a, b in zip(T.task_a, T.task_b)]
    return T, cuts


# ═══ GPU: time-resolved activity for j, features at training checkpoints ════════════════════════════════════════════════
J_BIN, J_PRE, J_POST, J_TRIALS = 5, 75, 25, 240          # 100-ms bins from 1.5 s before go to 0.5 s after it


def _simulate(net, bat, cfg, device, t, n, seed, binned=False):
    import torch
    from ..train import run_trials
    from ..tasks import hemifield, popvec, ring_slice, GRACE
    rng = np.random.default_rng([seed, 17, bat.names.index(t)])
    P = bat.sample(rng, n * (2 if t in Q.NOGO else 1), task=t, test=True)
    keep = np.flatnonzero(P['respond'])
    P = {k: (v[keep] if k != 'meta' else {m: a_[keep] for m, a_ in v.items()}) for k, v in P.items()}
    r0, T = int(P['resp_on'][0]), int(P['T'][0]); nb = (J_PRE + J_POST) // J_BIN
    Xw = {w: [] for w in Q.WINDOWS}; Y, XB = [], []
    for s in range(0, len(keep), 80):
        sub = {k: (v[s:s + 80] if k != 'meta' else None) for k, v in P.items()}
        with torch.no_grad():
            Z, R = run_trials(net, bat, sub, device, noise=True, keep_rates=True, sigma_in=cfg['sigma_in'], chunk=80)
        for w, (a, b) in Q.WINDOWS.items():
            Xw[w].append(R[:, :, r0 + a:r0 + b].mean(2).astype(np.float16))
        ring = Z[:, :, r0 + GRACE:T, ring_slice(int(P['resp_ring'][0]))].mean(2)
        Y.append(hemifield(popvec(ring)).astype(np.int8))
        if binned:
            B = np.full((R.shape[0], R.shape[1], nb, R.shape[3]), np.nan, np.float16)
            for i in range(nb):
                t0, t1 = r0 - J_PRE + i * J_BIN, r0 - J_PRE + (i + 1) * J_BIN
                if t0 >= 0 and t1 <= R.shape[2]:
                    B[:, :, i] = R[:, :, t0:t1].mean(2)
            XB.append(B)
        del Z, R
        if device == 'mps':
            torch.mps.empty_cache()
    out = {f'{t}|{w}': np.concatenate(Xw[w], 1) for w in Q.WINDOWS}
    out[f'{t}|y'] = np.concatenate(Y, 1)
    epochs = dict(go=r0, fix_off=int(P['fix_off'][0]), ev_on=np.asarray(P['ev_on'][0]).tolist(), ev_off=np.asarray(P['ev_off'][0]).tolist(),
                  n_ev=int(np.asarray(P['n_ev'])[0]) if 'n_ev' in P else None)
    if binned:
        out[f'{t}|bins'] = np.concatenate(XB, 1)
    return out, epochs
