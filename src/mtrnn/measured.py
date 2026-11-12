"""The mouse-like and monkey-like task pairs of the multitask networks, read exactly like the paper's data.

Pairs: mouse-like = Go Mod1 vs Go Mod2 (same answer, cue on a different input ring: the PPC visual / auditory task pair);
monkey-like = Sac vs Pursuit (memory-guided saccade vs step-ramp pursuit: the FEF pair), in both output layouts.

Per network (the analogue of one recording session):
    session    the measured read-out of readout.py: data trial counts (mouse 50, FEF 150 per task x choice cell) and measurement noise
               (k x each unit's within-condition SD; k from results/readout/<run>/measured.parquet, calibrated there only on the
               within-task accuracy of held-out-gated units, PPC 0.600 / FEF 0.565)
    index      lib.population.unit_index: the paper's snCCGP on the whole session and on the label half of 10
               interleaved splits; group = within-task accuracy >= 0.55 (that module's rule); TI / TD = top / bottom 20 % pooled
               over the 10 networks of a condition (the analogue of pooling the sessions of a species)
    decoding   lib.population.decode_job on the decode half: full group within / across (the paper's cd_decode),
               size-matched TI-only / TD-only / random (Fig. 5 three-point), removal of TI / TD vs the same number of random
               units, XOR / task / choice (balanced four-condition decoder)
    ftv        gain-free FTV and signed d' per unit (whole session): one-task (|FTV_d| >= 0.5) vs reversed TD units
    sessions   data-sized "sessions": random subsets of the group (mouse 100, FEF 43 units; the data's median group sizes are
               172 / 43, the mouse subset is capped below the networks' group size), %TI from the split-0 label half, across-task
               decoding of the subset on the other half; 10 subsets per network

    python -m mtrnn.measured            -> results/measured/<run>__<pair>/{index,decode,ftv,sessions}.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import readout as Q

from .lib import population as A
from .lib.measures import dprime2
from .paths import MEASURED

OUT = MEASURED                         # results/measured/<run>__<pair>/
PAIRS = {'mouse_like': dict(tasks=('Go Mod1', 'Go Mod2'), row='PPC', n_cell=50, sub=100),
         'monkey_like': dict(tasks=('Sac', 'Pursuit'), row='FEF', n_cell=150, sub=43),
         'mouse_tmaze': dict(tasks=('T-maze Vis', 'T-maze Aud'), row='PPC T-maze', n_cell=50, sub=100)}   # only in runs with the T-maze tasks
RUNS = ['main_relu', 'main_softplus']
PROP = 0.20
N_SUB = 10


def make_session(F, run, pair, k, obs_k, seed=7):
    a, b = PAIRS[pair]['tasks']; w = Q.primary_window(a, b)
    X, y, ctx, act = Q.measured_session(F, a, b, w, k, PAIRS[pair]['n_cell'], obs_k, seed)
    return dict(X=X, y=y, ctx=ctx, mean_rate=np.where(act, np.abs(X).mean(0), 0.0))


def index_job(sess, seed):
    I = A.unit_index(sess, -np.ones(sess['X'].shape[1], int)); I['seed'] = seed
    return I


def decode_and_more(sess, I, seed, sub):
    D = A.decode_job(sess, I, seed); D['seed'] = seed
    X, y, ctx = sess['X'], sess['y'], sess['ctx']
    q1, d1 = dprime2(X, y, ctx, 1); q2, d2 = dprime2(X, y, ctx, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        ftv_d = np.where(q1 + q2 > 0, (q1 - q2) / (q1 + q2), np.nan)
    a = I[I.half == 'all'].set_index('unit').sort_index()
    Fv = pd.DataFrame(dict(seed=seed, unit=a.index, group=a.group.to_numpy(), cls_all=a.cls.to_numpy(), snCCGP=a.snCCGP.to_numpy(),
                           sn_within=a.sn_within.to_numpy(), ftv_d=ftv_d, d1=d1, d2=d2))
    # data-sized sessions: label half = split 0 half A, decode half B
    lab = I[(I['mode'] == 'interleaved') & (I.split == 0) & (I.half == 'A')].set_index('unit').cls
    g = np.flatnonzero(a.group.to_numpy())
    Ah, Bh = A.make_halves(y, ctx, seed=0, mode='interleaved')
    rng = np.random.default_rng([seed, 4242]); rows = []
    for s in range(N_SUB):
        cols = rng.choice(g, min(sub, len(g)), replace=False)
        r = A.cd_decode(A.zctx(X[Bh][:, cols], ctx[Bh]), y[Bh], ctx[Bh])
        c = lab.loc[cols]
        rows.append(dict(seed=seed, sub=s, n_units=len(cols), pct_TI=float((c == 'TI').mean()), pct_TD=float((c == 'TD').mean()),
                         within=np.nan if r is None else r['within_cd'], across=np.nan if r is None else r['across_cd']))
    return D, Fv, pd.DataFrame(rows)


def run_condition(run, pair, n_jobs=10):
    from joblib import Parallel, delayed
    F = Q.load_features(run); seeds = list(F['seeds'])
    a, b = PAIRS[pair]['tasks']
    M = pd.read_parquet(os.path.join(Q.OUT, run, 'measured.parquet'))
    obs_k = float(M[M.row == PAIRS[pair]['row']].obs_k.iloc[0])
    keep = Q.ok_seeds(run, a, b, seeds)
    sess = {seeds[k]: make_session(F, run, pair, k, obs_k) for k in keep}
    Is = Parallel(n_jobs=n_jobs)(delayed(index_job)(sess[s], s) for s in sess)
    cuts = A.pooled_cuts(Is, prop_ti=PROP, prop_td=PROP)
    Is = [A.apply_cuts(I, cuts) for I in Is]
    res = Parallel(n_jobs=n_jobs)(delayed(decode_and_more)(sess[I.seed.iloc[0]], I, int(I.seed.iloc[0]), PAIRS[pair]['sub']) for I in Is)
    od = os.path.join(OUT, f'{run}__{pair}'); os.makedirs(od, exist_ok=True)
    pd.concat(Is).to_parquet(os.path.join(od, 'index.parquet'))
    for name, j in (('decode', 0), ('ftv', 1), ('sessions', 2)):
        pd.concat([r[j] for r in res], ignore_index=True).to_parquet(os.path.join(od, f'{name}.parquet'))
    cuts.to_parquet(os.path.join(od, 'cuts.parquet'))
    D = pd.concat([r[0] for r in res]); f = D[D.set == 'full']
    print(f'{run} {pair}: k = {obs_k:.2f}, {len(keep)} networks, within {f.within_cd.mean():.3f}, across {f.across_cd.mean():.3f}', flush=True)


if __name__ == '__main__':
    # python -m mtrnn.measured [run ...]   (default: the two main sets of networks); pairs a run lacks are skipped
    for run in (sys.argv[1:] or RUNS):
        names = list(Q.load_features(run)['names'])
        for pair in PAIRS:
            if all(t in names for t in PAIRS[pair]['tasks']):
                run_condition(run, pair)
