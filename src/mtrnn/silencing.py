"""Follow-up Q1: removal of TI / TD / random units and TASK PERFORMANCE (behaviour), for every task pair and network.

Labels: the unit index of `units.parquet` (readout.py, whole session, the pair's primary window), choice units (`group`), TI / TD
= top / bottom 20 % of the index pooled over all pairs and the networks that perform both tasks of a pair (exactly the cuts of
`RNN_figure_main.ipynb`, `D[act]['cuts']`). Performance is measured on newly simulated trials, so labelling on the whole session
does not reuse the test trials.

Removal = silencing: the unit's rate is multiplied by 0 at every time step (Yang's lesion, `BatchedYangRNN(silence=...)`).
Performance = `Q.task_perf`'s metric: Yang's rule at the last step, training-distribution trials of one task, noise on
(N_TRIALS per task, the same trial set for every pair and condition).

Conditions per pair and network:
    intact
    TI_n / TD_n / random_n     size-matched: n = min(#TI, #TD) of the pair (the rule behind `n_subset`), n >= 6; random = n choice
                               units, N_RAND draws
    TI_all / TD_all            every TI (TD) unit of the pair, with random_TIall / random_TDall = the same number of random choice
                               units (N_RAND draws): the behavioural analogue of `rem_TI` / `rem_TD` in heldout_pairs.parquet
    cumulative                 k units removed in order of the index: TD-first (lowest first), TI-first (highest first), random
                               (N_RAND_CUM draws); k on K_GRID (all pairs) or K_FINE with N_RAND_FINE draws (the two analog pairs)

    python -m mtrnn.silencing main_relu   -> results/readout/<run>/silencing/q1_lesion_perf.parquet, q1_cumulative.parquet
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd
import torch

from . import readout as Q
from .model import BatchedYangRNN
from .tasks import render, performance
from .paths import pick_device
from .train import RUNS, load_nets

N_TRIALS = 200
MIN_N = 6
N_RAND = 4
N_RAND_CUM = 2
N_RAND_FINE = 6
K_GRID = [8, 16, 32, 48, 64, 96, 128]
K_FINE = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160]
ANALOG = {'mouse-like': ('Go Mod1', 'Go Mod2'), 'monkey-like': ('Sac', 'Pursuit')}
M_COPIES = 4                        # conditions run at once (the K networks are replicated M times); kept small: memory


def out_dir(run):
    d = os.path.join(Q.OUT, run, 'silencing'); os.makedirs(d, exist_ok=True); return d


def labels(run):
    """Choice units of every (pair, network) in the primary window with the pooled cuts. -> (U, (lo, hi)); U has `ok` (network
    performs both tasks) and `cls_ti` in TI / Int / TD."""
    tp = Q.task_perf(run)
    U = pd.read_parquet(os.path.join(Q.OUT, run, 'units.parquet'))
    U = U[U.group & (U.window == [Q.primary_window(a, b) for a, b in zip(U.task_a, U.task_b)])].copy()
    U['ok'] = [tp[(int(s), a)] >= Q.MIN_TASK_PERF and tp[(int(s), b)] >= Q.MIN_TASK_PERF for s, a, b in zip(U.seed, U.task_a, U.task_b)]
    v = U.loc[U.ok, 'snCCGP']; lo, hi = v.quantile(.2), v.quantile(.8)
    U['cls_ti'] = np.where(U.snCCGP >= hi, 'TI', np.where(U.snCCGP <= lo, 'TD', 'Int'))
    return U, (float(lo), float(hi))


class Runner:
    """The K networks replicated M times; perf(task, masks) runs one trial set of a task through up to M (K, N) masks at once."""

    def __init__(self, run, device=None, M=M_COPIES, n_trials=N_TRIALS, seed=777, tag='model', sigma_rec=None):
        device = pick_device(device)
        net, bat, cfg, _ = load_nets(os.path.join(RUNS, run), tag=tag, device='cpu')
        self.K, self.N, self.M, self.dev, self.bat, self.cfg = net.K, net.N, M, torch.device(device), bat, cfg
        self.seeds = list(net.seeds)
        self.big = BatchedYangRNN.stack([net.extract(k) for _ in range(M) for k in range(net.K)])
        if sigma_rec is not None:
            self.big.sigma_rec = float(sigma_rec)
        self.big = self.big.to(self.dev)
        rng = np.random.default_rng([seed, 31])
        self.P = {t: bat.sample(rng, n_trials, task=t) for t in bat.names}          # training distribution, as Q.task_perf
        self.chunk = 50

    @torch.no_grad()
    def perf(self, task, masks, torch_seed=0):
        """masks: list of (K, N) 0/1 arrays (<= M). -> (len(masks), K) fraction correct."""
        m = len(masks)
        full = np.ones((self.M, self.K, self.N), np.float32); full[:m] = np.stack(masks)
        sil = torch.as_tensor(full.reshape(self.M * self.K, self.N), device=self.dev)
        P = self.P[task]; n = len(P['T']); Tm = int(P['T'].max()); last = P['T'] - 1
        torch.manual_seed(torch_seed)
        ok = np.zeros((self.M * self.K, n), bool)
        for s in range(0, n, self.chunk):
            sub = {k: (v[s:s + self.chunk] if k != 'meta' else None) for k, v in P.items()}
            d = render(sub, self.bat.n_rules, self.dev, sigma_in=self.cfg['sigma_in'], alpha=self.big.alpha, noise=True, T_pad=Tm)
            u = d['inputs'][None].expand(self.big.K, *d['inputs'].shape)
            _, Z = self.big(u, noise=True, silence=sil, keep_rates=False)
            li = torch.as_tensor(last[s:s + self.chunk], device=self.dev)
            zl = Z[:, torch.arange(len(li), device=self.dev), li].cpu().numpy()          # (MK, b, O) at each trial's last step
            for j in range(self.M * self.K if m == self.M else m * self.K):
                ok[j, s:s + len(li)] = performance(zl[j], sub['respond'], sub['resp_dir'], sub['resp_ring'])
        del Z, u, d
        if self.dev.type == 'mps':
            torch.mps.empty_cache()                 # trial lengths differ by task; without this the MPS cache grows without bound
        return ok.mean(1).reshape(self.M, self.K)[:m]


def _mask(K, N, removed):
    """removed: list over networks of unit arrays (None = nothing)."""
    m = np.ones((K, N), np.float32)
    for k, r in enumerate(removed):
        if r is not None and len(r):
            m[k, np.asarray(r, int)] = 0
    return m


def pair_conditions(Up, seeds, K, N, rng):
    """Size-matched and full-tail conditions for one pair. -> list of (cond, draw, mask, n_removed per network)."""
    by = {s: g for s, g in Up.groupby('seed')}
    ti, td, grp = [], [], []
    for s in seeds:
        g = by.get(s)
        if g is None:
            ti.append(np.array([], int)); td.append(np.array([], int)); grp.append(np.array([], int)); continue
        ti.append(g.unit[g.cls_ti == 'TI'].to_numpy()); td.append(g.unit[g.cls_ti == 'TD'].to_numpy()); grp.append(g.unit.to_numpy())
    n = [min(len(a), len(b)) for a, b in zip(ti, td)]
    nm = lambda arr: [len(x) if x is not None else 0 for x in arr]
    out = [('intact', 0, np.ones((K, N), np.float32), [0] * K)]
    sm = [i if n[k] >= MIN_N else None for k, i in enumerate(ti)]
    out.append(('TI_n', 0, _mask(K, N, [None if x is None else rng.choice(x, n[k], replace=False) for k, x in enumerate(sm)]), [n[k] if n[k] >= MIN_N else 0 for k in range(K)]))
    sm = [i if n[k] >= MIN_N else None for k, i in enumerate(td)]
    out.append(('TD_n', 0, _mask(K, N, [None if x is None else rng.choice(x, n[k], replace=False) for k, x in enumerate(sm)]), [n[k] if n[k] >= MIN_N else 0 for k in range(K)]))
    for d in range(N_RAND):
        r = [rng.choice(grp[k], n[k], replace=False) if n[k] >= MIN_N else None for k in range(K)]
        out.append(('random_n', d, _mask(K, N, r), nm(r)))
    for cls, pool in (('TI', ti), ('TD', td)):
        out.append((f'{cls}_all', 0, _mask(K, N, pool), nm(pool)))
        for d in range(N_RAND):
            r = [rng.choice(grp[k], len(pool[k]), replace=False) if 0 < len(pool[k]) < len(grp[k]) else None for k in range(K)]
            out.append((f'random_{cls}all', d, _mask(K, N, r), nm(r)))
    return out, n


def cumulative_conditions(Up, seeds, K, N, rng, k_grid, n_rand):
    by = {s: g.sort_values('snCCGP') for s, g in Up.groupby('seed')}
    order = {'TD': [], 'TI': []}; rand = [[] for _ in range(n_rand)]
    for s in seeds:
        g = by.get(s); u = np.array([], int) if g is None else g.unit.to_numpy()
        order['TD'].append(u); order['TI'].append(u[::-1])
        for d in range(n_rand):
            rand[d].append(rng.permutation(u))
    out = []
    for kk in k_grid:
        for o in ('TD', 'TI'):
            out.append((o, 0, kk, _mask(K, N, [x[:kk] if len(x) >= kk else None for x in order[o]])))
        for d in range(n_rand):
            out.append(('random', d, kk, _mask(K, N, [x[:kk] if len(x) >= kk else None for x in rand[d]])))
    return out


def _run_conds(R, a, b, conds, key):
    """conds: list of (..., mask at position key). -> {task: (len(conds), K)}"""
    res = {}
    for t in (a, b):
        vals = []
        for s in range(0, len(conds), R.M):
            vals.append(R.perf(t, [c[key] for c in conds[s:s + R.M]], torch_seed=s))
        res[t] = np.concatenate(vals)
    return res


def run(run_name, device=None, pairs=None, analog_only=False, tag='model', sigma_rec=None, suffix=''):
    U, cuts = labels(run_name)
    R = Runner(run_name, device=device, tag=tag, sigma_rec=sigma_rec)
    K, N, seeds = R.K, R.N, R.seeds
    tp = Q.task_perf(run_name)
    names = R.bat.names
    pairs = pairs or list(itertools.combinations(names, 2))
    if analog_only:
        pairs = [p for p in pairs if p in ANALOG.values()]
    Ug = {k: g for k, g in U.groupby(['task_a', 'task_b'])}
    sfx = suffix + ('_analog' if analog_only else '')
    parts = os.path.join(out_dir(run_name), f'q1_parts{sfx}'); os.makedirs(parts, exist_ok=True)    # one file per pair: resumable
    t0 = time.time()
    for i, (a, b) in enumerate(pairs):
        pf = os.path.join(parts, f'{i:03d}.parquet')
        if os.path.exists(pf):
            continue
        rows, crows = [], []
        Up = Ug[(a, b)]
        rng = np.random.default_rng([i, 2026])
        conds, n_sm = pair_conditions(Up, seeds, K, N, rng)
        res = _run_conds(R, a, b, conds, 2)
        cnt = Up.groupby('seed').cls_ti.value_counts().unstack(fill_value=0)
        for j, (c, d, _, nrem) in enumerate(conds):
            for k, s in enumerate(seeds):
                base = dict(task_a=a, task_b=b, seed=s, ok=tp[(s, a)] >= Q.MIN_TASK_PERF and tp[(s, b)] >= Q.MIN_TASK_PERF, cond=c, draw=d,
                            n_removed=nrem[k], n_sm=n_sm[k], n_group=int(cnt.loc[s].sum()) if s in cnt.index else 0,
                            n_TI=int(cnt.loc[s].get('TI', 0)) if s in cnt.index else 0, n_TD=int(cnt.loc[s].get('TD', 0)) if s in cnt.index else 0)
                rows.append({**base, 'perf_a': res[a][j, k], 'perf_b': res[b][j, k]})
        fine = (a, b) in ANALOG.values()
        cc = cumulative_conditions(Up, seeds, K, N, rng, K_FINE if fine else K_GRID, N_RAND_FINE if fine else N_RAND_CUM)
        rc = _run_conds(R, a, b, cc, 3)
        for j, (o, d, kk, m) in enumerate(cc):
            for k, s in enumerate(seeds):
                if (m[k] == 0).sum() == kk:
                    crows.append(dict(task_a=a, task_b=b, seed=s, order=o, draw=d, k=kk, perf_a=rc[a][j, k], perf_b=rc[b][j, k]))
        pd.DataFrame(crows).to_parquet(pf.replace('.parquet', '_cum.parquet')); pd.DataFrame(rows).to_parquet(pf)
        if i % 10 == 0 or fine:
            mem = torch.mps.driver_allocated_memory() / 1e9 if R.dev.type == 'mps' else np.nan
            print(f'  {i + 1}/{len(pairs)} {a} vs {b}: {time.time() - t0:.0f}s, MPS memory {mem:.1f} GB', flush=True)
    fs = sorted(f for f in os.listdir(parts) if f.endswith('.parquet') and not f.endswith('_cum.parquet'))
    T = pd.concat([pd.read_parquet(os.path.join(parts, f)) for f in fs], ignore_index=True)
    C = pd.concat([pd.read_parquet(os.path.join(parts, f.replace('.parquet', '_cum.parquet'))) for f in fs], ignore_index=True)
    od = out_dir(run_name)
    T.to_parquet(os.path.join(od, f'q1_lesion_perf{sfx}.parquet')); C.to_parquet(os.path.join(od, f'q1_cumulative{sfx}.parquet'))
    print(f'{run_name}: cuts {cuts}; {len(pairs)} pairs in {time.time() - t0:.0f}s', flush=True)
    return T, C


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--device', default=None)
    p.add_argument('--analog_only', action='store_true'); p.add_argument('--n_pairs', type=int, default=None)
    a = p.parse_args()
    pairs = None
    if a.n_pairs:
        names = load_nets(os.path.join(RUNS, a.run))[1].names
        pairs = list(itertools.combinations(names, 2))[:a.n_pairs]
    run(a.run, a.device, pairs=pairs, analog_only=a.analog_only)
