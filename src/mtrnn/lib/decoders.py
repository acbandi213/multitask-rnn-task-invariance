"""The paper's linear decoder, generalised from the choice dichotomy to any dichotomy of the four conditions.

The decoder is the coding-direction rule of `cd_decode` (analysis/paper/setup_population.py): weights = difference of
the two class means on the training trials, unit norm; threshold = midpoint of the two class-mean projections; a test
trial is assigned to class 1 if its projection exceeds the threshold. No regularisation, no free parameter.

`cd_decode` itself only handles choice within / across the two tasks, so the rule is re-expressed here on a BALANCED
condition tensor and batched over leading dimensions (neuron-subset draws, time bins):

    Xb  (..., 4, m, n)     4 conditions x m trials per condition x n neurons
    conditions             0 = T1-L, 1 = T1-R, 2 = T2-L, 3 = T2-R        (cond = 2 * (task - 1) + choice)

    choice  {T1-L, T2-L} vs {T1-R, T2-R}      task  {T1-L, T1-R} vs {T2-L, T2-R}      xor  {T1-L, T2-R} vs {T1-R, T2-L}

gate.validate() checks the batched rule against `cd_decode` on real data: the across-task accuracy must be identical
(same training set, no folds), the 5-fold within-task accuracy must agree to within 0.005 on the session mean (the fold
assignment differs: cd_decode uses StratifiedKFold(shuffle), here trials are drawn in random order and fold = position % 5).
"""
from __future__ import annotations

import numpy as np

COND = ('T1-L', 'T1-R', 'T2-L', 'T2-R')
DICH = {'choice': np.array([-1, 1, -1, 1]), 'task': np.array([-1, -1, 1, 1]), 'xor': np.array([-1, 1, 1, -1])}
# CCGP (Bernardi et al. 2020): train on one pair of conditions, test on the held-out pair, both directions.
#   choice: train within one task, test in the other          task: train on left trials, test on right trials (and back)
CCGP_SPLITS = {'choice': (((0, 1), (2, 3)), ((2, 3), (0, 1))),
               'task': (((0, 2), (1, 3)), ((1, 3), (0, 2)))}
N_FOLDS = 5


def conditions(y, ctx):
    return 2 * (np.asarray(ctx, int) - 1) + np.asarray(y, int)


def zpool(X):
    """z-score every neuron over all trials given (label-agnostic; keeps the task and choice main effects)."""
    Z = np.array(X, float, copy=True)
    mu = Z.mean(0); sd = Z.std(0); sd[sd == 0] = 1.0
    return (Z - mu) / sd


def balanced_index(cond, m, rng, n_draws):
    """(n_draws, 4, m) trial indices: m trials per condition without replacement, in random order."""
    pools = [np.flatnonzero(cond == c) for c in range(4)]
    idx = np.empty((n_draws, 4, m), int)
    for d in range(n_draws):
        for c in range(4):
            idx[d, c] = rng.permutation(pools[c])[:m]
    return idx


def gather(X, idx, cols):
    """X (trials, n_all) -> (D, 4, m, n). cols: 1-D (one neuron set for every draw) or (D, n) (one set per draw)."""
    cols = np.asarray(cols, int)
    if cols.ndim == 1:
        return X[:, cols][idx]
    return X[idx[..., None], cols[:, None, None, :]]


def cd_rule(mu1, mu0):
    """Class means (..., n) on the training trials -> unit-norm weights, midpoint threshold, validity flag."""
    w = mu1 - mu0
    nrm = np.linalg.norm(w, axis=-1, keepdims=True)
    ok = nrm[..., 0] > 0
    w = np.divide(w, nrm, out=np.zeros_like(w), where=nrm > 0)
    thr = 0.5 * ((w * mu1).sum(-1) + (w * mu0).sum(-1))
    return w, thr, ok


def _acc(X0, X1, w, thr, ok):
    """Accuracy of rule (w, thr) on test trials X0 (true class 0) and X1 (true class 1), each (..., k, m, n) with equal m."""
    p0 = np.einsum('...kmn,...n->...km', X0, w) > thr[..., None, None]
    p1 = np.einsum('...kmn,...n->...km', X1, w) > thr[..., None, None]
    acc = 0.5 * ((~p0).mean((-1, -2)) + p1.mean((-1, -2)))
    return np.where(ok, acc, np.nan)


def within_dichotomies(Xb, dich=('choice', 'task', 'xor'), n_folds=N_FOLDS):
    """5-fold cross-validated accuracy of each dichotomy with all four conditions included. Returns {name: (...)}."""
    m = Xb.shape[-2]
    fold = np.arange(m) % n_folds
    out = {k: [] for k in dich}
    for f in range(n_folds):
        te = fold == f
        mu = Xb[..., ~te, :].mean(-2)                       # (..., 4, n) condition means on the training trials
        Xte = Xb[..., te, :]                                # (..., 4, m_te, n)
        for k in dich:
            s = DICH[k]
            w, thr, ok = cd_rule(mu[..., s > 0, :].mean(-2), mu[..., s < 0, :].mean(-2))
            out[k].append(_acc(Xte[..., s < 0, :, :], Xte[..., s > 0, :, :], w, thr, ok))
    with np.errstate(invalid='ignore'):
        return {k: np.nanmean(v, axis=0) for k, v in out.items()}


def within_task_choice(Xb, n_folds=N_FOLDS):
    """5-fold choice accuracy inside each task, averaged over the two tasks (the balanced analogue of cd_decode's `within`)."""
    m = Xb.shape[-2]
    fold = np.arange(m) % n_folds
    accs = []
    for a0, a1 in ((0, 1), (2, 3)):
        fa = []
        for f in range(n_folds):
            te = fold == f
            mu = Xb[..., ~te, :].mean(-2)
            w, thr, ok = cd_rule(mu[..., a1, :], mu[..., a0, :])
            fa.append(_acc(Xb[..., [a0], :, :][..., te, :], Xb[..., [a1], :, :][..., te, :], w, thr, ok))
        with np.errstate(invalid='ignore'):
            accs.append(np.nanmean(fa, axis=0))
    return np.mean(accs, axis=0)


def ccgp(Xb, which=('choice', 'task')):
    """Cross-condition generalisation: train on one pair of conditions (all m trials), test on the held-out pair."""
    mu = Xb.mean(-2)
    out = {}
    for k in which:
        accs = []
        for (a0, a1), (b0, b1) in CCGP_SPLITS[k]:
            w, thr, ok = cd_rule(mu[..., a1, :], mu[..., a0, :])
            accs.append(_acc(Xb[..., [b0], :, :], Xb[..., [b1], :, :], w, thr, ok))
        out[k] = np.mean(accs, axis=0)          # NaN if either direction has no rule (cd_decode returns None there)
    return out


def geometry(Xb):
    """All population quantities of one balanced tensor: within-dichotomy accuracies, shattering dimensionality,
    CCGP(choice), CCGP(task), within-task choice accuracy."""
    w = within_dichotomies(Xb)
    c = ccgp(Xb)
    return dict(acc_choice=w['choice'], acc_task=w['task'], acc_xor=w['xor'], sd=(w['choice'] + w['task'] + w['xor']) / 3.0,
                ccgp_choice=c['choice'], ccgp_task=c['task'], wt_choice=within_task_choice(Xb))
