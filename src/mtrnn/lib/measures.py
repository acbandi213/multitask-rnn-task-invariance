"""Choice selectivity per task, and measurement noise.

add_obs_noise adds independent Gaussian noise to every unit on every trial, SD = k x the unit's within-condition SD, so that a noise-free
network unit can be read at the reliability of a recorded neuron.
"""
from __future__ import annotations

import numpy as np


def dprime2(X, y, ctx, c):
    """Gain-free version: squared choice d' in task c minus its sampling bias (1 / n_R + 1 / n_L), floored at 0, and the signed d'."""
    m = ctx == c
    a, b = X[m & (y == 1)], X[m & (y == 0)]
    sd = np.sqrt(0.5 * (a.var(0, ddof=1) + b.var(0, ddof=1))); sd[sd == 0] = np.nan
    d = (a.mean(0) - b.mean(0)) / sd
    return np.clip(d ** 2 - (1 / len(a) + 1 / len(b)), 0, None), d


def add_obs_noise(X, y, ctx, k, seed):
    if not k:
        return X
    R = X.copy()
    for c in np.unique(ctx):
        for v in (0, 1):
            m = (ctx == c) & (y == v)
            if m.any():
                R[m] -= X[m].mean(0)
    sd = R.std(0)
    return X + k * sd[None, :] * np.random.default_rng([seed, 99]).standard_normal(X.shape)
