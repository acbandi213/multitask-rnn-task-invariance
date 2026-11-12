"""Cross-task generalization index of single units (the estimator used for the recordings, copied unchanged).

For every unit (column of X) and the two tasks (ctx = 1 / 2), `_decode_matrix` returns a 2 x 2 accuracy matrix of a one-dimensional
nearest-centroid threshold classifier of the binary label y:
    diagonal      stratified k-fold cross-validation within a task (within-task accuracy)
    off-diagonal  threshold fitted on all trials of one task, scored on the other task
`_summarise` gives sn_within (mean of the diagonal) and snCCGP (mean of the two off-diagonal entries): the cross-task generalization
index. Nothing is folded about 0.5: a unit that reverses its preference between tasks scores below 0.5.
method='fast' is the vectorised threshold used everywhere here; any other value fits one LinearSVC per unit.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

CONTEXTS = (1, 2)


def _pipeline(C: float = 1.0) -> Pipeline:
    """The Fig. 3f classifier, unchanged."""
    return Pipeline([('scaler', StandardScaler()),
                     ('svm', LinearSVC(C=C, class_weight='balanced', max_iter=5000))])


def _folds(y, ctx, c, n_splits=5, seed=42):
    """Stratified folds within one context; returns list of (train_pos, test_pos) into the
    full-length arrays. Trial-wise by construction."""
    idx = np.flatnonzero(ctx == c)
    yy = y[idx]
    if len(np.unique(yy)) < 2:
        return []
    k = min(n_splits, int(np.bincount(yy).min()))
    if k < 2:
        return []
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(idx[tr], idx[te]) for tr, te in cv.split(idx.reshape(-1, 1), yy)]


def _centroid_acc(Xtr, ytr, Xte, yte, wtr=None, wte=None):
    """Vectorised nearest-centroid threshold. X: (n_trials, n_neurons) -> acc (n_neurons,)."""
    def wmean(X, m, w):
        if w is None:
            return X[m].mean(0)
        ww = w[m][:, None]
        return (X[m] * ww).sum(0) / ww.sum()

    m0, m1 = ytr == 0, ytr == 1
    if not m0.any() or not m1.any():
        return np.full(Xtr.shape[1], np.nan)
    c0, c1 = wmean(Xtr, m0, wtr), wmean(Xtr, m1, wtr)
    thr = (c0 + c1) / 2.0
    sgn = np.sign(c1 - c0)
    sgn[sgn == 0] = 1.0
    pred = (((Xte - thr) * sgn) > 0).astype(np.int8)
    ok = (pred == yte[:, None])
    if wte is None:
        return ok.mean(0)
    ww = wte[:, None]
    return (ok * ww).sum(0) / ww.sum()


def _svc_acc(Xtr, ytr, Xte, yte, wtr=None, wte=None, C=1.0):
    """One LinearSVC fit. X may be (n,1) for a single neuron or (n,p) for a population."""
    clf = _pipeline(C)
    if wtr is None:
        clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr, svm__sample_weight=wtr)
    return clf.score(Xte, yte) if wte is None else clf.score(Xte, yte, sample_weight=wte)


def _decode_matrix(X, y, ctx, method='fast', n_splits=5, seed=42, w=None, C=1.0):
    """Core: 2x2 accuracy matrix for every column of X at once.

    Diagonal entries are stratified k-fold CV within a context; off-diagonal entries train on all of
    one context and score the other. Nothing is folded about 0.5.
    """
    n_out = X.shape[1]
    M = {}
    for a in CONTEXTS:
        ia = np.flatnonzero(ctx == a)
        for b in CONTEXTS:
            key = f'{a}_{b}'
            if a == b:
                folds = _folds(y, ctx, a, n_splits, seed)
                if not folds:
                    M[key] = np.full(n_out, np.nan)
                    continue
                accs = []
                for tr, te in folds:
                    if len(np.unique(y[tr])) < 2:
                        continue
                    f = _centroid_acc if method == 'fast' else None
                    if f is not None:
                        accs.append(f(X[tr], y[tr], X[te], y[te],
                                      None if w is None else w[tr], None if w is None else w[te]))
                    else:
                        accs.append(np.array([
                            _svc_acc(X[tr, j:j + 1], y[tr], X[te, j:j + 1], y[te],
                                     None if w is None else w[tr], None if w is None else w[te], C)
                            for j in range(n_out)]))
                M[key] = np.nanmean(accs, axis=0) if accs else np.full(n_out, np.nan)
            else:
                ib = np.flatnonzero(ctx == b)
                if len(np.unique(y[ia])) < 2 or len(np.unique(y[ib])) < 2:
                    M[key] = np.full(n_out, np.nan)
                    continue
                if method == 'fast':
                    M[key] = _centroid_acc(X[ia], y[ia], X[ib], y[ib],
                                           None if w is None else w[ia],
                                           None if w is None else w[ib])
                else:
                    M[key] = np.array([
                        _svc_acc(X[np.ix_(ia, [j])], y[ia], X[np.ix_(ib, [j])], y[ib],
                                 None if w is None else w[ia], None if w is None else w[ib], C)
                        for j in range(n_out)])
    return M


def _summarise(M):
    w1, w2 = M['1_1'], M['2_2']
    within = np.nanmean([w1, w2], axis=0)
    ccgp = np.nanmean([M['1_2'], M['2_1']], axis=0)
    return {'sn_within_1': w1, 'sn_within_2': w2, 'sn_within': within,
            'snCCGP': ccgp, 'snGap': within - ccgp, 'acc_matrix': M}
