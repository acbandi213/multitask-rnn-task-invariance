"""Performance and basic structure of one set of trained networks (the checks of Yang et al. 2019).

The other read-outs need `perf.csv` from here (which network performs which task), so run this first.

For one training run (10 networks):
    performance per task              n_perf training-distribution trials per task, noise on, Yang's rule
    hemifield choice accuracy         analysis-setting trials (grid directions, fixed timing), noise on: the response ring's direction
                                      in the response epoch, read as right / left; the binary choice the TI / TD lens needs
    task variance (TV)                noise off, analysis setting; for unit i and task A the variance across trials at each time
                                      step from the first stimulus onset on, averaged over time (Yang Methods)
    clusters                          active units (summed TV > 1e-3), TV normalised by each unit's max over tasks, k-means
                                      k = 2 ... 30, k chosen by the silhouette score among k >= 5 (see K_MIN_BEST)
    cluster lesions                   performance change per task when one cluster is silenced (Yang Fig. 2e)
    FTV                               (TV_A - TV_B) / (TV_A + TV_B) over active units for Yang's Fig. 4 pairs and the anchors

    python -m mtrnn.evaluate main_relu [--device cuda|mps|cpu]
writes results/performance/<run>/: perf.csv, hemi.csv, tv.npz, silhouette.csv, clusters.csv, lesion.csv, ftv.csv, tsne.csv, provenance.csv
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from .tasks import ANCHOR_PAIRS, MOUSE_TASKS, hemifield, popvec, ring_slice, resp_ring_rates, RESP_MS, GRACE, st
from .paths import PERFORMANCE, pick_device
from .train import RUNS, load_nets, run_trials, perf_by_task, eval_set, provenance_check

OUT = PERFORMANCE                     # results/performance/<run>/
ACTIVE_TV = 1e-3
K_MIN_BEST = 5            # best k is chosen among k >= 5: the silhouette curve is flat from k ~ 8 on and k = 2 (units active in the DM
                          # family vs the rest) often edges it out, which says nothing about task families (Yang: chose over 2 ... 30, peak ~15)
YANG_PAIRS = [('DM 1', 'Anti', 'disjoint'), ('Dly DM 1', 'DM 1', 'inclusive'), ('DM 1', 'Ctx DM 1', 'mixed'),
              ('Ctx DM 1', 'Ctx DM 2', 'disjoint-equal'), ('DMC', 'DNMC', 'disjoint-mixed')]
FTV_PAIRS = YANG_PAIRS + [(a, b, f'anchor: {k}') for k, (a, b) in ANCHOR_PAIRS.items()]


def task_variance(net, bat, device, n_per_task=240, seed=7):
    """(K, n_tasks, N) noise-free task variance, analysis setting."""
    K, N = net.K, net.N
    TV = np.zeros((K, bat.n_rules, N))
    rng = np.random.default_rng([seed, 99])
    for t in range(bat.n_rules):
        P = bat.sample(rng, n_per_task, task=t, test=True)
        assert len(np.unique(P['T'])) == 1
        _, R = run_trials(net, bat, P, device, noise=False, keep_rates=True, sigma_in=0.0, chunk=n_per_task)
        t0 = int(P['ev_on'][:, 0].min())
        TV[:, t] = R[:, :, t0:].var(1).mean(1)                     # var over trials, mean over time
    return TV


def cluster_tv(TV, k_range=range(2, 31), seed=0):
    """One network's TV (n_tasks, N) -> active units, labels, silhouette per k, best k, normalised TV."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    active = np.flatnonzero(TV.sum(0) > ACTIVE_TV)
    X = (TV[:, active] / TV[:, active].max(0, keepdims=True)).T              # (n_active, n_tasks)
    sil, labs = {}, {}
    for k in k_range:
        if k >= len(active):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labs[k] = km.labels_; sil[k] = silhouette_score(X, km.labels_)
    kbest = max((k for k in sil if k >= K_MIN_BEST), key=sil.get)
    lab = labs[kbest]
    # order clusters by the task with the highest mean normalised TV (then by that value), as in Yang's display
    peak = np.array([X[lab == c].mean(0).argmax() for c in range(kbest)])
    order = np.lexsort((-np.array([X[lab == c].mean(0).max() for c in range(kbest)]), peak))
    remap = np.empty(kbest, int); remap[order] = np.arange(kbest)
    return active, remap[lab], sil, kbest, X


def ftv(TV, a, b, active):
    ta, tb = TV[a, active], TV[b, active]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ta + tb > 0, (ta - tb) / (ta + tb), np.nan)


def hemifield_accuracy(net, bat, device, tasks, n_per_task=200, seed=11):
    """Analysis-setting trials, noise on: per task the fraction of respond trials whose response-epoch direction (mean ring
    activity from 100 ms after the response cue to the end) is in the correct hemifield; plus Yang performance on the same trials."""
    rng = np.random.default_rng([seed, 5])
    rows = []
    for name in tasks:
        P = bat.sample(rng, n_per_task, task=name, test=True)
        Z, _ = run_trials(net, bat, P, device, noise=True)
        pt, ok = perf_by_task(Z, P, bat.n_rules)
        m = P['respond']
        t1 = P['resp_on'] + GRACE
        for k in range(net.K):
            ring = np.stack([resp_ring_rates(Z[k, i, t1[i]:P['T'][i]], P['resp_ring'][i], P['resp_share'][i]).mean(0) for i in np.flatnonzero(m)])
            ch = hemifield(popvec(ring)); truth = hemifield(P['resp_dir'][m])
            rows.append(dict(seed=net.seeds[k], task=name, hemi_acc=float((ch == truth).mean()), perf=float(ok[k].mean()),
                             p_right=float(truth.mean()), n=int(m.sum())))
    return pd.DataFrame(rows)


@torch.no_grad()
def cluster_lesions(net, bat, device, labels_by_net, E, base_perf):
    """labels_by_net: list of (active, labels) per network. For c = 0 .. max k - 1 silence each network's own cluster c at once."""
    K, N = net.K, net.N
    kmax = max(l.max() + 1 for _, l in labels_by_net)
    rows = []
    for c in range(kmax):
        mask = np.ones((K, N), np.float32)
        has = np.zeros(K, bool)
        for k, (act, lab) in enumerate(labels_by_net):
            if (lab == c).any():
                mask[k, act[lab == c]] = 0; has[k] = True
        Z, _ = run_trials(net, bat, E, device, noise=True, silence=torch.as_tensor(mask, device=device))
        pt, _ = perf_by_task(Z, E, bat.n_rules)
        for k in np.flatnonzero(has):
            for t, name in enumerate(bat.names):
                rows.append(dict(seed=net.seeds[k], cluster=c, n_units=int((labels_by_net[k][1] == c).sum()), task=name,
                                 perf=float(pt[k, t]), dperf=float(pt[k, t] - base_perf[k, t])))
    return pd.DataFrame(rows)


def run(name, device=None, n_perf=200, tsne_seed_index=0, tag='model'):
    device = pick_device(device)
    rd = os.path.join(RUNS, name); od = os.path.join(OUT, name if tag == 'model' else f'{name}__{tag}'); os.makedirs(od, exist_ok=True)
    t0 = time.time()
    dev = torch.device(device)
    prov = pd.DataFrame(provenance_check(rd, tag=tag, device=device)); prov.to_csv(os.path.join(od, 'provenance.csv'), index=False)
    print('provenance ok:', prov.ok.all(), flush=True)
    net, bat, cfg, cks = load_nets(rd, tag=tag, device=device)
    seeds = net.seeds
    # performance per task
    E = eval_set(bat, n_perf, seed=777)
    Z, _ = run_trials(net, bat, E, dev, noise=True)
    pt, _ = perf_by_task(Z, E, bat.n_rules)
    pd.DataFrame([dict(seed=s, task=t, perf=pt[k, j]) for k, s in enumerate(seeds) for j, t in enumerate(bat.names)]).to_csv(
        os.path.join(od, 'perf.csv'), index=False)
    print(f'perf: mean {pt.mean():.3f}, min task per net {np.round(pt.min(1), 3)}', flush=True)
    # hemifield accuracy (anchors + Go / Anti / DM 1 for reference)
    tasks = list(dict.fromkeys([t for p in ANCHOR_PAIRS.values() for t in p] + ['Go', 'Anti', 'Dly Go', 'DM 1', 'Ctx DM 1']
                               + [t for t in MOUSE_TASKS if t in bat.names]))
    hemifield_accuracy(net, bat, dev, tasks).to_csv(os.path.join(od, 'hemi.csv'), index=False)
    # task variance and clusters
    TV = task_variance(net, bat, dev)
    np.savez_compressed(os.path.join(od, 'tv.npz'), TV=TV, names=np.array(bat.names), seeds=np.array(seeds))
    sil_rows, cl_rows, ftv_rows, labs = [], [], [], []
    for k, s in enumerate(seeds):
        active, lab, sil, kbest, X = cluster_tv(TV[k])
        labs.append((active, lab))
        sil_rows += [dict(seed=s, k=kk, silhouette=v, best=kk == kbest) for kk, v in sil.items()]
        cl_rows.append(pd.DataFrame(dict(seed=s, unit=active, cluster=lab, k=kbest)))
        for a, b, kind in FTV_PAIRS:
            f = ftv(TV[k], bat.index(a), bat.index(b), active)
            ftv_rows.append(pd.DataFrame(dict(seed=s, task_a=a, task_b=b, kind=kind, unit=active, ftv=f)))
        if k == tsne_seed_index:
            from sklearn.manifold import TSNE
            Y = TSNE(n_components=2, method='exact', learning_rate=100, perplexity=30, random_state=0).fit_transform(X)
            pd.DataFrame(dict(seed=s, unit=active, cluster=lab, x=Y[:, 0], y=Y[:, 1])).to_csv(os.path.join(od, 'tsne.csv'), index=False)
        print(f'  seed {s}: {len(active)} active units, best k = {kbest} (silhouette {sil[kbest]:.3f})', flush=True)
    pd.DataFrame(sil_rows).to_csv(os.path.join(od, 'silhouette.csv'), index=False)
    pd.concat(cl_rows).to_csv(os.path.join(od, 'clusters.csv'), index=False)
    pd.concat(ftv_rows).to_csv(os.path.join(od, 'ftv.csv'), index=False)
    # cluster lesions on a smaller evaluation set
    E2 = eval_set(bat, 100, seed=778)
    Z2, _ = run_trials(net, bat, E2, dev, noise=True)
    base, _ = perf_by_task(Z2, E2, bat.n_rules)
    cluster_lesions(net, bat, dev, labs, E2, base).to_csv(os.path.join(od, 'lesion.csv'), index=False)
    json.dump(dict(run=name, seconds=time.time() - t0, cfg=cfg), open(os.path.join(od, 'readout.json'), 'w'), indent=1)
    print(f'done {name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--device', default=None); p.add_argument('--tag', default='model')
    a = p.parse_args()
    run(a.name, a.device, tag=a.tag)
