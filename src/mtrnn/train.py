"""Train K networks of one condition together on the extended battery (Yang's interleaved regime).

Every network has its own trial stream (numpy rng seeded by its seed) and private noise; the loss is a sum over networks; gradients
are clipped element-wise to [-1, 1] (Yang's reference code) which acts on each network separately; Adam is element-wise. So a
network trains as it would alone. Mini-batches mix tasks (Ctx DM 1 / 2 five times as frequent, as in Yang); trials are padded to
the longest one in the batch and padded steps carry zero loss.

Loss per network = sum(mask (z - z_hat)^2) / sum(mask) + l1_rate mean|r| + l2_rate mean r^2 + l1_weight sum|W| + l2_weight sum W^2.
Default l1_rate = 1e-4 (one of the levels of Yang's sweep). Without any rate penalty, Adam's sign-normalised first updates of W_rec are
rank-one and positive (every rate is >= 0, so dL/dW_ij = delta_i r_j has the sign of delta_i along a whole row); within ~10 iterations
this adds a uniform positive mode with eigenvalue > 1 and the rates explode (ReLU under diag or orth init, at lr 1e-3 and 1e-4;
Softplus under diag init). A small rate penalty prevents it for both activations (pilot 2026-10-19).

Evaluation every `eval_every` iterations: a fixed set of training-distribution trials (n_eval per task, noise on), Yang's performance
rule per task. Stops when every network has min-over-tasks performance >= `criterion` on `crit_consecutive` consecutive
evaluations (or at max_iters). Checkpoints at `ckpt_iters` and at the end.

Provenance: every model.pt stores the hash of its config and its noise-free performance on a
regenerated fixed analysis set; `provenance_check(run_dir)` reloads each checkpoint, re-hashes, regenerates and compares.

    python -m mtrnn.train --config configs/main_relu.json            # the networks of Figure 5
    python -m mtrnn.train --name pilot --act softplus --iters 300 --seeds 0 1 2
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from .model import BatchedYangRNN
from .paths import MODELS, pick_device
from .tasks import Battery, render, performance

RUNS = MODELS                          # models/<run>/

DEFAULTS = dict(
    layout='separate', tasks=None, ctx_weight=5.0, easy_dm=False, dms_min_offset_deg=10.0, mask_pre_ms=0, pursuit_share=None, mouse_tasks=False,
    n_units=256, tau_ms=100.0, dt_ms=20.0, act='softplus', init='diag', rec_gain=0.5, sigma_rec=0.05, sigma_in=0.01,
    batch=64, lr=1e-3, clip_value=1.0, l1_rate=1e-4, l2_rate=0.0, l1_weight=0.0, l2_weight=0.0,
    max_iters=40000, eval_every=500, n_eval=64, criterion=0.95, crit_consecutive=2, stop_at_criterion=True,
    ckpt_iters=[1000, 2000, 5000, 10000, 20000, 40000],
)
HASH_EXCLUDE = ('seeds', 'device', 'name', 'max_iters', 'ckpt_iters', 'stop_at_criterion', 'eval_every')


def full_config(cfg):
    c = copy.deepcopy(DEFAULTS); c.update(cfg); return c


def config_hash(cfg):
    c = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDE}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cat_params(bs):
    out = {}
    for k in bs[0]:
        if k == 'meta':
            keys = set().union(*[b['meta'].keys() for b in bs])
            out['meta'] = {m: np.concatenate([b['meta'].get(m, np.full(len(b['T']), np.nan)) for b in bs]) for m in keys}
        else:
            out[k] = np.concatenate([b[k] for b in bs])
    return out


def build(cfg, seeds):
    bat = Battery(cfg['layout'], cfg['tasks'], cfg['ctx_weight'], cfg.get('easy_dm', False), cfg.get('dms_min_offset_deg', 10.0),
                  cfg.get('pursuit_share'), cfg.get('mouse_tasks', False))
    net = BatchedYangRNN(bat.n_inputs, bat.n_outputs, seeds, n_units=cfg['n_units'], tau_ms=cfg['tau_ms'], dt_ms=cfg['dt_ms'],
                         act=cfg['act'], init=cfg['init'], rec_gain=cfg['rec_gain'], sigma_rec=cfg['sigma_rec'])
    return bat, net


def loss_k(net, Z, R, d, cfg):
    l = ((Z - d['targets']) ** 2 * d['mask']).sum((1, 2, 3)) / d['mask'].sum((1, 2, 3))
    if cfg['l1_rate']:
        l = l + cfg['l1_rate'] * R.abs().mean((1, 2, 3))
    if cfg['l2_rate']:
        l = l + cfg['l2_rate'] * (R ** 2).mean((1, 2, 3))
    W = (net.W_in, net.W_rec, net.W_out)
    if cfg['l1_weight']:
        l = l + cfg['l1_weight'] * sum(w.abs().sum((1, 2)) for w in W)
    if cfg['l2_weight']:
        l = l + cfg['l2_weight'] * sum((w ** 2).sum((1, 2)) for w in W)
    return l


def reshape_k(d, K):
    return {k: (v.reshape(K, -1, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in d.items()}


@torch.no_grad()
def run_trials(net, bat, params, device, noise=True, chunk=1024, silence=None, rule_pattern=None, keep_rates=False, sigma_in=0.01,
               w_rec_mask=None):
    """The same trial set through every network. Returns z (K, n, T, O) [and rates (K, n, T, N) if keep_rates] as numpy, plus the
    rendered valid / T arrays. Trials are processed in chunks (padded to the longest trial in the whole set)."""
    n = len(params['T']); Tm = int(params['T'].max())
    Zs, Rs = [], []
    for s in range(0, n, chunk):
        sub = {k: (v[s:s + chunk] if k != 'meta' else None) for k, v in params.items()}
        rp = None if rule_pattern is None else rule_pattern[s:s + chunk]
        d = render(sub, bat.n_rules, device, sigma_in=sigma_in, alpha=net.alpha, noise=noise, T_pad=Tm, rule_pattern=rp,
                   **getattr(bat, 'render_kw', {}))
        u = d['inputs'][None].expand(net.K, *d['inputs'].shape)
        R, Z = net(u, noise=noise, silence=silence, keep_rates=keep_rates, w_rec_mask=w_rec_mask)
        Zs.append(Z.cpu().numpy())
        if keep_rates:
            Rs.append(R.cpu().numpy())
    Z = np.concatenate(Zs, 1)
    return (Z, np.concatenate(Rs, 1)) if keep_rates else (Z, None)


def perf_by_task(Z, params, n_tasks):
    """Z (K, n, T, O) -> (K, n_tasks) Yang performance per task, and (K, n) per-trial correctness."""
    last = params['T'] - 1
    ok = np.stack([performance(Z[k, np.arange(len(last)), last], params['respond'], params['resp_dir'], params['resp_ring'],
                               params.get('resp_share'))
                   for k in range(Z.shape[0])])
    rule = params['rule']
    return np.stack([[ok[k, rule == t].mean() if (rule == t).any() else np.nan for t in range(n_tasks)] for k in range(Z.shape[0])]), ok


def eval_set(bat, n_per_task, seed=2019, test=False):
    rng = np.random.default_rng([seed, 31])
    tasks = np.repeat(np.arange(bat.n_rules), n_per_task)
    return bat.sample(rng, len(tasks), tasks=tasks, test=test)


def provenance_perf(net, bat, device):
    """Noise-free performance per task on the fixed analysis set (32 trials per task): a deterministic fingerprint."""
    P = eval_set(bat, 32, seed=4242, test=True)
    Z, _ = run_trials(net, bat, P, device, noise=False)
    return perf_by_task(Z, P, bat.n_rules)[0]


def save_checkpoints(net, bat, cfg, seeds, run_dir, it, hist, crit_iter, device, tag):
    prov = provenance_perf(net, bat, device)
    h = config_hash(cfg)
    for k, s in enumerate(seeds):
        o = os.path.join(run_dir, f'seed{s}'); os.makedirs(o, exist_ok=True)
        sd = {name: getattr(net, name)[k].detach().cpu().clone() for name in ('W_in', 'W_rec', 'b', 'W_out', 'b_out')}
        ck = dict(state=sd, cfg=cfg, seed=int(s), iter=int(it), tasks=bat.names, layout=bat.layout, cfg_hash=h,
                  prov_perf=prov[k].tolist(), crit_iter=crit_iter[k], history=[e[k] for e in hist],
                  written=time.time(), pid=os.getpid())
        torch.save(ck, os.path.join(o, f'{tag}.pt'))


def load_nets(run_dir, seeds=None, tag='model', device='cpu'):
    """-> (batched net, battery, cfg, list of checkpoints)."""
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    seeds = cfg['seeds'] if seeds is None else seeds
    cks = [torch.load(os.path.join(run_dir, f'seed{s}', f'{tag}.pt'), map_location='cpu', weights_only=False) for s in seeds]
    bat, net = build(cks[0]['cfg'], seeds)
    with torch.no_grad():
        for name in ('W_in', 'W_rec', 'b', 'W_out', 'b_out'):
            getattr(net, name).copy_(torch.stack([c['state'][name] for c in cks]))
    return net.to(device), bat, cks[0]['cfg'], cks


def provenance_check(run_dir, tag='model', device='cpu', tol_mean=0.01, tol_max=0.07):
    """For every seed: stored config hash == hash of run config.json; regenerated noise-free fingerprint == stored one."""
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    h = config_hash(cfg)
    net, bat, _, cks = load_nets(run_dir, tag=tag, device=device)
    prov = provenance_perf(net, bat, device)
    rows = []
    for k, c in enumerate(cks):
        dif = np.abs(prov[k] - np.asarray(c['prov_perf']))
        rows.append(dict(seed=c['seed'], iter=c['iter'], hash_ok=c['cfg_hash'] == h, mean_diff=float(dif.mean()), max_diff=float(dif.max()),
                         ok=bool(c['cfg_hash'] == h and dif.mean() <= tol_mean and dif.max() <= tol_max)))
    return rows


def train(cfg, seeds, run_dir, device=None, verbose=True):
    device = pick_device(device)
    cfg = full_config(cfg); cfg['seeds'] = [int(s) for s in seeds]
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    dev = torch.device(device)
    torch.manual_seed(1000 * int(seeds[0]) + 19)
    bat, net = build(cfg, seeds)
    net.to(dev)
    K, B = net.K, cfg['batch']
    rngs = [np.random.default_rng([int(s), 2019]) for s in seeds]
    E = eval_set(bat, cfg['n_eval'])
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    hist, crit_iter, streak = [], [None] * K, [0] * K
    t0 = time.time(); t_train = 0.0; n_steps = 0
    log = open(os.path.join(run_dir, 'train_log.txt'), 'a')
    for it in range(1, cfg['max_iters'] + 1):
        ts = time.time()
        P = cat_params([bat.sample(r, B) for r in rngs])
        d = reshape_k(render(P, bat.n_rules, dev, sigma_in=cfg['sigma_in'], alpha=net.alpha, pre_on=int(round(cfg.get('mask_pre_ms', 0) / cfg['dt_ms'])),
                             **bat.render_kw), K)
        R, Z = net(d['inputs'])
        lk = loss_k(net, Z, R, d, cfg)
        opt.zero_grad(set_to_none=True)
        lk.sum().backward()
        if cfg['clip_value']:
            for p in net.parameters():
                p.grad.clamp_(-cfg['clip_value'], cfg['clip_value'])
        opt.step()
        if dev.type == 'mps':
            torch.mps.synchronize()
        t_train += time.time() - ts; n_steps += int(P['T'].max())
        if it % cfg['eval_every'] == 0 or it == cfg['max_iters']:
            Zv, _ = run_trials(net, bat, E, dev, noise=True, sigma_in=cfg['sigma_in'])
            pt, _ = perf_by_task(Zv, E, bat.n_rules)
            ev = [dict(iter=it, trials=it * B, loss=float(lk[k].item()), max_rate=float(R[k].max().item()), mean_rate=float(R[k].mean().item()), perf=pt[k].tolist(), perf_min=float(pt[k].min()),
                       perf_mean=float(pt[k].mean()), sec=time.time() - t0) for k in range(K)]
            hist.append(ev)
            for k in range(K):
                streak[k] = streak[k] + 1 if pt[k].min() >= cfg['criterion'] else 0
                if crit_iter[k] is None and streak[k] >= cfg['crit_consecutive']:
                    crit_iter[k] = it - (cfg['crit_consecutive'] - 1) * cfg['eval_every']
            worst = [bat.names[i] for i in np.argsort(pt.mean(0))[:3]]
            msg = (f"{it:6d} loss {lk.mean().item():.4f} perf mean {pt.mean():.3f} min-task (median net) {np.median(pt.min(1)):.3f} "
                   f"max rate {R.max().item():.1f} nets at criterion {sum(c is not None for c in crit_iter)}/{K} worst {worst} "
                   f"{t_train / it:.3f} s/it {time.time() - t0:.0f}s")
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
            with open(os.path.join(run_dir, 'history.json'), 'w') as f:
                json.dump(dict(names=bat.names, hist=hist, crit_iter=crit_iter, sec_per_iter=t_train / it, mean_T=n_steps / it), f)
            if cfg['stop_at_criterion'] and all(c is not None for c in crit_iter):
                break
        if it in cfg['ckpt_iters']:
            save_checkpoints(net, bat, cfg, seeds, run_dir, it, hist, crit_iter, dev, f'ckpt_{it:06d}')
    save_checkpoints(net, bat, cfg, seeds, run_dir, it, hist, crit_iter, dev, 'model')
    timing = dict(iters=it, seconds_total=time.time() - t0, sec_per_iter=t_train / it, mean_T_steps=n_steps / it, K=K, batch=B,
                  device=device, crit_iter=crit_iter)
    with open(os.path.join(run_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f, indent=1)
    log.close()
    return net, bat, hist, timing


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Train one set of networks (same settings, one network per seed) together.')
    p.add_argument('--config', default=None, help='JSON file with name, seeds and any training setting (configs/*.json); '
                                                 'command-line flags override it')
    p.add_argument('--name', default=None); p.add_argument('--act', default=None); p.add_argument('--layout', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None); p.add_argument('--iters', type=int, default=None)
    p.add_argument('--batch', type=int, default=None); p.add_argument('--device', default=None); p.add_argument('--eval_every', type=int, default=None)
    p.add_argument('--n_units', type=int, default=None); p.add_argument('--no_stop', action='store_true'); p.add_argument('--easy_dm', action='store_true'); p.add_argument('--ref_code', action='store_true')
    p.add_argument('--l2_rate', type=float, default=None); p.add_argument('--l1_rate', type=float, default=None)
    p.add_argument('--pursuit_share', type=float, default=None); p.add_argument('--mouse_tasks', action='store_true')
    p.add_argument('--l1_weight', type=float, default=None); p.add_argument('--lr', type=float, default=None)
    p.add_argument('--out', default=None, help='parent folder of the run (default: models/)')
    a = p.parse_args(argv)
    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    name = a.name or cfg.pop('name', None); cfg.pop('name', None)
    seeds = a.seeds or cfg.pop('seeds', None) or list(range(10)); cfg.pop('seeds', None)
    if name is None:
        p.error('give --name or a --config with a "name"')
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        p.error(f'unknown settings in {a.config}: {sorted(unknown)}')
    for k in ('act', 'layout', 'batch', 'eval_every', 'n_units', 'l2_rate', 'l1_rate', 'l1_weight', 'lr'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.iters:
        cfg['max_iters'] = a.iters
    if a.easy_dm:
        cfg['easy_dm'] = True
    if a.ref_code:                      # reference-code values where it differs from the Methods
        cfg.update(dms_min_offset_deg=20.0, mask_pre_ms=100)
    if a.pursuit_share is not None:
        cfg['pursuit_share'] = a.pursuit_share
    if a.mouse_tasks:
        cfg['mouse_tasks'] = True
    if a.no_stop:
        cfg['stop_at_criterion'] = False
    run_dir = os.path.join(a.out or RUNS, name)
    if os.path.exists(os.path.join(run_dir, 'seed%d' % seeds[0], 'model.pt')):
        p.error(f'{run_dir} already holds trained networks; choose another --name (or --out) so they are not overwritten')
    train(cfg, seeds, run_dir, device=a.device)


if __name__ == '__main__':
    main()
