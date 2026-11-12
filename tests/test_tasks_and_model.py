"""The task battery and the network: shapes, determinism, and that a short training run executes."""
import numpy as np
import torch

from mtrnn.tasks import Battery, render
from mtrnn.train import build, full_config, train, load_nets, provenance_check


def test_battery_has_27_tasks():
    bat = Battery('shared', None, 5.0, True, 20.0, None, False)
    assert len(bat.names) == 27 and len(set(bat.names)) == 27
    for t in ('Go', 'Anti', 'Ctx DM 1', 'DMS', 'Go Mod1', 'Go Mod2', 'Sac', 'Pursuit'):
        assert t in bat.names


def test_trials_are_reproducible_from_the_seed():
    bat = Battery('shared', None, 5.0, True, 20.0, None, False)
    a = bat.sample(np.random.default_rng(3), 32); b = bat.sample(np.random.default_rng(3), 32)
    for k in a:
        if k != 'meta':
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k
    d = render(a, bat.n_rules, 'cpu', sigma_in=0.0, alpha=0.2, noise=False, **bat.render_kw)
    assert d['inputs'].shape[0] == 32 and d['inputs'].shape[2] == bat.n_inputs
    assert d['targets'].shape[2] == bat.n_outputs and torch.isfinite(d['inputs']).all()


def test_network_forward_and_initialisation():
    cfg = full_config(dict(act='relu', layout='shared', easy_dm=True, n_units=32))
    bat, net = build(cfg, [0, 1])
    _, net2 = build(cfg, [1])
    assert torch.equal(net.W_in[1], net2.W_in[0])          # a network depends on its own seed only
    u = torch.zeros(2, 4, 10, bat.n_inputs)
    R, Z = net(u, noise=False)
    assert R.shape == (2, 4, 10, 32) and Z.shape == (2, 4, 10, bat.n_outputs)


def test_short_training_run(tmp_path):
    run = str(tmp_path / 'tiny')
    train(dict(act='relu', layout='shared', easy_dm=True, n_units=32, max_iters=4, eval_every=2, n_eval=2, batch=8, ckpt_iters=[2]),
          [0, 1], run, device='cpu')
    net, bat, cfg, cks = load_nets(run)
    assert net.K == 2 and cks[0]['iter'] == 4
    assert all(r['ok'] for r in provenance_check(run))
    assert all(r['ok'] for r in provenance_check(run, tag='ckpt_000002'))
