"""The stored results: complete, consistent with reference/, and reproduced by the code where that is cheap."""
import os

import numpy as np
import pandas as pd
import pytest

from mtrnn import paths, panels as EC, readout as Q
from mtrnn.panels import robustness as R

NOTEBOOK_PANELS = ['performance', 'index_reliability', 'tail_size', 'softplus', 'pct_ti_within_type', 'noise_sweep', 'noise_sweep_pre_2_1',
                   'pursuit_dial', 'unit_removal', 'stim_vs_response', 'rule_weights', 'recurrent_weights', 'task_decoding_epochs',
                   'task_decoding', 'training_trajectory', 'silencing']


def capture(monkeypatch, module):
    out = {}
    monkeypatch.setattr(module, 'save', lambda name, obj: out.__setitem__(name, obj))
    return out


def test_everything_the_notebooks_read_is_there(results):
    for p in NOTEBOOK_PANELS:
        assert os.path.exists(os.path.join(paths.PANELS, f'{p}.pkl')), p
    for f in ('units.parquet', 'pairs.parquet', 'heldout_pairs_p90.parquet'):
        assert os.path.exists(os.path.join(paths.READOUT, 'main_relu', f)), f
    assert len(Q.task_perf('main_relu')) == 10 * 27


def test_reference_values_are_the_ones_in_the_stored_panels(results):
    for name, window in (('noise_sweep', 'fig3f'), ('noise_sweep_pre_2_1', 'pre_2_1')):
        d = EC.load(name); ref = R._three_point_data(window)
        for sp in ('mouse', 'macaque'):
            for p in ('TD', 'random', 'TI'):
                assert d[sp]['data'][p] == pytest.approx(ref[sp][p], abs=1e-12), (name, sp, p)


def test_performance_panel_is_reproduced(results, features, monkeypatch):
    out = capture(monkeypatch, R); R.panel_performance()
    old = EC.load('performance'); new = out['performance']
    pd.testing.assert_frame_equal(new['perf'].reset_index(drop=True), old['perf'].reset_index(drop=True))
    assert new['dropped'] == old['dropped']


def test_pursuit_dial_panel_is_reproduced(results, monkeypatch):
    out = capture(monkeypatch, R); R.panel_pursuit_dial(EC.PAPER_WINDOW)
    old = EC.load('pursuit_dial'); new = out['pursuit_dial']
    pd.testing.assert_frame_equal(new['table'], old['table'])
    assert new['best_g'] == pytest.approx(old['best_g']) and new['obs_k'] == old['obs_k']


def test_noise_sweep_refit_is_reproduced(results, monkeypatch):
    out = capture(monkeypatch, R); R.panel_noise_sweep_retarget('pre_2_1')
    old = EC.load('noise_sweep_pre_2_1'); new = out['noise_sweep_pre_2_1']
    for sp in ('mouse', 'macaque'):
        assert new[sp]['best_sigma'] == pytest.approx(old[sp]['best_sigma'], abs=1e-12)


@pytest.mark.slow
def test_weight_panels_are_reproduced_from_the_networks(results, monkeypatch):
    """Rule-input weights (ED 8e) depend only on the networks and the stored unit table: recomputed exactly."""
    from mtrnn.panels import roles
    out = capture(monkeypatch, roles); roles.panel_rule_weights()
    pd.testing.assert_frame_equal(out['rule_weights']['table'], EC.load('rule_weights')['table'])


@pytest.mark.slow
def test_tail_size_panel_is_reproduced_from_the_features(results, features, monkeypatch, tmp_path):
    """Held-out TI / TD labels and the three-point decoding at 10 / 20 / 30 % tails (ED 7c), from the stored unit features."""
    from mtrnn.panels import common
    old = EC.load('tail_size')
    monkeypatch.setattr(common, 'CACHE', str(tmp_path))            # recompute the held-out labels instead of reading a cached copy
    monkeypatch.setattr(common, 'save', lambda name, obj: None)
    out = capture(monkeypatch, R); R.panel_tail_size()
    new = out['tail_size']
    for prop in old:
        a, b = new[prop]['table'], old[prop]['table']
        num = a.select_dtypes('number').columns
        np.testing.assert_allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=0, atol=1e-10, equal_nan=True)
