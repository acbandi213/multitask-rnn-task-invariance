"""The seven sets of trained networks: each file is what its config says, and the configs in configs/ would retrain them."""
import json
import os

import pytest

from conftest import ROOT, RUNS
from mtrnn import paths
from mtrnn.train import provenance_check, full_config, config_hash, HASH_EXCLUDE


@pytest.mark.parametrize('run', RUNS)
def test_provenance(run):
    """Stored config hash equals the hash of the run's config.json, and the noise-free performance fingerprint is regenerated."""
    rows = provenance_check(os.path.join(paths.MODELS, run), device='cpu')
    assert len(rows) == 10 and all(r['ok'] for r in rows), rows


def test_checkpoints_of_the_main_run():
    for it in (1000, 2000, 5000, 10000, 20000):
        assert all(r['ok'] for r in provenance_check(os.path.join(paths.MODELS, 'main_relu'), tag=f'ckpt_{it:06d}', device='cpu'))


@pytest.mark.parametrize('run', RUNS)
def test_config_file_retrains_the_same_settings(run):
    with open(os.path.join(ROOT, 'configs', f'{run}.json')) as f:
        c = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    assert c.pop('name') == run
    seeds = c.pop('seeds')
    new = full_config(c)
    with open(os.path.join(paths.MODELS, run, 'config.json')) as f:
        old = json.load(f)
    assert seeds == old['seeds']
    for k, v in old.items():                       # every stored setting is reproduced
        if k not in HASH_EXCLUDE:
            assert new[k] == v, (k, new[k], v)
    for k in set(new) - set(old):                  # settings added after the main runs were trained are at their neutral default
        assert new[k] in (None, False), (k, new[k])
    if set(new) - {'seeds'} == set(old) - {'seeds'}:
        assert config_hash(new) == config_hash(old)
